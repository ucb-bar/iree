// Copyright 2024 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _GNU_SOURCE 1 

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// IREE Core Headers
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"
#include "iree/modules/hal/module.h" 
#include "iree/modules/hal/types.h"

// IREE Task System & Loader Headers
#include "iree/task/api.h"
#include "iree/task/topology.h"
#include "iree/hal/drivers/local_task/task_device.h" 
#include "iree/hal/local/loaders/registration/init.h" 

// ============================================================================
// 1. File Loading Helper
// ============================================================================

iree_status_t read_file_contents(const char* path, iree_allocator_t allocator, iree_const_byte_span_t* out_span) {
  FILE* file = fopen(path, "rb");
  if (!file) return iree_make_status(IREE_STATUS_NOT_FOUND, "failed to open file: %s", path);

  fseek(file, 0, SEEK_END);
  size_t file_size = ftell(file);
  fseek(file, 0, SEEK_SET);

  uint8_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator, file_size, (void**)&buffer));

  if (fread(buffer, 1, file_size, file) != file_size) {
    fclose(file);
    iree_allocator_free(allocator, buffer);
    return iree_make_status(IREE_STATUS_DATA_LOSS, "failed to read file: %s", path);
  }

  fclose(file);
  *out_span = iree_make_const_byte_span(buffer, file_size);
  return iree_ok_status();
}

// ============================================================================
// 2. Topology Helpers
// ============================================================================

void iree_task_topology_initialize_from_mask(iree_task_topology_t* out_topology, uint64_t mask) {
  iree_task_topology_initialize(out_topology);
  uint32_t cpu_ids[64];
  iree_host_size_t count = 0;
  for (int i = 0; i < 64; ++i) {
    if ((mask >> i) & 1) cpu_ids[count++] = (uint32_t)i;
  }
  if (count > 0) {
    iree_task_topology_initialize_from_logical_cpu_set(count, cpu_ids, out_topology);
  }
}

// ============================================================================
// 3. Device Creation
// ============================================================================

iree_status_t create_device_with_mask(iree_allocator_t host_allocator,
                                      uint64_t core_mask,
                                      const char* label,
                                      iree_hal_device_t** out_device) {
  iree_task_topology_t topology;
  iree_task_topology_initialize_from_mask(&topology, core_mask);

  iree_task_executor_options_t options;
  iree_task_executor_options_initialize(&options);
  options.worker_local_memory_size = 64 * 1024;

  iree_task_executor_t* executor = NULL;
  IREE_RETURN_IF_ERROR(iree_task_executor_create(options, &topology, host_allocator, &executor));

  iree_host_size_t loader_count = 0;
  iree_hal_executable_loader_t* loaders[8] = {NULL};
  IREE_RETURN_IF_ERROR(iree_hal_create_all_available_executable_loaders(
      NULL, IREE_ARRAYSIZE(loaders), &loader_count, loaders, host_allocator));

  fprintf(stderr, "DEBUG: Device '%s' created with %d loaders.\n", label, (int)loader_count);

  iree_hal_allocator_t* device_allocator = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_create_heap(
      iree_make_cstring_view("local"), host_allocator, host_allocator, &device_allocator));

  iree_hal_task_device_params_t params;
  iree_hal_task_device_params_initialize(&params);
  
  iree_task_executor_t* executors[] = {executor};
  iree_status_t status = iree_hal_task_device_create(
      iree_make_cstring_view(label), &params, IREE_ARRAYSIZE(executors), executors,
      loader_count, loaders, device_allocator, host_allocator, out_device);

  iree_hal_allocator_release(device_allocator);
  for (iree_host_size_t i = 0; i < loader_count; ++i) iree_hal_executable_loader_release(loaders[i]);
  iree_task_executor_release(executor);
  iree_task_topology_deinitialize(&topology);
  
  return status;
}

// ============================================================================
// 4. Main Execution
// ============================================================================

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <path_to_vmfb> <entry_point_function>\n", argv[0]);
    return -1;
  }

  const char* module_path = argv[1];
  const char* entry_point_name = argv[2];
  iree_allocator_t host_allocator = iree_allocator_system();
  iree_vm_instance_t* instance = NULL;
  
  IREE_CHECK_OK(iree_vm_instance_create(64, host_allocator, &instance));

  // [FIX] REGISTER HAL TYPES
  // This tells the VM what "!hal.allocator", "!hal.buffer", etc. mean.
  IREE_CHECK_OK(iree_hal_module_register_all_types(instance));

  fprintf(stdout, "--- Creating Hardware Devices ---\n");

  iree_hal_device_t* device_a = NULL;
  IREE_CHECK_OK(create_device_with_mask(host_allocator, 1, "local-device-a", &device_a)); 

  iree_hal_device_t* device_b = NULL;
  IREE_CHECK_OK(create_device_with_mask(host_allocator, 32, "local-device-b", &device_b));

  iree_hal_device_t* device_ab = NULL;
  IREE_CHECK_OK(create_device_with_mask(host_allocator, 3, "local-device-ab", &device_ab));

  // --- Create HAL Module ---
  iree_hal_device_t* devices[] = {device_a, device_b, device_ab};
  iree_vm_module_t* hal_module = NULL;
  
  IREE_CHECK_OK(iree_hal_module_create(
      instance, /*device_policy=*/(iree_hal_module_device_policy_t){0},
      IREE_ARRAYSIZE(devices), devices,
      IREE_HAL_MODULE_FLAG_NONE,
      /*debug_sink=*/(iree_hal_module_debug_sink_t){0},
      host_allocator,
      &hal_module));

  // --- Load User Bytecode Module ---
  fprintf(stdout, "Loading module: %s...\n", module_path);
  iree_const_byte_span_t module_data;
  IREE_CHECK_OK(read_file_contents(module_path, host_allocator, &module_data));

  iree_vm_module_t* user_module = NULL;
  IREE_CHECK_OK(iree_vm_bytecode_module_create(
      instance, module_data, iree_allocator_null(), host_allocator, &user_module));

  // --- Create VM Context ---
  iree_vm_module_t* modules[] = {hal_module, user_module};
  iree_vm_context_t* context = NULL;
  IREE_CHECK_OK(iree_vm_context_create_with_modules(
      instance, IREE_VM_CONTEXT_FLAG_NONE,
      IREE_ARRAYSIZE(modules), modules,
      host_allocator, &context));

  // --- Resolve Function ---
  iree_vm_function_t function;
  IREE_CHECK_OK(iree_vm_module_lookup_function_by_name(
      user_module, IREE_VM_FUNCTION_LINKAGE_EXPORT,
      iree_make_cstring_view(entry_point_name), &function));

  // --- Prepare Inputs ---
  // CHANGE: Capacity is now 3 (Input + WaitFence + SignalFence)
  iree_vm_list_t* inputs = NULL;
  IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 3, host_allocator, &inputs));
  
  iree_vm_list_t* outputs = NULL;
  IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1, host_allocator, &outputs));

  const float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  const iree_hal_dim_t shape[] = {4};

  iree_hal_buffer_view_t* input_buffer = NULL;
  IREE_CHECK_OK(iree_hal_buffer_view_allocate_buffer_copy(
      device_a, iree_hal_device_allocator(device_a), IREE_ARRAYSIZE(shape), shape,
      IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      iree_make_const_byte_span(input_data, sizeof(input_data)), &input_buffer));

  // 1. Push the Input Tensor (Argument #1)
  iree_vm_ref_t input_ref = iree_hal_buffer_view_move_ref(input_buffer);
  IREE_CHECK_OK(iree_vm_list_push_ref_move(inputs, &input_ref));

  // 2. Push NULL for Wait Fence (Argument #2)
  // This tells the runtime: "Start execution immediately, don't wait for anything."
  iree_vm_ref_t null_ref = iree_vm_ref_null();
  IREE_CHECK_OK(iree_vm_list_push_ref_retain(inputs, &null_ref));

  // 3. Push NULL for Signal Fence (Argument #3)
  // This tells the runtime: "I will wait for the invocation to complete synchronously."
  IREE_CHECK_OK(iree_vm_list_push_ref_retain(inputs, &null_ref));

  // --- Invoke ---
  fprintf(stdout, "Invoking '%s'...\n", entry_point_name);
  for (int i = 0; i < 1000000; ++i) {
    // Reset output list (optional but good practice if reusing)
    // For this simple test, just invoking repeatedly is enough to spike the CPU
    IREE_CHECK_OK(iree_vm_invoke(
        context, function, IREE_VM_INVOCATION_FLAG_NONE,
        /*policy=*/NULL, inputs, outputs, host_allocator));
  }
  
  // --- Process Outputs ---
  fprintf(stdout, "Execution Complete. Reading output...\n");
  
  iree_hal_buffer_view_t* output_view = iree_vm_list_get_buffer_view_assign(outputs, 0);
  
  float result_data[4] = {0};
  IREE_CHECK_OK(iree_hal_buffer_map_read(
      iree_hal_buffer_view_buffer(output_view), 0,
      result_data, sizeof(result_data)));

  fprintf(stdout, "Result: [%.1f, %.1f, %.1f, %.1f]\n", 
          result_data[0], result_data[1], result_data[2], result_data[3]);

  // --- Cleanup ---
  iree_allocator_free(host_allocator, (void*)module_data.data);
  iree_vm_list_release(inputs);
  iree_vm_list_release(outputs);
  iree_vm_context_release(context);
  iree_vm_module_release(hal_module);
  iree_vm_module_release(user_module);
  iree_hal_device_release(device_a);
  iree_hal_device_release(device_b);
  iree_hal_device_release(device_ab);
  iree_vm_instance_release(instance);

  return 0;
}