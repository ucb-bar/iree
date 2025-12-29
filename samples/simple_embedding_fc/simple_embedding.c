// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// NOTE: <stdio.h> removed for bare-metal compliance.

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

// A function to create the HAL device from the different backend targets.
extern iree_status_t create_sample_device(iree_allocator_t host_allocator,
                                          iree_hal_device_t** out_device);

// A function to load the vm bytecode module from the different backend targets.
extern const iree_const_byte_span_t load_bytecode_module_data();

iree_status_t Run() {
  iree_vm_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_instance_create(
      IREE_VM_TYPE_CAPACITY_DEFAULT, iree_allocator_system(), &instance));
  IREE_RETURN_IF_ERROR(iree_hal_module_register_all_types(instance));

  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(create_sample_device(iree_allocator_system(), &device),
                       "create device");
  
  iree_vm_module_t* hal_module = NULL;
  // CHANGED: Use null debug sink (no stdio/stderr)
  IREE_RETURN_IF_ERROR(iree_hal_module_create(
      instance, iree_hal_module_device_policy_default(), /*device_count=*/1,
      &device, IREE_HAL_MODULE_FLAG_SYNCHRONOUS,
      iree_hal_module_debug_sink_null(), iree_allocator_system(),
      &hal_module));

  // Load bytecode module from the embedded data.
  const iree_const_byte_span_t module_data = load_bytecode_module_data();

  iree_vm_module_t* bytecode_module = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_create(
      instance, module_data, iree_allocator_null(), iree_allocator_system(),
      &bytecode_module));

  // Allocate a context that will hold the module state across invocations.
  iree_vm_context_t* context = NULL;
  iree_vm_module_t* modules[] = {hal_module, bytecode_module};
  
  // FIXED: Typo was here (removed '/')
  IREE_RETURN_IF_ERROR(iree_vm_context_create_with_modules(
      instance, IREE_VM_CONTEXT_FLAG_NONE, IREE_ARRAYSIZE(modules), &modules[0],
      iree_allocator_system(), &context));
      
  iree_vm_module_release(hal_module);
  iree_vm_module_release(bytecode_module);

  // Lookup the entry point function.
  const char kMainFunctionName[] = "module.main_graph";
  iree_vm_function_t main_function;
  IREE_RETURN_IF_ERROR(iree_vm_context_resolve_function(
      context, iree_make_cstring_view(kMainFunctionName), &main_function));

  // Input setup for 16x1024xf32
  const iree_hal_dim_t kInputShape[] = {16, 1024};
  const size_t kInputElements = 16 * 1024;
  const size_t kInputByteSize = kInputElements * sizeof(float);

  float* input_data = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(iree_allocator_system(), 
                                             kInputByteSize, 
                                             (void**)&input_data));
  
  // Initialize input data
  for (size_t i = 0; i < kInputElements; ++i) {
      input_data[i] = 0.5f; 
  }

  // Allocate buffers in device-local memory
  iree_hal_buffer_view_t* arg0_buffer_view = NULL;
  iree_status_t status = iree_hal_buffer_view_allocate_buffer_copy(
      device, iree_hal_device_allocator(device), IREE_ARRAYSIZE(kInputShape), kInputShape,
      IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      iree_make_const_byte_span(input_data, kInputByteSize), &arg0_buffer_view);
  
  iree_allocator_free(iree_allocator_system(), input_data);
  IREE_RETURN_IF_ERROR(status);

  // Setup call inputs
  iree_vm_list_t* inputs = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_list_create(iree_vm_make_undefined_type_def(),
                          /*capacity=*/1, iree_allocator_system(), &inputs),
      "can't allocate input vm list");

  iree_vm_ref_t arg0_buffer_view_ref =
      iree_hal_buffer_view_move_ref(arg0_buffer_view);
  IREE_RETURN_IF_ERROR(
      iree_vm_list_push_ref_move(inputs, &arg0_buffer_view_ref));

  // Prepare outputs list
  iree_vm_list_t* outputs = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_list_create(iree_vm_make_undefined_type_def(),
                          /*capacity=*/1, iree_allocator_system(), &outputs),
      "can't allocate output vm list");

  // Synchronously invoke the function.
  IREE_RETURN_IF_ERROR(iree_vm_invoke(
      context, main_function, IREE_VM_INVOCATION_FLAG_NONE,
      /*policy=*/NULL, inputs, outputs, iree_allocator_system()));

  // Get the result buffers
  iree_hal_buffer_view_t* ret_buffer_view =
      iree_vm_list_get_buffer_view_assign(outputs, 0);
  if (ret_buffer_view == NULL) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "can't find return buffer view");
  }

  // Output reading for 16x10xf32
  const size_t kOutputElements = 16 * 10;
  float* results = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(iree_allocator_system(),
                                             kOutputElements * sizeof(float),
                                             (void**)&results));
  
  status = iree_hal_device_transfer_d2h(
      device, iree_hal_buffer_view_buffer(ret_buffer_view), 0, results,
      kOutputElements * sizeof(float), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout());

  // Prevent optimization removal of the read
  if (iree_status_is_ok(status)) {
    volatile float accumulator = 0.0f;
    for (size_t i = 0; i < kOutputElements; ++i) {
        accumulator += results[i];
    }
    (void)accumulator;
  }

  iree_allocator_free(iree_allocator_system(), results);
  iree_vm_list_release(inputs);
  iree_vm_list_release(outputs);
  iree_hal_device_release(device);
  iree_vm_context_release(context);
  iree_vm_instance_release(instance);
  return status;
}

int main() {
  const iree_status_t result = Run();
  int ret = (int)iree_status_code(result);
  
  // CHANGED: Ignore status (no printing on bare metal)
  iree_status_ignore(result);
  
  return ret;
}