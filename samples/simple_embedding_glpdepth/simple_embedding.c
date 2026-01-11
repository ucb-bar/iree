// Copyright 2025 The IREE Authors
// Licensed under the Apache License v2.0 with LLVM Exceptions.

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

// --- RISC-V Hardware Timer ---
static inline uint64_t read_cycles() {
    uint64_t cycles;
    asm volatile("rdcycle %0" : "=r"(cycles));
    return cycles;
}

extern iree_status_t create_sample_device(iree_allocator_t host_allocator,
                                          iree_hal_device_t** out_device);
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
  IREE_RETURN_IF_ERROR(iree_hal_module_create(
      instance, iree_hal_module_device_policy_default(), /*device_count=*/1,
      &device, IREE_HAL_MODULE_FLAG_SYNCHRONOUS,
      iree_hal_module_debug_sink_stdio(stderr), iree_allocator_system(),
      &hal_module));

  const iree_const_byte_span_t module_data = load_bytecode_module_data();

  iree_vm_module_t* bytecode_module = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_create(
      instance, module_data, iree_allocator_null(), iree_allocator_system(),
      &bytecode_module));

  iree_vm_context_t* context = NULL;
  iree_vm_module_t* modules[] = {hal_module, bytecode_module};
  IREE_RETURN_IF_ERROR(iree_vm_context_create_with_modules(
      instance, IREE_VM_CONTEXT_FLAG_NONE, IREE_ARRAYSIZE(modules), &modules[0],
      iree_allocator_system(), &context));
  iree_vm_module_release(hal_module);
  iree_vm_module_release(bytecode_module);

  // Match function name for glpdepth
  const char kMainFunctionName[] = "module.main_graph";
  iree_vm_function_t main_function;
  IREE_RETURN_IF_ERROR(iree_vm_context_resolve_function(
      context, iree_make_cstring_view(kMainFunctionName), &main_function));

  // --- 1. PREPARE INPUT DATA ---
  // Input: [1, 3, 224, 224] f32
  const int kBatch = 1;
  const int kChannels = 3;
  const int kHeight = 224;
  const int kWidth = 224;
  const int kInputElements = kBatch * kChannels * kHeight * kWidth;
  const size_t kInputBytes = kInputElements * sizeof(float);

  // Allocate and fill with dummy data (1.0f)
  float* kInputData = (float*)malloc(kInputBytes);
  for (int i = 0; i < kInputElements; ++i) {
      kInputData[i] = 1.0f; 
  }

  // --- 2. ALLOCATE BUFFER ---
  iree_hal_dim_t input_shape[] = {kBatch, kChannels, kHeight, kWidth};
  
  iree_hal_buffer_view_t* input_buffer_view = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer_copy(
      device, iree_hal_device_allocator(device), IREE_ARRAYSIZE(input_shape), input_shape,
      IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      iree_make_const_byte_span(kInputData, kInputBytes), 
      &input_buffer_view));

  // --- 3. BUILD LISTS ---
  iree_vm_list_t* inputs = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1, iree_allocator_system(), &inputs));
  iree_vm_ref_t input_ref = iree_hal_buffer_view_move_ref(input_buffer_view);
  IREE_RETURN_IF_ERROR(iree_vm_list_push_ref_move(inputs, &input_ref));

  iree_vm_list_t* outputs = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1, iree_allocator_system(), &outputs));
  
  // --- 4. BENCHMARK ---
  fprintf(stdout, "Starting Benchmark for glpdepth...\n");

  const int kWarmupIterations = 2;
  const int kBenchmarkIterations = 5;

  // A. Warmup
  for (int i = 0; i < kWarmupIterations; ++i) {
    iree_vm_list_resize(outputs, 0); 
    IREE_RETURN_IF_ERROR(iree_vm_invoke(
        context, main_function, IREE_VM_INVOCATION_FLAG_NONE,
        NULL, inputs, outputs, iree_allocator_system()));
  }

  // B. Timed Loop
  uint64_t start_cycles = read_cycles();
  for (int i = 0; i < kBenchmarkIterations; ++i) {
    iree_vm_list_resize(outputs, 0); 
    IREE_RETURN_IF_ERROR(iree_vm_invoke(
        context, main_function, IREE_VM_INVOCATION_FLAG_NONE,
        NULL, inputs, outputs, iree_allocator_system()));
  }
  uint64_t end_cycles = read_cycles();

  // --- 5. REPORT ---
  uint64_t total_cycles = end_cycles - start_cycles;
  double avg_cycles = (double)total_cycles / kBenchmarkIterations;

  fprintf(stdout, "--------------------------------------------------\n");
  fprintf(stdout, "Benchmark Complete.\n");
  fprintf(stdout, "Iterations: %d\n", kBenchmarkIterations);
  fprintf(stdout, "Total Cycles: %lu\n", total_cycles);
  fprintf(stdout, "Avg Cycles  : %.2f\n", avg_cycles);
  fprintf(stdout, "--------------------------------------------------\n");

  free(kInputData);
  iree_vm_list_release(inputs);
  iree_vm_list_release(outputs);
  iree_hal_device_release(device);
  iree_vm_context_release(context);
  iree_vm_instance_release(instance);
  return iree_ok_status();
}

int main() {
  const iree_status_t result = Run();
  if (!iree_status_is_ok(result)) {
    iree_status_fprint(stderr, result);
    iree_status_free(result);
    return 1;
  }
  fprintf(stdout, "Test Success\n");
  return 0;
}