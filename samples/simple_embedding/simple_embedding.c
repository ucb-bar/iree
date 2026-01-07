// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h> 

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"


#define L_TRACE_ENCODER_BASE_ADDRESS 0x3000000

typedef struct {
  volatile uint32_t TR_TE_CTRL;        // 0x00
  volatile uint32_t TR_TE_INFO;        // 0x04
  volatile uint32_t TR_TE_BUBBLE[6];   // 0x08-0x1C
  volatile uint32_t TR_TE_TARGET;      // 0x20
  volatile uint32_t TR_TE_BRANCH_MODE; // 0x24
} LTraceEncoderType;

static inline void l_trace_encoder_start(uint32_t hart_id) {
    // FIX: Cast base address to uintptr_t so it matches 64-bit pointer size
    uintptr_t base_addr = (uintptr_t)L_TRACE_ENCODER_BASE_ADDRESS;
    LTraceEncoderType *encoder = (LTraceEncoderType *)(base_addr + (hart_id * 0x1000));
    
    encoder->TR_TE_CTRL |= (0x1 << 1); 
}

static inline void l_trace_encoder_stop(uint32_t hart_id) {
    // FIX: Cast base address to uintptr_t so it matches 64-bit pointer size
    uintptr_t base_addr = (uintptr_t)L_TRACE_ENCODER_BASE_ADDRESS;
    LTraceEncoderType *encoder = (LTraceEncoderType *)(base_addr + (hart_id * 0x1000));
    
    encoder->TR_TE_CTRL &= ~(0x1 << 1); 
}

// A function to create the HAL device from the different backend targets.
// The HAL device is returned based on the implementation, and it must be
// released by the caller.
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

  // Match function name in MLIR
  const char kMainFunctionName[] = "module.vanilla_matmul_large";
  iree_vm_function_t main_function;
  IREE_RETURN_IF_ERROR(iree_vm_context_resolve_function(
      context, iree_make_cstring_view(kMainFunctionName), &main_function));

  // --- 1. PREPARE DATA ---
  // Size: 128x128 = 16384 elements
  const int kDim = 128;
  const int kCount = kDim * kDim;

  // LHS (Filled with 4)
  int8_t* kInt8_4 = (int8_t*)malloc(kCount * sizeof(int8_t));
  for (int i = 0; i < kCount; ++i) kInt8_4[i] = 4;

  // RHS (Filled with 2)
  int8_t* kInt8_2 = (int8_t*)malloc(kCount * sizeof(int8_t));
  for (int i = 0; i < kCount; ++i) kInt8_2[i] = 2;

  // ACC (Filled with 0)
  int32_t* kInt32_Zero = (int32_t*)malloc(kCount * sizeof(int32_t));
  for (int i = 0; i < kCount; ++i) kInt32_Zero[i] = 0;

  // --- 2. ALLOCATE BUFFERS ---
  iree_hal_dim_t shape[2] = {kDim, kDim};
  
  iree_hal_buffer_view_t* arg0_buffer_view = NULL;
  iree_hal_buffer_view_t* arg1_buffer_view = NULL;
  iree_hal_buffer_view_t* arg2_buffer_view = NULL; 

  // Allocate LHS (A)
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer_copy(
      device, iree_hal_device_allocator(device), IREE_ARRAYSIZE(shape), shape,
      IREE_HAL_ELEMENT_TYPE_SINT_8, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      iree_make_const_byte_span(kInt8_4, kCount * sizeof(int8_t)), &arg0_buffer_view));

  // Allocate RHS (B)
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer_copy(
      device, iree_hal_device_allocator(device), IREE_ARRAYSIZE(shape), shape,
      IREE_HAL_ELEMENT_TYPE_SINT_8, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      iree_make_const_byte_span(kInt8_2, kCount * sizeof(int8_t)), &arg1_buffer_view));

  // Allocate ACC (C)
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer_copy(
      device, iree_hal_device_allocator(device), IREE_ARRAYSIZE(shape), shape,
      IREE_HAL_ELEMENT_TYPE_SINT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      iree_make_const_byte_span(kInt32_Zero, kCount * sizeof(int32_t)), &arg2_buffer_view));

  // --- 3. BUILD INPUT LIST ---
  iree_vm_list_t* inputs = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_list_create(iree_vm_make_undefined_type_def(),
                          /*capacity=*/3, 
                          iree_allocator_system(), &inputs),
      "can't allocate input vm list");

  iree_vm_ref_t arg0_ref = iree_hal_buffer_view_move_ref(arg0_buffer_view);
  iree_vm_ref_t arg1_ref = iree_hal_buffer_view_move_ref(arg1_buffer_view);
  iree_vm_ref_t arg2_ref = iree_hal_buffer_view_move_ref(arg2_buffer_view);

  IREE_RETURN_IF_ERROR(iree_vm_list_push_ref_move(inputs, &arg0_ref));
  IREE_RETURN_IF_ERROR(iree_vm_list_push_ref_move(inputs, &arg1_ref));
  IREE_RETURN_IF_ERROR(iree_vm_list_push_ref_move(inputs, &arg2_ref));

  iree_vm_list_t* outputs = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_list_create(iree_vm_make_undefined_type_def(),
                          /*capacity=*/1, iree_allocator_system(), &outputs),
      "can't allocate output vm list");
  
  // --- 4. INVOKE ---
  fprintf(stdout, "Invoking 128x128 Matmul...\n");
  IREE_RETURN_IF_ERROR(iree_vm_invoke(
      context, main_function, IREE_VM_INVOCATION_FLAG_NONE,
      /*policy=*/NULL, inputs, outputs, iree_allocator_system()));
  fprintf(stdout, "Invocation Complete.\n");

  // --- 5. VERIFY RESULTS ---
  iree_hal_buffer_view_t* ret_buffer_view =
      iree_vm_list_get_buffer_view_assign(outputs, 0);
  if (ret_buffer_view == NULL) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "can't find return buffer view");
  }

  // Read back results
  int32_t* results = (int32_t*)malloc(kCount * sizeof(int32_t));
  IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2h(
      device, iree_hal_buffer_view_buffer(ret_buffer_view), 0, results,
      kCount * sizeof(int32_t), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));

  // Verification Logic:
  // Dot Product Length K = 128.
  // Expected = 128 * (4 * 2) = 1024.
  int errors = 0;
  for (iree_host_size_t i = 0; i < kCount; ++i) {
    if (results[i] != 1024) {
        if (errors < 10) { // Print only first 10 errors
            fprintf(stderr, "Mismatch at %d: Expected 1024, got %d\n", (int)i, results[i]);
        }
        errors++;
    }
  }

  if (errors > 0) {
      fprintf(stderr, "Total errors: %d\n", errors);
      return iree_make_status(IREE_STATUS_UNKNOWN, "result mismatches");
  }

  // Cleanup
  free(kInt8_4);
  free(kInt8_2);
  free(kInt32_Zero);
  free(results);
  iree_vm_list_release(inputs);
  iree_vm_list_release(outputs);
  iree_hal_device_release(device);
  iree_vm_context_release(context);
  iree_vm_instance_release(instance);
  return iree_ok_status();
}

int main() {
  // --- START TRACE ---
  fprintf(stdout, "Starting Trace...\n");
  l_trace_encoder_start(0);


  fprintf(stdout, "simple_embedding started\n");
  const iree_status_t result = Run();
  if (!iree_status_is_ok(result)) {
    iree_status_fprint(stderr, result);
    iree_status_free(result);
    return 1;
  }
  fprintf(stdout, "simple_embedding done\n");

  // --- STOP TRACE ---
  l_trace_encoder_stop(0);
  fprintf(stdout, "Trace Stopped.\n");
  return 0;
}
