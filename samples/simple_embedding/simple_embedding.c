// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

// --- TRACE ENCODER DEFINITIONS (Must be BEFORE Run()) ---

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

// A function to load the vm bytecode module from the different backend targets.
// The bytecode module is generated for the specific backend and platform.
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

  // Load bytecode module from the embedded data.
  const iree_const_byte_span_t module_data = load_bytecode_module_data();

  iree_vm_module_t* bytecode_module = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_create(
      instance, module_data, iree_allocator_null(), iree_allocator_system(),
      &bytecode_module));

  // Allocate a context that will hold the module state across invocations.
  iree_vm_context_t* context = NULL;
  iree_vm_module_t* modules[] = {hal_module, bytecode_module};
  IREE_RETURN_IF_ERROR(iree_vm_context_create_with_modules(
      instance, IREE_VM_CONTEXT_FLAG_NONE, IREE_ARRAYSIZE(modules), &modules[0],
      iree_allocator_system(), &context));
  iree_vm_module_release(hal_module);
  iree_vm_module_release(bytecode_module);

  // Lookup the entry point function.
  // We use "module.vanilla_matmul" which takes 3 arguments: (A, B, C).
  const char kMainFunctionName[] = "module.vanilla_matmul";
  iree_vm_function_t main_function;
  IREE_RETURN_IF_ERROR(iree_vm_context_resolve_function(
      context, iree_make_cstring_view(kMainFunctionName), &main_function));

  // --- 1. PREPARE DATA ---
  // We need 8x8 = 64 elements for the full register utilization test.
  // A (Filled with 4.0f)
  float kFloat4[64];
  for (int i = 0; i < 64; ++i) kFloat4[i] = 4.0f;

  // B (Filled with 2.0f)
  float kFloat2[64];
  for (int i = 0; i < 64; ++i) kFloat2[i] = 2.0f;

  // C (Accumulator, Filled with 0.0f)
  // This corresponds to the `outs(%C)` in the MLIR.
  float kFloatZero[64];
  for (int i = 0; i < 64; ++i) kFloatZero[i] = 0.0f;

  // --- 2. ALLOCATE BUFFERS ---
  // Rank 2 shape: 8x8
  iree_hal_dim_t shape[2] = {8, 8};
  
  iree_hal_buffer_view_t* arg0_buffer_view = NULL;
  iree_hal_buffer_view_t* arg1_buffer_view = NULL;
  iree_hal_buffer_view_t* arg2_buffer_view = NULL; // New for input C

  // Allocate Input A
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer_copy(
      device, iree_hal_device_allocator(device), IREE_ARRAYSIZE(shape), shape,
      IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      iree_make_const_byte_span(kFloat4, sizeof(kFloat4)), &arg0_buffer_view));

  // Allocate Input B
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer_copy(
      device, iree_hal_device_allocator(device), IREE_ARRAYSIZE(shape), shape,
      IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      iree_make_const_byte_span(kFloat2, sizeof(kFloat2)), &arg1_buffer_view));

  // Allocate Input C (Accumulator)
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer_copy(
      device, iree_hal_device_allocator(device), IREE_ARRAYSIZE(shape), shape,
      IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      iree_make_const_byte_span(kFloatZero, sizeof(kFloatZero)), &arg2_buffer_view));

  // --- 3. BUILD INPUT LIST ---
  iree_vm_list_t* inputs = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_list_create(iree_vm_make_undefined_type_def(),
                          /*capacity=*/3, // CHANGED: We now have 3 inputs
                          iree_allocator_system(), &inputs),
      "can't allocate input vm list");

  iree_vm_ref_t arg0_ref = iree_hal_buffer_view_move_ref(arg0_buffer_view);
  iree_vm_ref_t arg1_ref = iree_hal_buffer_view_move_ref(arg1_buffer_view);
  iree_vm_ref_t arg2_ref = iree_hal_buffer_view_move_ref(arg2_buffer_view);

  IREE_RETURN_IF_ERROR(iree_vm_list_push_ref_move(inputs, &arg0_ref));
  IREE_RETURN_IF_ERROR(iree_vm_list_push_ref_move(inputs, &arg1_ref));
  IREE_RETURN_IF_ERROR(iree_vm_list_push_ref_move(inputs, &arg2_ref));

  // Prepare outputs list to accept the results from the invocation.
  iree_vm_list_t* outputs = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_list_create(iree_vm_make_undefined_type_def(),
                          /*capacity=*/1, iree_allocator_system(), &outputs),
      "can't allocate output vm list");
  
  // --- START TRACE ---
  fprintf(stdout, "Starting Trace...\n");
  l_trace_encoder_start(0);

  // --- 4. INVOKE ---
  IREE_RETURN_IF_ERROR(iree_vm_invoke(
      context, main_function, IREE_VM_INVOCATION_FLAG_NONE,
      /*policy=*/NULL, inputs, outputs, iree_allocator_system()));

  // --- STOP TRACE ---
  l_trace_encoder_stop(0);
  fprintf(stdout, "Trace Stopped.\n");

  // Get the result buffers from the invocation.
  iree_hal_buffer_view_t* ret_buffer_view =
      iree_vm_list_get_buffer_view_assign(outputs, 0);
  if (ret_buffer_view == NULL) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "can't find return buffer view");
  }

  // --- 5. VERIFY RESULTS ---
  // Read back 64 floats (8x8).
  float results[64];
  // Initialize to zero to ensure we are reading fresh data
  for(int i=0; i<64; ++i) results[i] = 0.0f;

  IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2h(
      device, iree_hal_buffer_view_buffer(ret_buffer_view), 0, results,
      sizeof(results), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));

  // Verification Logic:
  // Row of A (all 4.0) dot Column of B (all 2.0).
  // Length is 8.
  // Result = 8 * (4.0 * 2.0) = 8 * 8.0 = 64.0.
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(results); ++i) {
    if (results[i] != 64.0f) {
        fprintf(stderr, "Result mismatch at index %zu: Expected 64.0f, got %f\n", i, results[i]);
        return iree_make_status(IREE_STATUS_UNKNOWN, "result mismatches");
    }
  }

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

  fprintf(stdout, "vanilla_matmul test started\n");
  //const iree_status_t result = Run();
  //int ret = (int)iree_status_code(result);
  //if (!iree_status_is_ok(result)) {
  //  iree_status_fprint(stderr, result);
  //  iree_status_free(result);
  //}
  fprintf(stdout, "vanilla_matmul test done\n");

  // --- STOP TRACE ---
  l_trace_encoder_stop(0);
  fprintf(stdout, "Trace Stopped.\n");

  //return ret;
  return 0;
}