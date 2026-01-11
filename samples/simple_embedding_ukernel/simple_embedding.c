// Copyright 2025 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h> 
#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

// --- Hardware Timer ---
static inline uint64_t read_cycles() {
    uint64_t cycles;
    asm volatile("rdcycle %0" : "=r"(cycles));
    return cycles;
}

// External declarations from device_embedded_sync.c
extern iree_status_t create_sample_device(iree_allocator_t host_allocator,
                                          iree_hal_device_t** out_device);
extern const iree_const_byte_span_t load_bytecode_module_by_index(int index);

// Struct to hold benchmark results
typedef struct {
    int size;
    int use_ukernel; // 1 = All, 0 = None
    double avg_cycles;
    double ops_per_cycle;
} BenchResult;

// --- Run Single Model ---
iree_status_t RunModel(int model_index, int size, int use_ukernel, BenchResult* out_result) {
  iree_vm_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_instance_create(
      IREE_VM_TYPE_CAPACITY_DEFAULT, iree_allocator_system(), &instance));
  IREE_RETURN_IF_ERROR(iree_hal_module_register_all_types(instance));

  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(create_sample_device(iree_allocator_system(), &device));
  
  iree_vm_module_t* hal_module = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_module_create(
      instance, iree_hal_module_device_policy_default(), 1, &device,
      IREE_HAL_MODULE_FLAG_SYNCHRONOUS, iree_hal_module_debug_sink_stdio(stderr),
      iree_allocator_system(), &hal_module));

  const iree_const_byte_span_t module_data = load_bytecode_module_by_index(model_index);
  if (module_data.data_length == 0) {
      return iree_make_status(IREE_STATUS_NOT_FOUND, "Module data not found for index %d", model_index);
  }

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

  // CHANGED: Function name is now always "module.main"
  iree_vm_function_t main_function;
  IREE_RETURN_IF_ERROR(iree_vm_context_resolve_function(
      context, iree_make_cstring_view("module.main"), &main_function));

  // --- Prepare Data ---
  const iree_host_size_t kCount = (iree_host_size_t)size * size;
  
  // Data generation: A=4, B=2
  int8_t* valA = (int8_t*)malloc(kCount * sizeof(int8_t));
  int8_t* valB = (int8_t*)malloc(kCount * sizeof(int8_t));
  memset(valA, 4, kCount * sizeof(int8_t));
  memset(valB, 2, kCount * sizeof(int8_t));

  iree_hal_dim_t shape[2] = {size, size};
  iree_hal_buffer_view_t *bv0 = NULL, *bv1 = NULL;

  iree_hal_buffer_params_t params = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT
  };

  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer_copy(
      device, iree_hal_device_allocator(device), 2, shape,
      IREE_HAL_ELEMENT_TYPE_SINT_8, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      params, iree_make_const_byte_span(valA, kCount), &bv0));

  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer_copy(
      device, iree_hal_device_allocator(device), 2, shape,
      IREE_HAL_ELEMENT_TYPE_SINT_8, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      params, iree_make_const_byte_span(valB, kCount), &bv1));

  // CHANGED: Only 2 inputs now
  iree_vm_list_t* inputs = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(), 2, iree_allocator_system(), &inputs));
  
  iree_vm_ref_t ref0 = iree_hal_buffer_view_move_ref(bv0);
  iree_vm_ref_t ref1 = iree_hal_buffer_view_move_ref(bv1);
  
  iree_vm_list_push_ref_move(inputs, &ref0);
  iree_vm_list_push_ref_move(inputs, &ref1);

  iree_vm_list_t* outputs = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1, iree_allocator_system(), &outputs));

  // --- Benchmark ---
  printf("Running: Size=%d, Ukernel=%s\n", size, use_ukernel ? "ALL" : "NONE");
  
  const int kWarmup = 2;
  const int kIters = 10;

  printf("  Warming up...\n");

  for(int i=0; i<kWarmup; ++i) {
      iree_vm_list_resize(outputs, 0);
      IREE_RETURN_IF_ERROR(iree_vm_invoke(context, main_function, IREE_VM_INVOCATION_FLAG_NONE, NULL, inputs, outputs, iree_allocator_system()));
  }

  printf("  Running benchmark (%d iterations)...\n", kIters);

  uint64_t start = read_cycles();
  for(int i=0; i<kIters; ++i) {
      iree_vm_list_resize(outputs, 0);
      IREE_RETURN_IF_ERROR(iree_vm_invoke(context, main_function, IREE_VM_INVOCATION_FLAG_NONE, NULL, inputs, outputs, iree_allocator_system()));
  }
  uint64_t end = read_cycles();

  double avg = (double)(end - start) / kIters;
  double total_ops = 2.0 * (double)size * (double)size * (double)size;
  double efficiency = total_ops / avg;

    printf("  Average Cycles: %.2f\n", avg);

  out_result->size = size;
  out_result->use_ukernel = use_ukernel;
  out_result->avg_cycles = avg;
  out_result->ops_per_cycle = efficiency;

  // --- Verify ---
  iree_hal_buffer_view_t* ret_bv = iree_vm_list_get_buffer_view_assign(outputs, 0);
  int32_t* res_data = (int32_t*)malloc(kCount * sizeof(int32_t));
  IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2h(device, iree_hal_buffer_view_buffer(ret_bv), 0, res_data, kCount*4, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  
  int32_t expected = 8 * size; 
  int errs = 0;
  for(int i=0; i<kCount; i+=101) { 
      if(res_data[i] != expected) {
          if(errs < 1) printf("  Error: Expected %d, got %d at index %d\n", expected, res_data[i], i);
          errs++;
      }
  }
  if(errs > 0) printf("  Verification FAILED.\n");
  else printf("  Verification Passed.\n");

  free(valA); free(valB); free(res_data);
  iree_vm_list_release(inputs);
  iree_vm_list_release(outputs);
  iree_hal_device_release(device);
  iree_vm_context_release(context);
  iree_vm_instance_release(instance);

  return iree_ok_status();
}

int main() {
    const int kNumConfigs = 12;
    BenchResult results[12];

    int sizes[] = {64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024, 2048, 2048};
    int uks[]   = {0,  1,  1,   0,   1,   0,   1,   0,   1,    0,    1,    0};

    printf("=== Starting Multi-Model Benchmark (Up to 2048) ===\n");

    for(int i=0; i<kNumConfigs; ++i) {
        iree_status_t status = RunModel(i, sizes[i], uks[i], &results[i]);
        if(!iree_status_is_ok(status)) {
            iree_status_fprint(stderr, status);
            iree_status_free(status);
            return 1;
        }
    }

    printf("\n=======================================================================\n");
    printf("BENCHMARK REPORT (RISC-V 64)\n");
    printf("=======================================================================\n");
    printf("| Size      | Ukernel | Avg Cycles         | Ops/Cycle | Speedup vs None |\n");
    printf("|-----------|---------|--------------------|-----------|-----------------|\n");

    for (int i = 0; i < kNumConfigs; i += 2) {
        BenchResult rAll = results[i];
        BenchResult rNone = results[i+1];
        
        double speedup = rNone.avg_cycles / rAll.avg_cycles;

        printf("| %4d x%4d | ALL     | %18.2f | %9.2f | %13.2fx |\n", 
               rAll.size, rAll.size, rAll.avg_cycles, rAll.ops_per_cycle, speedup);
        printf("| %4d x%4d | NONE    | %18.2f | %9.2f |             1.0 |\n", 
               rNone.size, rNone.size, rNone.avg_cycles, rNone.ops_per_cycle);
        printf("|-----------|---------|--------------------|-----------|-----------------|\n");
    }

    return 0;
}