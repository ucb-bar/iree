// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Vanilla IREE Scheduler for Concurrent Model Execution
// 
// This example demonstrates vanilla IREE scheduling where we let IREE's
// built-in scheduler handle all async execution and dependency management.
// The same workload as the oracle version but with automatic scheduling.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/runtime/api.h"

// Timing statistics
typedef struct {
  const char* model_name;
  int64_t min_time_us;
  int64_t max_time_us;
  int64_t total_time_us;
  int execution_count;
} timing_stats_t;

// Get current time in microseconds
static int64_t get_time_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

// Update timing statistics
static void update_stats(timing_stats_t* stats, int64_t time_us) {
  if (stats->execution_count == 0) {
    stats->min_time_us = time_us;
    stats->max_time_us = time_us;
  } else {
    if (time_us < stats->min_time_us) stats->min_time_us = time_us;
    if (time_us > stats->max_time_us) stats->max_time_us = time_us;
  }
  stats->total_time_us += time_us;
  stats->execution_count++;
}

// Print statistics
static void print_stats(const timing_stats_t* stats) {
  if (stats->execution_count == 0) {
    fprintf(stdout, "%s: No executions\n", stats->model_name);
    return;
  }
  
  int64_t avg_time_us = stats->total_time_us / stats->execution_count;
  fprintf(stdout, "%s Stats:\n", stats->model_name);
  fprintf(stdout, "  Executions: %d\n", stats->execution_count);
  fprintf(stdout, "  Min time: %lld us\n", (long long)stats->min_time_us);
  fprintf(stdout, "  Max time: %lld us\n", (long long)stats->max_time_us);
  fprintf(stdout, "  Avg time: %lld us\n", (long long)avg_time_us);
  fprintf(stdout, "  Total time: %lld us\n", (long long)stats->total_time_us);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr,
            "Usage: concurrent_scheduling_vanilla <pipeline.vmfb> [iterations]\n");
    fprintf(stderr, "  The pipeline.vmfb should be compiled with --iree-execution-model=async-external\n");
    return -1;
  }

  const char* pipeline_path = argv[1];
  int iterations = 10;
  if (argc >= 3) {
    iterations = atoi(argv[2]);
  }

  fprintf(stdout, "=== Vanilla IREE Scheduler for Concurrent Model Execution ===\n");
  fprintf(stdout, "Using IREE's automatic scheduling and dependency tracking\n");
  fprintf(stdout, "Iterations: %d\n\n", iterations);

  iree_allocator_t host_allocator = iree_allocator_system();

  // Create instance
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_runtime_instance_t* instance = NULL;
  IREE_CHECK_OK(iree_runtime_instance_create(&instance_options, host_allocator,
                                              &instance));

  // Create device (local-task allows multi-threading)
  iree_hal_device_t* device = NULL;
  IREE_CHECK_OK(iree_runtime_instance_try_create_default_device(
      instance, iree_make_cstring_view("local-task"), &device));

  // Create session
  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  
  iree_runtime_session_t* session = NULL;
  IREE_CHECK_OK(iree_runtime_session_create_with_device(
      instance, &session_options, device,
      iree_runtime_instance_host_allocator(instance), &session));

  // Load pipeline module
  IREE_CHECK_OK(iree_runtime_session_append_bytecode_module_from_file(
      session, pipeline_path));

  // Initialize statistics
  timing_stats_t stats_total = {.model_name = "Total Pipeline", .execution_count = 0};

  fprintf(stdout, "Starting execution with vanilla IREE scheduling...\n\n");
  
  int64_t pipeline_start = get_time_us();

  // Main execution loop - IREE handles all scheduling automatically
  for (int iter = 0; iter < iterations; iter++) {
    int64_t iter_start = get_time_us();
    
    fprintf(stdout, "=== Iteration %d/%d ===\n", iter + 1, iterations);

    // Prepare inputs
    iree_vm_list_t* inputs = NULL;
    iree_vm_list_t* outputs = NULL;
    IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 2,
                                      host_allocator, &inputs));
    IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 2,
                                      host_allocator, &outputs));

    // Create input tensor for Model A: 1x28x28x3
    const iree_hal_dim_t shape_a[] = {1, 28, 28, 3};
    size_t input_size_a = 1 * 28 * 28 * 3 * sizeof(float);
    float* input_data_a = (float*)malloc(input_size_a);
    for (size_t i = 0; i < 1 * 28 * 28 * 3; i++) {
      input_data_a[i] = 0.5f;
    }

    iree_hal_buffer_view_t* input_view_a = NULL;
    IREE_CHECK_OK(iree_hal_buffer_view_allocate_buffer_copy(
        iree_runtime_session_device(session),
        iree_runtime_session_device_allocator(session),
        IREE_ARRAYSIZE(shape_a), shape_a,
        IREE_HAL_ELEMENT_TYPE_FLOAT_32,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
            .access = IREE_HAL_MEMORY_ACCESS_ALL,
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        },
        iree_make_const_byte_span(input_data_a, input_size_a),
        &input_view_a));
    
    iree_vm_ref_t input_view_a_ref = iree_hal_buffer_view_move_ref(input_view_a);
    IREE_CHECK_OK(iree_vm_list_push_ref_move(inputs, &input_view_a_ref));

    // Create input tensor for Model C: 1x32x32x8
    const iree_hal_dim_t shape_c[] = {1, 32, 32, 8};
    size_t input_size_c = 1 * 32 * 32 * 8 * sizeof(float);
    float* input_data_c = (float*)malloc(input_size_c);
    for (size_t i = 0; i < 1 * 32 * 32 * 8; i++) {
      input_data_c[i] = 0.3f;
    }

    iree_hal_buffer_view_t* input_view_c = NULL;
    IREE_CHECK_OK(iree_hal_buffer_view_allocate_buffer_copy(
        iree_runtime_session_device(session),
        iree_runtime_session_device_allocator(session),
        IREE_ARRAYSIZE(shape_c), shape_c,
        IREE_HAL_ELEMENT_TYPE_FLOAT_32,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
            .access = IREE_HAL_MEMORY_ACCESS_ALL,
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        },
        iree_make_const_byte_span(input_data_c, input_size_c),
        &input_view_c));
    
    iree_vm_ref_t input_view_c_ref = iree_hal_buffer_view_move_ref(input_view_c);
    IREE_CHECK_OK(iree_vm_list_push_ref_move(inputs, &input_view_c_ref));

    // Execute the entire pipeline - IREE automatically handles:
    // - Model A execution
    // - Model C execution (concurrent with A)
    // - Model B execution (waits for A)
    // - All synchronization and dependency tracking
    fprintf(stdout, "  Launching pipeline (IREE auto-scheduling)...\n");
    IREE_CHECK_OK(iree_runtime_session_call_by_name(
        session, iree_make_cstring_view("pipeline_vanilla"),
        inputs, outputs));

    int64_t iter_end = get_time_us();
    int64_t iter_time = iter_end - iter_start;
    update_stats(&stats_total, iter_time);
    fprintf(stdout, "  Pipeline completed in %lld us\n\n", (long long)iter_time);

    // Cleanup
    free(input_data_a);
    free(input_data_c);
    iree_vm_list_release(inputs);
    iree_vm_list_release(outputs);
  }

  int64_t pipeline_end = get_time_us();
  int64_t total_pipeline_time = pipeline_end - pipeline_start;

  // Print final statistics
  fprintf(stdout, "\n=== Execution Statistics ===\n");
  print_stats(&stats_total);
  fprintf(stdout, "\nTotal pipeline time: %lld us (%.3f ms)\n",
          (long long)total_pipeline_time,
          total_pipeline_time / 1000.0);

  // Cleanup
  iree_runtime_session_release(session);
  iree_hal_device_release(device);
  iree_runtime_instance_release(instance);

  fprintf(stdout, "\nVanilla IREE scheduling completed successfully!\n");
  return 0;
}
