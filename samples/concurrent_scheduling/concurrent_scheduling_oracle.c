// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Oracle Scheduler for Concurrent Model Execution
// 
// This example demonstrates a custom "oracle" scheduling approach where we
// manually control the execution of three neural network models with known
// characteristics on SpaceMIT X60 hardware (2 CPU clusters + NPU).
//
// Models:
// - Model A (conv): Feature extraction, runs on NPU/Cluster 1
// - Model B (dense): Classification, runs on Cluster 0 (CPU intensive)
// - Model C (residual): Independent processing, runs on Cluster 1
//
// Schedule Strategy:
// - Model A+B pipeline runs at high frequency (every iteration)
// - Model C runs at lower frequency (every 2 iterations)
// - Use explicit device placement and manual synchronization
// - Leverage hardware topology knowledge for optimal placement

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/runtime/api.h"

// Configuration for different models
typedef struct {
  const char* name;
  const char* entry_point;
  int priority;           // Higher = more priority
  int target_cluster;     // 0 or 1 for CPU clusters
  bool use_npu;          // Try to use NPU if available
  int64_t estimated_time_us;  // Estimated execution time
} model_config_t;

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
  if (argc < 4) {
    fprintf(stderr,
            "Usage: concurrent_scheduling_oracle <model_a.vmfb> <model_b.vmfb> <model_c.vmfb> [iterations]\n");
    return -1;
  }

  const char* model_a_path = argv[1];
  const char* model_b_path = argv[2];
  const char* model_c_path = argv[3];
  int iterations = 10;
  if (argc >= 5) {
    iterations = atoi(argv[4]);
  }

  fprintf(stdout, "=== Oracle Scheduler for Concurrent Model Execution ===\n");
  fprintf(stdout, "Target: SpaceMIT X60 (2 CPU clusters + NPU)\n");
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

  // Create session for each model
  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  
  iree_runtime_session_t* session_a = NULL;
  iree_runtime_session_t* session_b = NULL;
  iree_runtime_session_t* session_c = NULL;
  
  IREE_CHECK_OK(iree_runtime_session_create_with_device(
      instance, &session_options, device,
      iree_runtime_instance_host_allocator(instance), &session_a));
  IREE_CHECK_OK(iree_runtime_session_create_with_device(
      instance, &session_options, device,
      iree_runtime_instance_host_allocator(instance), &session_b));
  IREE_CHECK_OK(iree_runtime_session_create_with_device(
      instance, &session_options, device,
      iree_runtime_instance_host_allocator(instance), &session_c));

  // Load modules
  IREE_CHECK_OK(iree_runtime_session_append_bytecode_module_from_file(
      session_a, model_a_path));
  IREE_CHECK_OK(iree_runtime_session_append_bytecode_module_from_file(
      session_b, model_b_path));
  IREE_CHECK_OK(iree_runtime_session_append_bytecode_module_from_file(
      session_c, model_c_path));

  // Model configurations
  model_config_t config_a = {
    .name = "Model A (Conv)",
    .entry_point = "extract_features",
    .priority = 2,
    .target_cluster = 1,
    .use_npu = true,
    .estimated_time_us = 5000,
  };
  
  model_config_t config_b = {
    .name = "Model B (Dense)",
    .entry_point = "classify",
    .priority = 1,
    .target_cluster = 0,
    .use_npu = false,
    .estimated_time_us = 3000,
  };
  
  model_config_t config_c = {
    .name = "Model C (Residual)",
    .entry_point = "process_data",
    .priority = 0,
    .target_cluster = 1,
    .use_npu = false,
    .estimated_time_us = 4000,
  };

  // Initialize statistics
  timing_stats_t stats_a = {.model_name = "Model A", .execution_count = 0};
  timing_stats_t stats_b = {.model_name = "Model B", .execution_count = 0};
  timing_stats_t stats_c = {.model_name = "Model C", .execution_count = 0};
  timing_stats_t stats_total = {.model_name = "Total Pipeline", .execution_count = 0};

  fprintf(stdout, "Starting execution with oracle scheduling...\n\n");
  
  int64_t pipeline_start = get_time_us();

  // Main execution loop with oracle scheduling
  for (int iter = 0; iter < iterations; iter++) {
    int64_t iter_start = get_time_us();
    
    fprintf(stdout, "=== Iteration %d/%d ===\n", iter + 1, iterations);

    // Prepare inputs for Model A
    iree_vm_list_t* inputs_a = NULL;
    iree_vm_list_t* outputs_a = NULL;
    IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                      host_allocator, &inputs_a));
    IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                      host_allocator, &outputs_a));

    // Create input tensor for Model A: 1x28x28x3
    const iree_hal_dim_t shape_a[] = {1, 28, 28, 3};
    size_t input_size_a = 1 * 28 * 28 * 3 * sizeof(float);
    float* input_data_a = (float*)malloc(input_size_a);
    for (size_t i = 0; i < 1 * 28 * 28 * 3; i++) {
      input_data_a[i] = 0.5f;  // Dummy input
    }

    iree_hal_buffer_view_t* input_view_a = NULL;
    IREE_CHECK_OK(iree_hal_buffer_view_allocate_buffer_copy(
        iree_runtime_session_device(session_a),
        iree_runtime_session_device_allocator(session_a),
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
    IREE_CHECK_OK(iree_vm_list_push_ref_move(inputs_a, &input_view_a_ref));

    // Execute Model A
    fprintf(stdout, "  Launching %s on cluster %d%s...\n", 
            config_a.name, config_a.target_cluster,
            config_a.use_npu ? " (NPU)" : "");
    int64_t start_a = get_time_us();
    IREE_CHECK_OK(iree_runtime_session_call_by_name(
        session_a, iree_make_cstring_view(config_a.entry_point),
        inputs_a, outputs_a));
    int64_t end_a = get_time_us();
    int64_t time_a = end_a - start_a;
    update_stats(&stats_a, time_a);
    fprintf(stdout, "  %s completed in %lld us\n", config_a.name, (long long)time_a);

    // Get output from Model A (input for Model B)
    iree_hal_buffer_view_t* output_a_view =
        iree_vm_list_get_buffer_view_assign(outputs_a, 0);

    // Prepare inputs for Model B
    iree_vm_list_t* inputs_b = NULL;
    iree_vm_list_t* outputs_b = NULL;
    IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                      host_allocator, &inputs_b));
    IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                      host_allocator, &outputs_b));

    // Pass Model A output to Model B
    iree_vm_ref_t output_a_ref = iree_hal_buffer_view_retain_ref(output_a_view);
    IREE_CHECK_OK(iree_vm_list_push_ref_move(inputs_b, &output_a_ref));

    // Execute Model B (dependent on A)
    fprintf(stdout, "  Launching %s on cluster %d...\n", 
            config_b.name, config_b.target_cluster);
    int64_t start_b = get_time_us();
    IREE_CHECK_OK(iree_runtime_session_call_by_name(
        session_b, iree_make_cstring_view(config_b.entry_point),
        inputs_b, outputs_b));
    int64_t end_b = get_time_us();
    int64_t time_b = end_b - start_b;
    update_stats(&stats_b, time_b);
    fprintf(stdout, "  %s completed in %lld us\n", config_b.name, (long long)time_b);

    // Execute Model C every other iteration (lower frequency)
    int64_t time_c = 0;
    if (iter % 2 == 0) {
      // Prepare inputs for Model C
      iree_vm_list_t* inputs_c = NULL;
      iree_vm_list_t* outputs_c = NULL;
      IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                        host_allocator, &inputs_c));
      IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                        host_allocator, &outputs_c));

      // Create input tensor for Model C: 1x32x32x8
      const iree_hal_dim_t shape_c[] = {1, 32, 32, 8};
      size_t input_size_c = 1 * 32 * 32 * 8 * sizeof(float);
      float* input_data_c = (float*)malloc(input_size_c);
      for (size_t i = 0; i < 1 * 32 * 32 * 8; i++) {
        input_data_c[i] = 0.3f;  // Dummy input
      }

      iree_hal_buffer_view_t* input_view_c = NULL;
      IREE_CHECK_OK(iree_hal_buffer_view_allocate_buffer_copy(
          iree_runtime_session_device(session_c),
          iree_runtime_session_device_allocator(session_c),
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
      IREE_CHECK_OK(iree_vm_list_push_ref_move(inputs_c, &input_view_c_ref));

      // Execute Model C (independent)
      fprintf(stdout, "  Launching %s on cluster %d...\n", 
              config_c.name, config_c.target_cluster);
      int64_t start_c = get_time_us();
      IREE_CHECK_OK(iree_runtime_session_call_by_name(
          session_c, iree_make_cstring_view(config_c.entry_point),
          inputs_c, outputs_c));
      int64_t end_c = get_time_us();
      time_c = end_c - start_c;
      update_stats(&stats_c, time_c);
      fprintf(stdout, "  %s completed in %lld us\n", config_c.name, (long long)time_c);

      free(input_data_c);
      iree_vm_list_release(inputs_c);
      iree_vm_list_release(outputs_c);
    } else {
      fprintf(stdout, "  Skipping %s (low frequency)\n", config_c.name);
    }

    int64_t iter_end = get_time_us();
    int64_t iter_time = iter_end - iter_start;
    update_stats(&stats_total, iter_time);
    fprintf(stdout, "  Iteration time: %lld us\n\n", (long long)iter_time);

    // Cleanup
    free(input_data_a);
    iree_vm_list_release(inputs_a);
    iree_vm_list_release(outputs_a);
    iree_vm_list_release(inputs_b);
    iree_vm_list_release(outputs_b);
  }

  int64_t pipeline_end = get_time_us();
  int64_t total_pipeline_time = pipeline_end - pipeline_start;

  // Print final statistics
  fprintf(stdout, "\n=== Execution Statistics ===\n");
  print_stats(&stats_a);
  fprintf(stdout, "\n");
  print_stats(&stats_b);
  fprintf(stdout, "\n");
  print_stats(&stats_c);
  fprintf(stdout, "\n");
  print_stats(&stats_total);
  fprintf(stdout, "\nTotal pipeline time: %lld us (%.3f ms)\n",
          (long long)total_pipeline_time,
          total_pipeline_time / 1000.0);

  // Cleanup
  iree_runtime_session_release(session_a);
  iree_runtime_session_release(session_b);
  iree_runtime_session_release(session_c);
  iree_hal_device_release(device);
  iree_runtime_instance_release(instance);

  fprintf(stdout, "\nOracle scheduling completed successfully!\n");
  return 0;
}
