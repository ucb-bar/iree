# Concurrent Model Scheduling Example

This example demonstrates concurrent execution of multiple neural network models
with different scheduling strategies, designed to showcase the trade-offs between
vanilla IREE automatic scheduling and custom "oracle" scheduling with hardware-aware
placement.

## Overview

The example simulates a real-world scenario where multiple ML models need to run
concurrently at different frequencies:

- **Model A (Convolutional Feature Extractor)**: Processes images and produces
  features. Simulates early layers of a MobileNet-style network.
  
- **Model B (Dense Classifier)**: Takes features from Model A and produces
  classification results. Simulates later layers of a MobileNet-style network.
  
- **Model C (Residual Processor)**: Independent workload that processes data
  concurrently. Simulates a ResNet-style bottleneck block.

### Dependencies and Frequencies

```
Input A → [Model A] → Features → [Model B] → Classification
                                      ↓
                              (High frequency: every iteration)

Input C → [Model C] → Output C
            ↓
    (Lower frequency: every 2 iterations)
```

### Target Hardware: SpaceMIT X60

- 2 CPU clusters (each with 4 cores)
- 1 cluster has NPU acceleration
- Different models can be assigned to different clusters

## Architecture Comparison

### 1. Vanilla IREE Scheduling

Uses IREE's built-in async execution model with automatic dependency tracking:

- **Pros**:
  - Fully automatic scheduling
  - No manual synchronization needed
  - Portable across different hardware
  - Uses coarse-fences ABI for async execution

- **Cons**:
  - No explicit control over device placement
  - May not leverage hardware topology optimally
  - General-purpose scheduler without workload-specific knowledge

### 2. Oracle Scheduling

Custom scheduling with explicit control and hardware awareness:

- **Pros**:
  - Explicit device placement (CPU cluster 0/1, NPU)
  - Hardware topology awareness
  - Workload-specific optimizations
  - Fine-grained control over execution order

- **Cons**:
  - More complex implementation
  - Hardware-specific (less portable)
  - Requires knowledge of model characteristics
  - Manual synchronization management

## Building

### Using CMake

```bash
cd /path/to/iree
mkdir build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
cmake --build . --target concurrent_scheduling_oracle
cmake --build . --target concurrent_scheduling_vanilla
```

## Running the Examples

### Method 1: Oracle Scheduling (Manual Control)

First, compile the individual model modules:

```bash
# Compile Model A
iree-compile \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    samples/concurrent_scheduling/model_a_conv.mlir \
    -o=model_a.vmfb

# Compile Model B
iree-compile \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    samples/concurrent_scheduling/model_b_dense.mlir \
    -o=model_b.vmfb

# Compile Model C
iree-compile \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    samples/concurrent_scheduling/model_c_residual.mlir \
    -o=model_c.vmfb
```

Run the oracle scheduler:

```bash
./build/samples/concurrent_scheduling/concurrent_scheduling_oracle \
    model_a.vmfb model_b.vmfb model_c.vmfb 10
```

The last argument (10) specifies the number of iterations.

### Method 2: Vanilla IREE Scheduling (Automatic)

Compile the pipeline with async execution model:

```bash
iree-compile \
    --iree-execution-model=async-external \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    samples/concurrent_scheduling/pipeline_vanilla_async.mlir \
    -o=pipeline.vmfb
```

Run the vanilla scheduler:

```bash
./build/samples/concurrent_scheduling/concurrent_scheduling_vanilla \
    pipeline.vmfb 10
```

### Method 3: Using iree-run-module (MLIR Pipeline Only)

This method uses the MLIR pipeline directly without the C runners:

```bash
# Compile all modules with async support
iree-compile \
    --iree-execution-model=async-external \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    samples/concurrent_scheduling/model_a_conv.mlir \
    -o=model_a_async.vmfb

iree-compile \
    --iree-execution-model=async-external \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    samples/concurrent_scheduling/model_b_dense.mlir \
    -o=model_b_async.vmfb

iree-compile \
    --iree-execution-model=async-external \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    samples/concurrent_scheduling/model_c_residual.mlir \
    -o=model_c_async.vmfb

iree-compile \
    --iree-execution-model=async-external \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    samples/concurrent_scheduling/pipeline_vanilla_async.mlir \
    -o=pipeline_async.vmfb

# Run the pipeline
iree-run-module \
    --device=local-task \
    --module=model_a_async.vmfb \
    --module=model_b_async.vmfb \
    --module=model_c_async.vmfb \
    --module=pipeline_async.vmfb \
    --function=pipeline_vanilla \
    --input=1x28x28x3xf32 \
    --input=1x32x32x8xf32
```

## Output and Performance Analysis

Both examples output detailed timing statistics:

```
=== Oracle Scheduler for Concurrent Model Execution ===
Target: SpaceMIT X60 (2 CPU clusters + NPU)
Iterations: 10

=== Iteration 1/10 ===
  Launching Model A (Conv) on cluster 1 (NPU)...
  Model A (Conv) completed in 5234 us
  Launching Model B (Dense) on cluster 0...
  Model B (Dense) completed in 3102 us
  Launching Model C (Residual) on cluster 1...
  Model C (Residual) completed in 4456 us
  Iteration time: 12892 us

...

=== Execution Statistics ===
Model A Stats:
  Executions: 10
  Min time: 5102 us
  Max time: 5456 us
  Avg time: 5234 us
  Total time: 52340 us

Model B Stats:
  Executions: 10
  Min time: 2987 us
  Max time: 3234 us
  Avg time: 3102 us
  Total time: 31020 us

Model C Stats:
  Executions: 5
  Min time: 4234 us
  Max time: 4678 us
  Avg time: 4456 us
  Total time: 22280 us
```

### Key Metrics to Compare

1. **Total pipeline time**: Overall execution time for all iterations
2. **Per-iteration time**: Average time per iteration
3. **Model execution time**: Time for each individual model
4. **Concurrency efficiency**: How well independent work overlaps

## Extending the Example

### For Real Hardware Testing

To test on actual SpaceMIT X60 hardware:

1. Modify the target backend in compilation:
   ```bash
   --iree-hal-local-target-device-backends=llvm-cpu \
   --iree-llvmcpu-target-cpu=spacemit-x60
   ```

2. Update the oracle scheduler to use actual CPU affinity:
   ```c
   // Set CPU affinity for threads
   cpu_set_t cpuset;
   CPU_ZERO(&cpuset);
   CPU_SET(cluster_id * 4, &cpuset);  // Pin to specific cluster
   pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
   ```

3. Enable NPU support if available:
   ```bash
   --iree-hal-target-device=local,local
   ```

### Adding More Models

To add additional models:

1. Create a new MLIR file (e.g., `model_d.mlir`)
2. Update `CMakeLists.txt` to compile it
3. Add it to the pipeline orchestrator
4. Update the C runners to include the new model

### Adjusting Frequencies

Modify the frequency logic in the orchestrators:

- In MLIR: Change the `scf.for` loop conditions
- In C: Modify the `if (iter % N == 0)` conditions

## References

- IREE Multiple Modules: `samples/multiple_modules/`
- IREE Async Execution: `samples/custom_module/async/`
- IREE Documentation: https://iree.dev/

## Performance Tips

1. **Compilation Flags**: Use `-O3` and target-specific optimizations
2. **Buffer Management**: Reuse buffers across iterations
3. **Device Selection**: Choose appropriate backends (CPU vs GPU vs NPU)
4. **Thread Pool Size**: Tune based on hardware topology
5. **Profiling**: Use IREE's built-in profiling tools to identify bottlenecks
