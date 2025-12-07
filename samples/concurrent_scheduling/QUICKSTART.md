# Quick Start Guide - Concurrent Scheduling Example

## What is This?

A complete example showing how to run multiple neural networks concurrently with:
- **Vanilla IREE**: Let IREE handle everything automatically
- **Oracle**: Control exactly how and where each model runs

Perfect for comparing automatic vs. manual scheduling on SpaceMIT X60 hardware.

## 5-Minute Quick Start

### Prerequisites
- IREE built from source
- iree-compile and iree-run-module in your PATH

### Step 1: Compile the Models
```bash
cd samples/concurrent_scheduling
./compile_models.sh
```

### Step 2: Run Vanilla IREE Scheduling
```bash
# Assuming IREE tools are built in ../../build/tools/
../../build/samples/concurrent_scheduling/concurrent_scheduling_vanilla pipeline.vmfb 10
```

### Step 3: Run Oracle Scheduling
```bash
../../build/samples/concurrent_scheduling/concurrent_scheduling_oracle \
    model_a.vmfb model_b.vmfb model_c.vmfb 10
```

### Step 4: Compare Results
Look at the timing statistics printed by each approach.

## What Gets Measured?

Both approaches output:
- Per-model execution times (min/max/avg)
- Per-iteration times
- Total pipeline time

Compare to see if custom scheduling helps!

## The Models

- **Model A**: Conv2D feature extraction (like MobileNet)
- **Model B**: Dense classifier (depends on A)
- **Model C**: Residual block (independent, runs less frequently)

## The Hardware Target

**SpaceMIT X60**:
- 2 CPU clusters (4 cores each)
- 1 cluster has NPU

Oracle tries to use this topology optimally.

## Files You Get

| File | Purpose |
|------|---------|
| `model_a_conv.mlir` | Convolutional model |
| `model_b_dense.mlir` | Dense classifier |
| `model_c_residual.mlir` | Residual processor |
| `pipeline_vanilla_async.mlir` | Async pipeline |
| `concurrent_scheduling_vanilla.c` | Vanilla runner |
| `concurrent_scheduling_oracle.c` | Oracle runner |
| `compile_models.sh` | Build helper |
| `test_example.sh` | Validation helper |

## Need More Details?

- `README.md` - Full documentation
- `IMPLEMENTATION.md` - Technical details
- `SUMMARY.md` - Design rationale

## Troubleshooting

**"iree-compile not found"**
→ Build IREE first: `cmake -G Ninja -B build && cmake --build build`

**"cannot load module"**
→ Run `compile_models.sh` first

**"permission denied"**
→ The executables are in the build directory, not current directory

## Customization

Want to modify the example?

1. **Change model frequency**: Edit the `if (iter % 2 == 0)` logic
2. **Add more models**: Copy a model file and update the orchestrators
3. **Adjust cluster placement**: Edit the `target_cluster` in oracle.c
4. **Change iterations**: Pass different number as last argument

## Expected Performance

On SpaceMIT X60, oracle scheduling may show:
- Better CPU utilization
- Lower latency for high-priority models
- More predictable timing

Results depend on actual hardware topology and workload characteristics.

## Example Output

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
```

## Success!

You now have a working example comparing scheduling approaches. Experiment and measure!
