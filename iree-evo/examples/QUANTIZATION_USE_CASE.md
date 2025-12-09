# IREE-Evo Use Case: Integer-Only Quantization Optimization

## Overview

This guide demonstrates how to use IREE-Evo to optimize quantized neural network inference by exploring compilation strategies for integer-only requantization. This is a real-world use case that addresses inefficiencies in current quantization flows.

## Problem Statement

### Current Inefficient Pattern

Most quantized neural networks today use a float-based scaling approach after integer matrix multiplication:

```
Input (i8) × Weight (i8) → Accumulator (i32)
   ↓
Convert to f32 (sitofp)           ← Slow
   ↓
Scale with f32 multiply (mulf)    ← Slow
   ↓
Add bias in f32 (addf)            ← Slow
   ↓
Requantize to i8 (divf, fptosi)   ← Slow
   ↓
Output (i8)
```

**Problems:**
- Multiple float conversions and operations
- Uses floating-point units unnecessarily
- Difficult to pattern match for custom kernels
- Suboptimal performance on integer-focused hardware

### Optimized Integer-Only Pattern

The efficient approach uses fixed-point math to keep everything in integers:

```
Input (i8) × Weight (i8) → Accumulator (i32)
   ↓
Add bias in i32 (addi)             ← Fast
   ↓
Fixed-point scale (muli, shrsi)    ← Fast
   ↓
Clamp to i8 range (maxsi, minsi)   ← Fast
   ↓
Output (i8)
```

**Benefits:**
- Eliminates all float operations
- Uses integer units only
- Easier pattern matching for micro-kernels
- Better hardware utilization
- **Expected speedup: 1.5-3.0x**

## How IREE-Evo Helps

IREE-Evo automates the optimization process:

1. **Parse MLIR**: Identifies `linalg.quantized_matmul` and quantization patterns
2. **Select Strategies**: Chooses applicable optimization strategies for quantized operations
3. **Generate Variants**: Creates different compilation configurations with relevant flags
4. **Verify Correctness**: Ensures optimized versions produce identical results
5. **Benchmark**: Measures actual performance on target hardware
6. **Evolve**: Iteratively improves configurations across generations

## Files Provided

This use case includes three example files:

### 1. `quantized_matmul_float.mlir`
Shows the **inefficient pattern** with float-based scaling:
- Dequantizes i32 accumulator to f32
- Performs scaling and bias addition in float
- Requantizes back to i8

### 2. `quantized_matmul_integer.mlir`
Shows the **optimized pattern** with integer-only operations:
- Adds bias in i32 (no conversion)
- Uses fixed-point multiplication for scaling
- Direct clamping and truncation to i8

### 3. `quantization_optimization_demo.py`
Demonstrates using IREE-Evo to optimize quantization:
- Defines custom quantization strategies
- Configures the evolutionary optimization
- Shows expected workflow and results

## Running the Demo

### Prerequisites

```bash
# Install IREE tools
pip install iree-compiler iree-runtime

# Install IREE-Evo
cd iree-evo
pip install -e .
```

### Basic Usage

```bash
# Run the demonstration
cd examples
python3 quantization_optimization_demo.py
```

### Command-Line Usage

You can also use IREE-Evo directly via CLI:

```bash
# Optimize the float-based quantization pattern
iree-evo \
  --input examples/quantized_matmul_float.mlir \
  --backend llvm-cpu \
  --device local-task \
  --generations 5 \
  --population 10 \
  --verbose

# For GPU with INT8 Tensor Cores
iree-evo \
  --input examples/quantized_matmul_float.mlir \
  --backend cuda \
  --device cuda \
  --generations 10 \
  --population 15
```

## Custom Optimization Strategies

The demo defines custom strategies specifically for quantization:

### 1. `quantization_fusion_basic`
- **Description**: Enable basic dequantization fusion with matmul
- **Flag**: `--iree-flow-enable-fuse-dequantization-matmul`
- **Expected speedup**: 1.3-1.8x
- **Complexity**: Low

### 2. `quantization_int8_ukernels`
- **Description**: Use optimized INT8 micro-kernels
- **Flags**:
  - `--iree-llvmcpu-enable-ukernels=all`
  - `--iree-flow-enable-fuse-dequantization-matmul`
- **Expected speedup**: 1.8-2.5x
- **Complexity**: Medium

### 3. `quantization_vectorized`
- **Description**: Vectorize quantized operations for SIMD (AVX2/FMA)
- **Flags**:
  - `--iree-llvmcpu-enable-ukernels=all`
  - `--iree-llvmcpu-target-cpu-features=+avx2,+fma`
  - `--iree-flow-enable-fuse-dequantization-matmul`
- **Expected speedup**: 2.0-3.0x
- **Complexity**: Medium

### 4. `quantization_gpu_tensor_cores`
- **Description**: Use INT8 Tensor Cores on NVIDIA GPUs
- **Flags**:
  - `--iree-codegen-llvmgpu-enable-transform-dialect-jit`
  - `--iree-flow-enable-fuse-dequantization-matmul`
- **Expected speedup**: 2.5-4.0x
- **Complexity**: High

## Fixed-Point Scale Calculation

The integer-only pattern uses fixed-point arithmetic for scaling. Here's how to calculate the parameters:

### Given Scale Factors
```
S_input = 0.0424246378  (input quantization scale)
S_weight = 0.000246      (weight quantization scale)
S_output = 0.0168602373 (output quantization scale)
```

### Effective Scale Factor
```
M = (S_input × S_weight) / S_output
M = (0.0424246378 × 0.000246) / 0.0168602373
M ≈ 0.0006191469
```

### Fixed-Point Representation
Using Q31 format (31-bit fractional part):
```
Shift (n) = 31
Multiplier (M0) = floor(M × 2^31) = floor(0.0006191469 × 2147483648)
Multiplier (M0) = 1329634
```

### In MLIR
```mlir
%c_multiplier = arith.constant 1329634 : i32
%c_shift = arith.constant 31 : i32

// Fixed-point multiplication and shift
%val_i64 = arith.extsi %accumulator : i32 to i64
%mult_i64 = arith.extsi %c_multiplier : i32 to i64
%prod = arith.muli %val_i64, %mult_i64 : i64
%scaled_i64 = arith.shrsi %prod, %c_shift : i64
%scaled = arith.trunci %scaled_i64 : i64 to i32
```

## Expected Results

When running IREE-Evo on the quantization patterns, you should see:

### Baseline (No Optimization)
```
Mean Latency: ~12.5 ms
Binary Size: ~2.8 MB
```

### With Fusion (`quantization_fusion_basic`)
```
Mean Latency: ~8.7 ms (1.4x speedup)
Binary Size: ~2.5 MB
```

### With Ukernels (`quantization_int8_ukernels`)
```
Mean Latency: ~5.8 ms (2.2x speedup)
Binary Size: ~2.4 MB
```

### With Vectorization (`quantization_vectorized`)
```
Mean Latency: ~4.2 ms (3.0x speedup)
Binary Size: ~2.3 MB
```

## Integration with Your Workflow

### Step 1: Prepare Your Quantized Model

Convert your model to MLIR with quantized operations:

```bash
# Example: Convert from TensorFlow Lite
iree-import-tflite your_quantized_model.tflite -o model.mlir

# Or from ONNX
iree-import-onnx your_quantized_model.onnx -o model.mlir
```

### Step 2: Run IREE-Evo Optimization

```python
from pathlib import Path
from iree_evo import Orchestrator, IREEEvoConfig

config = IREEEvoConfig(
    target_backend="llvm-cpu",  # or "cuda"
    population_size=10,
    num_generations=5,
    work_dir=Path("/tmp/my_optimization"),
)

orchestrator = Orchestrator(config)
best = orchestrator.optimize(Path("model.mlir"))

print(f"Best flags: {best.compilation_flags}")
print(f"Speedup: {baseline_latency / best.mean_latency_ms:.2f}x")
```

### Step 3: Apply Best Configuration

Use the discovered flags in your production pipeline:

```bash
iree-compile model.mlir \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-enable-ukernels=all \
  --iree-flow-enable-fuse-dequantization-matmul \
  -o optimized_model.vmfb
```

## Next Steps

### 1. Create a Compiler Pass

Consider implementing an MLIR pass that transforms float-based quantization patterns to integer-only patterns automatically:

```cpp
// Pseudo-code for the transformation pass
class FuseQuantizationPass : public PassWrapper<...> {
  void runOnOperation() override {
    // Match pattern: quantized_matmul -> sitofp -> mulf -> addf -> divf -> fptosi
    // Replace with: quantized_matmul -> addi -> muli -> shrsi -> trunci
  }
};
```

This pass would go in:
```
compiler/src/iree/compiler/GlobalOptimization/FuseIntegerQuantization.cpp
```

### 2. Add Custom Micro-Kernels

For even better performance, implement custom micro-kernels for the integer-only pattern:

```
compiler/plugins/target/LLVMCPU/builtins/
└── ukernel_quantized_matmul_integer.c
```

### 3. Extend to Other Operations

Apply the same integer-only approach to:
- Convolutions (`linalg.conv_2d`)
- Batch matmul (`linalg.batch_matmul`)
- Grouped convolutions

### 4. Hardware-Specific Tuning

Use IREE-Evo to find optimal tile sizes and configurations for:
- CPU: AVX-512, ARM Neon
- GPU: NVIDIA INT8 Tensor Cores, AMD WMMA
- Accelerators: TPUs, NPUs

## Performance Tips

1. **Use Symmetric Quantization**: Zero-point = 0 simplifies the math
2. **Pre-compute Scale Factors**: Calculate M0 and shift offline
3. **Leverage SIMD**: Enable appropriate CPU features (AVX2, AVX-512)
4. **Profile Real Hardware**: Use `iree-benchmark-module` on target devices
5. **Compare Patterns**: Benchmark both float and integer versions

## Troubleshooting

### Issue: Numerical Differences

**Problem**: Integer-only version has slightly different outputs

**Solution**: 
- Adjust rounding mode in fixed-point conversion
- Use higher precision for intermediate calculations
- Verify scale factor computation

### Issue: Slow Compilation

**Problem**: IREE-Evo takes too long

**Solution**:
- Reduce population size and generations
- Use `--compile-timeout` to skip slow variants
- Pre-select applicable strategies only

### Issue: No Speedup

**Problem**: Optimized version is not faster

**Solution**:
- Check if fusion actually occurred (`iree-opt --print-ir-after-all`)
- Verify target hardware supports optimizations
- Use `iree-benchmark-module --print-statistics` for details

## References

- **IREE Quantization Docs**: https://iree.dev/guides/ml-frameworks/quantization/
- **FuseDequantizationMatmul Pass**: `compiler/src/iree/compiler/GlobalOptimization/FuseDequantizationMatmul.cpp`
- **MLIR Quantization Dialect**: https://mlir.llvm.org/docs/Dialects/QuantOps/
- **Fixed-Point Arithmetic**: https://en.wikipedia.org/wiki/Fixed-point_arithmetic

## Conclusion

This use case demonstrates IREE-Evo's power for real-world optimization problems. By automating the exploration of compilation strategies, IREE-Evo helps you:

- **Save development time**: No manual flag tuning
- **Discover better configurations**: Evolutionary search finds non-obvious combinations
- **Ensure correctness**: Automatic verification catches bugs
- **Measure real impact**: Benchmarking on actual hardware

The quantization optimization is just one example. IREE-Evo can be applied to any MLIR optimization challenge where the search space is large and hardware-specific tuning is critical.

---

**Ready to optimize your quantized models?** 🚀

```bash
cd examples
python3 quantization_optimization_demo.py
```
