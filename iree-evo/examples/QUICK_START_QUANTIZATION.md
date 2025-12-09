# Quick Reference: IREE-Evo for Quantization Optimization

## Problem
Quantized neural networks use inefficient float-based scaling:
```
i32 → f32 → scale → bias → requant → i8  (SLOW)
```

## Solution
Use integer-only operations with fixed-point math:
```
i32 → bias(i32) → fixed-point scale → i8  (FAST)
```

## Expected Speedup
- **Basic fusion**: 1.3-1.8x
- **With ukernels**: 1.8-2.5x
- **Vectorized (AVX2)**: 2.0-3.0x
- **GPU Tensor Cores**: 2.5-4.0x

## Quick Start

### 1. Run the Demo
```bash
cd iree-evo/examples
python3 quantization_optimization_demo.py
```

### 2. CLI Usage
```bash
iree-evo \
  --input quantized_matmul_float.mlir \
  --backend llvm-cpu \
  --generations 5 \
  --population 10
```

### 3. Programmatic Usage
```python
from iree_evo import Orchestrator, IREEEvoConfig

config = IREEEvoConfig(
    target_backend="llvm-cpu",
    population_size=10,
    num_generations=5,
)

orchestrator = Orchestrator(config)
best = orchestrator.optimize("model.mlir")
print(f"Best flags: {best.compilation_flags}")
```

## Key Flags for Quantization

### Basic Fusion
```bash
--iree-flow-enable-fuse-dequantization-matmul
```

### With Micro-kernels (CPU)
```bash
--iree-llvmcpu-enable-ukernels=all
--iree-flow-enable-fuse-dequantization-matmul
```

### Vectorized (AVX2/FMA)
```bash
--iree-llvmcpu-enable-ukernels=all
--iree-llvmcpu-target-cpu-features=+avx2,+fma
--iree-flow-enable-fuse-dequantization-matmul
```

### GPU Tensor Cores
```bash
--iree-codegen-llvmgpu-enable-transform-dialect-jit
--iree-flow-enable-fuse-dequantization-matmul
```

## Files Included

- **quantized_matmul_float.mlir** - Inefficient pattern (float-based)
- **quantized_matmul_integer.mlir** - Optimized pattern (integer-only)
- **quantization_optimization_demo.py** - Demo script
- **QUANTIZATION_USE_CASE.md** - Full guide

## How IREE-Evo Helps

1. ✅ Parses quantized MLIR patterns
2. ✅ Selects applicable optimization strategies
3. ✅ Generates compilation variants
4. ✅ Verifies correctness automatically
5. ✅ Benchmarks on real hardware
6. ✅ Evolves best configuration

## Next Steps

1. **Create a compiler pass** to transform patterns automatically
2. **Add custom micro-kernels** for integer-only operations
3. **Extend to other ops** (conv2d, batch_matmul)
4. **Tune for your hardware** using IREE-Evo

## Need Help?

See **QUANTIZATION_USE_CASE.md** for:
- Detailed explanation
- Fixed-point math calculation
- Integration guide
- Troubleshooting tips

---

**Start optimizing:** `python3 quantization_optimization_demo.py` 🚀
