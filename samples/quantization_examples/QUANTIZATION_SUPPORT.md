# Quantization Support in IREE - Summary

This document summarizes the quantization capabilities in IREE and the example scripts provided in this directory.

## Quantization Types Supported by IREE

Based on investigation of the IREE compiler source code and test files, IREE supports the following quantization formats:

### 1. Integer Quantization

#### INT8 (i8, si8, ui8)
- **Status**: Fully supported, production-ready
- **Use case**: Most common quantization format for deployment
- **Performance**: ~4x size reduction, 2-4x speedup vs FP32
- **Accuracy**: Typically <1% loss with proper calibration
- **Hardware**: Supported on all CPU and GPU backends
- **Location in code**: 
  - Type definitions: `compiler/src/iree/compiler/Dialect/Flow/IR/FlowBase.td`
  - Optimization passes: `compiler/src/iree/compiler/GlobalOptimization/`

#### INT4 (i4, si4, ui4)
- **Status**: Supported, used for extreme compression
- **Use case**: Large models (LLMs), memory-constrained deployments
- **Performance**: ~8x size reduction vs FP32
- **Accuracy**: 1-3% loss (requires careful calibration or QAT)
- **Hardware**: Supported on CPU and GPU, may emulate on some platforms
- **Features**: 
  - Grouped quantization with per-group scales and zero points
  - Automatic dequantization fusion via `FuseDequantizationMatmul` pass
- **Location in code**:
  - Examples: `tests/e2e/linalg/*i4*.mlir`
  - Fusion pass: `compiler/src/iree/compiler/GlobalOptimization/FuseDequantizationMatmul.cpp`

### 2. Floating Point Quantization

#### FP8 E4M3FNUZ
- **Status**: Supported, optimized for AMD GPUs
- **Hardware**: AMD MI300 series (gfx942, gfx950)
- **Format**: 4 exponent bits, 3 mantissa bits (FN = Finite, No NaN; UZ = Unsigned Zero)
- **Use case**: Activations and gradients on AMD GPUs
- **Performance**: Hardware-accelerated matrix operations via MFMA instructions
- **Location in code**:
  - Types: `compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.cpp`
  - Kernels: `compiler/plugins/target/ROCM/builtins/mlir_ukernel/*f8E4M3FNUZ*`

#### FP8 E4M3FN
- **Status**: Supported, optimized for NVIDIA GPUs
- **Hardware**: NVIDIA Hopper architecture (H100, H200), SM 90+
- **Format**: 4 exponent bits, 3 mantissa bits (IEEE-like representation)
- **Use case**: Activations and gradients on NVIDIA GPUs
- **Performance**: Hardware-accelerated via Tensor Cores
- **Location in code**: Same as E4M3FNUZ

#### FP8 E5M2FNUZ
- **Status**: Supported, optimized for AMD GPUs
- **Hardware**: AMD MI300 series (gfx942, gfx950)
- **Format**: 5 exponent bits, 2 mantissa bits (wider range)
- **Use case**: Weights on AMD GPUs (wider range needed)
- **Location in code**:
  - Kernels: `compiler/plugins/target/ROCM/builtins/mlir_ukernel/*f8E5M2FNUZ*`

#### FP8 E5M2
- **Status**: Supported, optimized for NVIDIA GPUs
- **Hardware**: NVIDIA Hopper architecture (H100, H200)
- **Format**: 5 exponent bits, 2 mantissa bits (wider range)
- **Use case**: Weights on NVIDIA GPUs
- **Location in code**: Same as other FP8 types

#### FP4 E2M1FN
- **Status**: Experimental
- **Format**: 2 exponent bits, 1 mantissa bit
- **Use case**: Research purposes only
- **Limitations**: Significant accuracy challenges, limited practical use
- **Location in code**:
  - Tests: `tests/e2e/linalg/fp4_f32_conversion.mlir`

### 3. Not Supported

The following formats are **NOT** natively supported in IREE:
- FP4 for production use (only experimental)
- INT2 or INT3 quantization
- Binary neural networks (1-bit)
- Ternary quantization (2-bit with -1, 0, +1 values)

## Scripts Provided

### 1. quantize_mobilenet_v2.py
Main script demonstrating all quantization formats with MobileNet V2 as an example.

**Features**:
- Downloads MobileNet V2 ONNX model automatically
- Generates INT8 quantized model using ONNX Runtime
- Creates INT4 and FP8 example MLIR files showing quantization patterns
- Produces summary document with next steps

**Usage**:
```bash
# Download model
python quantize_mobilenet_v2.py --download

# Generate all formats
python quantize_mobilenet_v2.py --model mobilenet_v2.onnx --all

# Generate specific format
python quantize_mobilenet_v2.py --model mobilenet_v2.onnx --format int8
```

### 2. int8_quantization.py
Focused script for INT8 quantization using ONNX Runtime.

**Features**:
- Dynamic quantization (no calibration needed)
- Static quantization (with calibration data)
- Support for both QUInt8 and QInt8

**Usage**:
```bash
python int8_quantization.py --model input.onnx --output output_int8.onnx
python int8_quantization.py --model input.onnx --output output_int8.onnx --static
```

### 3. int4_quantization.py
Reference documentation and MLIR examples for INT4 quantization.

**Features**:
- Comprehensive MLIR examples showing i4 types
- Grouped quantization patterns
- Dequantization fusion documentation
- Best practices guide

**Usage**:
```bash
python int4_quantization.py --output int4_guide.txt
python int4_quantization.py --model model.onnx --output guide.txt
```

### 4. fp8_quantization.py
Reference documentation for FP8 quantization on GPUs.

**Features**:
- Support for all FP8 variants (E4M3FN, E4M3FNUZ, E5M2, E5M2FNUZ)
- Hardware-specific guidance
- MLIR examples with tensor operations
- Calibration strategies

**Usage**:
```bash
python fp8_quantization.py --format e4m3fn --output fp8_guide.txt
python fp8_quantization.py --format e5m2fnuz --output fp8_amd_guide.txt
```

## Compilation Workflow

After quantizing a model, compile it with IREE:

### For CPU
```bash
# Import ONNX to MLIR
iree-import-onnx model_quantized.onnx -o model.mlir

# Compile for CPU
iree-compile model.mlir \
  --iree-hal-target-backends=llvm-cpu \
  -o model.vmfb
```

### For NVIDIA GPU (FP8)
```bash
iree-compile model.mlir \
  --iree-hal-target-backends=cuda \
  --iree-cuda-target=sm_90 \
  -o model.vmfb
```

### For AMD GPU (FP8)
```bash
iree-compile model.mlir \
  --iree-hal-target-backends=rocm \
  --iree-rocm-target-chip=gfx942 \
  -o model.vmfb
```

## Key IREE Compiler Features for Quantization

1. **Automatic Pattern Recognition**: IREE recognizes quantization/dequantization patterns
2. **Fusion Optimization**: `FuseDequantizationMatmul` fuses dequant + compute operations
3. **Hardware Mapping**: Automatically maps to hardware-accelerated kernels (e.g., MFMA, Tensor Cores)
4. **Type Support**: Native support for i4, i8, f8 types throughout the compilation stack

## Performance Expectations

| Format | Size Reduction | Speed Improvement | Accuracy Loss | Hardware Support |
|--------|----------------|-------------------|---------------|------------------|
| INT8   | ~4x            | 2-4x              | <1%           | Universal        |
| INT4   | ~8x            | 1.5-2.5x          | 1-3%          | CPU, GPU         |
| FP8    | ~4x            | 2-4x (GPU)        | <0.5%         | Modern GPUs only |
| FP4    | ~8x            | Varies            | Significant   | Limited          |

## References

- IREE Compiler Source: `compiler/src/iree/compiler/`
- Quantization Tests: `tests/e2e/linalg/`
- FuseDequantizationMatmul: `compiler/src/iree/compiler/GlobalOptimization/FuseDequantizationMatmul.cpp`
- ROCM FP8 Kernels: `compiler/plugins/target/ROCM/builtins/`
- GPU Dialect: `compiler/src/iree/compiler/Codegen/Dialect/GPU/`

## Conclusion

IREE provides comprehensive support for:
- ✅ INT8 (production-ready, universal)
- ✅ INT4 (production-ready, grouped quantization)
- ✅ FP8 (production-ready, GPU-specific)
- ⚠️ FP4 (experimental only)

The provided scripts demonstrate how to:
1. Apply quantization to ONNX models
2. Understand IREE's quantization patterns
3. Compile quantized models for different backends
4. Achieve significant model compression and speedup
