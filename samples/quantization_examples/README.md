# IREE Quantization Examples

This directory contains examples demonstrating quantization support in IREE for various precision formats.

## Supported Quantization Types

IREE supports the following quantization formats:

### Integer Quantization
- **INT8 (i8/si8/ui8)**: 8-bit integer quantization - widely supported and most common for deployment
- **INT4 (i4/si4/ui4)**: 4-bit integer quantization - for extreme compression with acceptable accuracy loss

### Floating Point Quantization  
- **FP8 E4M3FNUZ**: 8-bit floating point with 4 exponent bits and 3 mantissa bits (AMD GPU optimized)
- **FP8 E4M3FN**: 8-bit floating point with 4 exponent bits and 3 mantissa bits (NVIDIA GPU optimized)
- **FP8 E5M2FNUZ**: 8-bit floating point with 5 exponent bits and 2 mantissa bits (wider range)
- **FP8 E5M2**: 8-bit floating point with 5 exponent bits and 2 mantissa bits
- **FP4 E2M1FN**: 4-bit floating point with 2 exponent bits and 1 mantissa bit (experimental)

**Note**: FP4 support is experimental and primarily for research purposes. FP8 formats are optimized for specific GPU architectures (AMD MI300 series, NVIDIA Hopper+).

## Hardware Support

Different quantization types are optimized for different hardware:

- **INT8/INT4**: Supported on most CPU and GPU backends
- **FP8 E4M3FNUZ/E5M2FNUZ**: Optimized for AMD GPUs (gfx942, gfx950)
- **FP8 E4M3FN/E5M2**: Optimized for NVIDIA GPUs with FP8 tensor cores
- **FP4**: Experimental, limited hardware support

## Scripts

This directory contains example scripts for applying quantization to ONNX models:

1. `quantize_mobilenet_v2.py` - Complete example showing INT8, INT4, and FP8 quantization workflows
2. `int8_quantization.py` - INT8 quantization using ONNX Runtime quantization
3. `int4_quantization.py` - INT4 grouped quantization example
4. `fp8_quantization.py` - FP8 quantization for GPU deployment

## Prerequisites

```bash
# Install IREE compiler
pip install iree-compiler

# Install ONNX and quantization tools
pip install onnx onnxruntime onnxruntime-tools

# For PyTorch model export (if needed)
pip install torch torchvision
```

## Usage

### Quick Start with MobileNet V2

```bash
# Download MobileNet V2 ONNX model
python quantize_mobilenet_v2.py --download

# Generate all quantization formats
python quantize_mobilenet_v2.py --model mobilenet_v2.onnx --all

# Or generate specific format
python quantize_mobilenet_v2.py --model mobilenet_v2.onnx --format int8
python quantize_mobilenet_v2.py --model mobilenet_v2.onnx --format int4
python quantize_mobilenet_v2.py --model mobilenet_v2.onnx --format fp8
```

### INT8 Quantization

```bash
python int8_quantization.py --model mobilenet_v2.onnx --output mobilenet_v2_int8.onnx
```

This uses dynamic quantization for weights and static quantization for activations (requires calibration data).

### INT4 Quantization  

```bash
python int4_quantization.py --model mobilenet_v2.onnx --output mobilenet_v2_int4.onnx
```

INT4 quantization uses grouped quantization with separate scales and zero points per group.

### FP8 Quantization

```bash
python fp8_quantization.py --model mobilenet_v2.onnx --format e4m3 --output mobilenet_v2_fp8.onnx
```

FP8 formats are designed for GPU inference with hardware acceleration.

## Compiling Quantized Models with IREE

After quantizing your ONNX model, compile it for your target backend:

```bash
# Import ONNX to MLIR
iree-import-onnx mobilenet_v2_quantized.onnx -o mobilenet_v2.mlir

# Compile for CPU
iree-compile mobilenet_v2.mlir \
    --iree-hal-target-backends=llvm-cpu \
    -o mobilenet_v2_cpu.vmfb

# Compile for CUDA GPU
iree-compile mobilenet_v2.mlir \
    --iree-hal-target-backends=cuda \
    -o mobilenet_v2_cuda.vmfb

# Compile for AMD GPU with FP8 support
iree-compile mobilenet_v2.mlir \
    --iree-hal-target-backends=rocm \
    --iree-rocm-target-chip=gfx942 \
    -o mobilenet_v2_rocm.vmfb
```

## Performance Considerations

- **INT8**: ~4x smaller models, 2-4x faster inference vs FP32, <1% accuracy loss
- **INT4**: ~8x smaller models, potential accuracy degradation, requires careful calibration
- **FP8**: ~4x smaller models, hardware-accelerated on modern GPUs, good accuracy retention
- **FP4**: Experimental, significant accuracy challenges

## Quantization-Aware Training

For best accuracy with INT8/INT4 quantization, consider using Quantization-Aware Training (QAT) with PyTorch or TensorFlow before exporting to ONNX.

## Additional Resources

- [IREE Compiler Documentation](https://iree.dev/guides/ml-frameworks/)
- [ONNX Runtime Quantization Guide](https://onnxruntime.ai/docs/performance/quantization.html)
- [IREE Global Optimization Passes](../../compiler/src/iree/compiler/GlobalOptimization/)

## Troubleshooting

### Model Compatibility
Not all ONNX operators support quantization. Check operator support in IREE documentation.

### Accuracy Issues
- Use calibration data representative of your actual use case
- Try Per-Channel quantization for better accuracy
- Consider QAT for models with significant accuracy degradation

### Hardware Requirements
FP8 quantization requires specific GPU hardware. Ensure your target device supports the chosen format.
