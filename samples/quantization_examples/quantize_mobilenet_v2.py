#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Quantization examples for IREE demonstrating INT8, INT4, and FP8 formats.

This script shows how to apply different quantization formats to an ONNX model
(MobileNet V2 is used as an example) and prepare it for IREE compilation.

Usage:
    # Download MobileNet V2 ONNX model
    python quantize_mobilenet_v2.py --download
    
    # Generate all quantization formats
    python quantize_mobilenet_v2.py --model mobilenet_v2.onnx --all
    
    # Generate specific format
    python quantize_mobilenet_v2.py --model mobilenet_v2.onnx --format int8
    python quantize_mobilenet_v2.py --model mobilenet_v2.onnx --format int4
    python quantize_mobilenet_v2.py --model mobilenet_v2.onnx --format fp8
"""

import argparse
import os
import sys
from pathlib import Path

def download_mobilenet_v2():
    """Download MobileNet V2 ONNX model from ONNX model zoo."""
    print("Downloading MobileNet V2 ONNX model...")
    
    try:
        import urllib.request
        model_url = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
        output_path = "mobilenet_v2.onnx"
        
        if os.path.exists(output_path):
            print(f"Model already exists at {output_path}")
            return output_path
            
        print(f"Downloading from {model_url}...")
        urllib.request.urlretrieve(model_url, output_path)
        print(f"Model downloaded successfully to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("\nAlternatively, download manually from:")
        print("https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet")
        return None


def quantize_int8(model_path, output_path):
    """
    Apply INT8 dynamic quantization to the model.
    
    INT8 quantization is the most common and widely supported format.
    It provides ~4x size reduction and 2-4x inference speedup with minimal accuracy loss.
    """
    print(f"\n{'='*60}")
    print("INT8 Quantization")
    print(f"{'='*60}")
    
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        import onnx
        
        print(f"Input model: {model_path}")
        print(f"Output model: {output_path}")
        print("Quantization type: Dynamic INT8 (weights and activations)")
        
        # Apply dynamic quantization
        quantize_dynamic(
            model_input=model_path,
            model_output=output_path,
            weight_type=QuantType.QUInt8,  # or QuantType.QInt8
        )
        
        # Get model sizes
        original_size = os.path.getsize(model_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"\n✓ INT8 quantization completed successfully!")
        print(f"  Original size: {original_size:.2f} MB")
        print(f"  Quantized size: {quantized_size:.2f} MB")
        print(f"  Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")
        
        return output_path
        
    except ImportError:
        print("Error: onnxruntime not installed. Install with:")
        print("  pip install onnxruntime")
        return None
    except Exception as e:
        print(f"Error during INT8 quantization: {e}")
        return None


def quantize_int4_simulation(model_path, output_path):
    """
    Simulate INT4 quantization by creating MLIR with i4 types.
    
    INT4 quantization provides ~8x size reduction but requires careful calibration.
    IREE supports i4 types natively and can fuse dequantization operations.
    
    Note: This creates a representation that demonstrates how INT4 works in IREE.
    Real INT4 quantization typically requires custom quantization or QAT.
    """
    print(f"\n{'='*60}")
    print("INT4 Quantization")
    print(f"{'='*60}")
    
    print(f"Input model: {model_path}")
    print(f"Output representation: {output_path}")
    print("Quantization type: INT4 (grouped quantization)")
    
    # Create an example MLIR snippet showing INT4 usage
    mlir_example = '''// INT4 Quantization Example for IREE
// This demonstrates how IREE handles i4 (4-bit integer) types
//
// INT4 quantization in IREE typically uses:
// - i4/si4/ui4 types for weights
// - Grouped quantization with per-group scales and zero points
// - Dequantization fusion for efficient computation

// Example: Grouped INT4 matmul with dequantization
// Based on: compiler/src/iree/compiler/GlobalOptimization/test/fuse_dequantization_matmul.mlir

func.func @int4_quantized_matmul_example(%weights: tensor<?x?x?xi4>, 
                                          %scales: tensor<?x?xf32>,
                                          %zero_points: tensor<?x?xf32>,
                                          %input: tensor<?x?xf32>) -> tensor<?xf32> {
  // Step 1: Dequantize INT4 weights
  // The i4 values are extended to i32, converted to float, 
  // then scaled and shifted using per-group parameters
  
  // Step 2: Fused matmul
  // IREE's optimization passes can fuse dequantization with matmul
  // for efficient execution (see FuseDequantizationMatmul pass)
  
  // This pattern is automatically recognized and optimized by IREE
  // when compiling models with INT4 quantized weights
  
  return // result
}

// To use INT4 quantization with your model:
//
// 1. Apply INT4 quantization to weights (using PyTorch/QAT or custom tools)
// 2. Export to ONNX with appropriate quantization nodes
// 3. Import to IREE MLIR: iree-import-onnx model.onnx -o model.mlir
// 4. IREE compiler will recognize quantization patterns and optimize
// 5. Compile: iree-compile model.mlir --iree-hal-target-backends=llvm-cpu
//
// Key compilation flags for quantized models:
//   --iree-global-opt-fuse-dequantization-matmul
//   (automatically enabled in default optimization pipeline)
'''
    
    with open(output_path, 'w') as f:
        f.write(mlir_example)
    
    print(f"\n✓ INT4 example created at {output_path}")
    print("\nINT4 quantization notes:")
    print("  • IREE natively supports i4/si4/ui4 types")
    print("  • Use grouped quantization (e.g., 128 elements per group)")
    print("  • IREE automatically fuses dequantization with compute operations")
    print("  • For real INT4 quantization, use PyTorch QAT or ONNX quantization tools")
    print("\nTo apply INT4 quantization to your model:")
    print("  1. Use PyTorch's torch.ao.quantization with qint4 dtypes")
    print("  2. Or use ONNX Runtime quantization with custom INT4 config")
    print("  3. Export to ONNX and compile with IREE")
    
    return output_path


def quantize_fp8_simulation(model_path, output_path, format_type="e4m3"):
    """
    Create example showing FP8 quantization in IREE.
    
    FP8 formats are optimized for modern GPU architectures:
    - E4M3: Better for gradients/activations (4 exp bits, 3 mantissa bits)
    - E5M2: Better for weights (5 exp bits, 2 mantissa bits, wider range)
    
    Hardware support:
    - AMD MI300 series: E4M3FNUZ, E5M2FNUZ
    - NVIDIA Hopper+: E4M3FN, E5M2
    """
    print(f"\n{'='*60}")
    print(f"FP8 Quantization ({format_type.upper()})")
    print(f"{'='*60}")
    
    print(f"Input model: {model_path}")
    print(f"Output representation: {output_path}")
    print(f"Quantization type: FP8 {format_type.upper()}")
    
    format_info = {
        "e4m3": {
            "amd": "f8E4M3FNUZ",
            "nvidia": "f8E4M3FN",
            "desc": "4 exponent bits, 3 mantissa bits - good for activations"
        },
        "e5m2": {
            "amd": "f8E5M2FNUZ",
            "nvidia": "f8E5M2",
            "desc": "5 exponent bits, 2 mantissa bits - wider range for weights"
        }
    }
    
    info = format_info.get(format_type, format_info["e4m3"])
    
    mlir_example = f'''// FP8 {format_type.upper()} Quantization Example for IREE
// {info['desc']}
//
// Hardware-specific types:
//   AMD GPUs (gfx942, gfx950): {info['amd']}
//   NVIDIA GPUs (Hopper+):      {info['nvidia']}

// Example: FP8 matmul on AMD GPU
// Based on: compiler/plugins/target/ROCM/builtins/mlir_ukernel/iree_uk_amdgpu_matmul_{info['amd'].lower()}.mlir

func.func @fp8_{format_type}_matmul_example(%lhs: tensor<?x?x{info['amd']}>, 
                                             %rhs: tensor<?x?x{info['amd']}>) -> tensor<?x?xf32> {{
  // FP8 matrix multiplication with accumulation in FP32
  // This pattern is automatically recognized and mapped to
  // hardware-accelerated kernels on supported GPUs
  
  // AMD MI300: Uses MFMA (Matrix Fused Multiply-Add) instructions
  // NVIDIA Hopper: Uses Tensor Cores with FP8 support
  
  return // result : tensor<?x?xf32>
}}

// To compile for AMD GPU with FP8:
//   iree-compile model.mlir \\
//     --iree-hal-target-backends=rocm \\
//     --iree-rocm-target-chip=gfx942 \\
//     -o model.vmfb
//
// To compile for NVIDIA GPU with FP8:
//   iree-compile model.mlir \\
//     --iree-hal-target-backends=cuda \\
//     --iree-cuda-target=sm_90 \\
//     -o model.vmfb

// FP8 quantization workflow:
//
// 1. Train model with FP8-aware training (PyTorch, JAX, or Transformer Engine)
// 2. Export to ONNX with FP8 nodes (Cast operations)
// 3. Import to IREE: iree-import-onnx model.onnx -o model.mlir
// 4. Compile for target GPU with FP8 support
//
// The IREE compiler will automatically:
//   - Recognize FP8 types ({info['amd']}, {info['nvidia']})
//   - Map to hardware-accelerated kernels
//   - Optimize data layouts for FP8 tensor operations
'''
    
    with open(output_path, 'w') as f:
        f.write(mlir_example)
    
    print(f"\n✓ FP8 {format_type.upper()} example created at {output_path}")
    print(f"\nFP8 {format_type.upper()} format details:")
    print(f"  • {info['desc']}")
    print(f"  • AMD type: {info['amd']}")
    print(f"  • NVIDIA type: {info['nvidia']}")
    print("\nHardware requirements:")
    print("  • AMD: MI300 series (gfx942, gfx950)")
    print("  • NVIDIA: Hopper architecture or newer (SM 90+)")
    print("\nPerformance benefits:")
    print("  • ~4x size reduction vs FP32")
    print("  • Hardware-accelerated on supported GPUs")
    print("  • Better accuracy than INT8 for many models")
    
    return output_path


def create_summary_file(results):
    """Create a summary file with information about all generated quantized models."""
    summary_path = "quantization_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("IREE Quantization Summary\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Generated Quantization Examples:\n\n")
        
        for format_name, file_path in results.items():
            if file_path and os.path.exists(file_path):
                size = os.path.getsize(file_path) / 1024  # KB
                f.write(f"✓ {format_name.upper()}: {file_path} ({size:.2f} KB)\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("Next Steps: Compile with IREE\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("For INT8 quantized model:\n")
        f.write("  # Import ONNX to MLIR\n")
        f.write("  iree-import-onnx mobilenet_v2_int8.onnx -o mobilenet_v2_int8.mlir\n\n")
        f.write("  # Compile for CPU\n")
        f.write("  iree-compile mobilenet_v2_int8.mlir \\\n")
        f.write("    --iree-hal-target-backends=llvm-cpu \\\n")
        f.write("    -o mobilenet_v2_int8_cpu.vmfb\n\n")
        f.write("  # Compile for GPU\n")
        f.write("  iree-compile mobilenet_v2_int8.mlir \\\n")
        f.write("    --iree-hal-target-backends=cuda \\\n")
        f.write("    -o mobilenet_v2_int8_cuda.vmfb\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("Supported Quantization Types in IREE\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Integer Types:\n")
        f.write("  • INT8 (i8, si8, ui8)  - Widely supported, best compatibility\n")
        f.write("  • INT4 (i4, si4, ui4)  - Extreme compression, needs calibration\n\n")
        
        f.write("Floating Point Types:\n")
        f.write("  • FP8 E4M3FNUZ - AMD GPU optimized\n")
        f.write("  • FP8 E4M3FN   - NVIDIA GPU optimized\n")
        f.write("  • FP8 E5M2FNUZ - AMD GPU, wider range\n")
        f.write("  • FP8 E5M2     - NVIDIA GPU, wider range\n")
        f.write("  • FP4 E2M1FN   - Experimental, research only\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("Performance Expectations\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("INT8:  ~4x size reduction, 2-4x speedup, <1% accuracy loss\n")
        f.write("INT4:  ~8x size reduction, variable speedup, accuracy depends on calibration\n")
        f.write("FP8:   ~4x size reduction, GPU-accelerated, good accuracy retention\n")
        f.write("FP4:   Experimental, significant accuracy challenges\n\n")
        
    print(f"\n✓ Summary written to {summary_path}")
    return summary_path


def main():
    parser = argparse.ArgumentParser(
        description="Quantization examples for IREE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to input ONNX model (default: mobilenet_v2.onnx)"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download MobileNet V2 ONNX model"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["int8", "int4", "fp8", "fp8_e4m3", "fp8_e5m2"],
        help="Quantization format to apply"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all quantization format examples"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for quantized models (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # Download model if requested
    if args.download:
        model_path = download_mobilenet_v2()
        if not model_path:
            return 1
        print("\nModel downloaded! Now run with --all or --format to quantize it.")
        return 0
    
    # Require model path
    if not args.model:
        if not os.path.exists("mobilenet_v2.onnx"):
            print("Error: No model specified and mobilenet_v2.onnx not found.")
            print("Run with --download first, or specify --model <path>")
            return 1
        args.model = "mobilenet_v2.onnx"
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate base name for outputs
    model_name = Path(args.model).stem
    
    results = {}
    
    # Process based on arguments
    if args.all or args.format == "int8":
        output_path = os.path.join(args.output_dir, f"{model_name}_int8.onnx")
        results["int8"] = quantize_int8(args.model, output_path)
    
    if args.all or args.format == "int4":
        output_path = os.path.join(args.output_dir, f"{model_name}_int4_example.mlir")
        results["int4"] = quantize_int4_simulation(args.model, output_path)
    
    if args.all or args.format in ["fp8", "fp8_e4m3"]:
        output_path = os.path.join(args.output_dir, f"{model_name}_fp8_e4m3_example.mlir")
        results["fp8_e4m3"] = quantize_fp8_simulation(args.model, output_path, "e4m3")
    
    if args.all or args.format == "fp8_e5m2":
        output_path = os.path.join(args.output_dir, f"{model_name}_fp8_e5m2_example.mlir")
        results["fp8_e5m2"] = quantize_fp8_simulation(args.model, output_path, "e5m2")
    
    # Create summary
    if results:
        create_summary_file(results)
        print("\n" + "="*70)
        print("Quantization complete! See quantization_summary.txt for details.")
        print("="*70)
    else:
        print("\nNo quantization format specified. Use --all or --format <type>")
        print("Available formats: int8, int4, fp8_e4m3, fp8_e5m2")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
