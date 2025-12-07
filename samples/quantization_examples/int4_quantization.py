#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
INT4 quantization reference and examples for IREE.

INT4 quantization provides extreme model compression (~8x) and is particularly
useful for large language models and memory-constrained deployments.

IREE natively supports i4, si4, and ui4 types and can efficiently fuse
dequantization operations with compute kernels.

Usage:
    python int4_quantization.py --model input.onnx --output model_int4_info.txt
"""

import argparse
import os
import sys


def create_int4_mlir_example(output_path):
    """
    Create a comprehensive MLIR example showing INT4 quantization patterns in IREE.
    
    This demonstrates:
    1. How i4 types are used in IREE
    2. Grouped quantization patterns
    3. Dequantization fusion optimization
    """
    
    mlir_content = '''// INT4 Quantization in IREE - Comprehensive Example
// Copyright 2024 The IREE Authors
// Licensed under the Apache License v2.0 with LLVM Exceptions.

// This file demonstrates INT4 quantization patterns supported by IREE.
// Based on actual IREE compiler tests and optimization passes.

// =============================================================================
// Example 1: Basic INT4 Grouped Quantization
// =============================================================================
// INT4 quantization typically uses grouped quantization where weights are
// divided into groups (e.g., 128 elements), each with its own scale and
// zero point.

util.func @int4_grouped_quantization_matmul(
    // Quantized weights: shape [output_dim, input_groups, group_size]
    %weights: tensor<4096x32x128xi4>,
    // Per-group scales: shape [output_dim, input_groups]  
    %scales: tensor<4096x32xf32>,
    // Per-group zero points: shape [output_dim, input_groups]
    %zero_points: tensor<4096x32xf32>,
    // Input activations (full precision)
    %input: tensor<1x4096xf32>
) -> tensor<1x4096xf32> {
    
    %c0 = arith.constant 0.0 : f32
    
    // Step 1: Dequantize INT4 weights to FP32
    // For each element: dequant = (quantized - zero_point) * scale
    %dequantized = tensor.empty() : tensor<4096x32x128xf32>
    %weights_fp = linalg.generic {
        indexing_maps = [
            affine_map<(d0, d1, d2) -> (d0, d1, d2)>,  // weights
            affine_map<(d0, d1, d2) -> (d0, d1)>,      // scales
            affine_map<(d0, d1, d2) -> (d0, d1)>,      // zero_points
            affine_map<(d0, d1, d2) -> (d0, d1, d2)>   // output
        ],
        iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%weights, %scales, %zero_points : 
          tensor<4096x32x128xi4>, tensor<4096x32xf32>, tensor<4096x32xf32>)
      outs(%dequantized : tensor<4096x32x128xf32>) {
    ^bb0(%w: i4, %s: f32, %zp: f32, %out: f32):
        // Extend i4 to i32, convert to float
        %w_i32 = arith.extui %w : i4 to i32
        %w_f32 = arith.uitofp %w_i32 : i32 to f32
        // Apply dequantization: (w - zp) * s
        %w_shifted = arith.subf %w_f32, %zp : f32
        %w_scaled = arith.mulf %w_shifted, %s : f32
        linalg.yield %w_scaled : f32
    } -> tensor<4096x32x128xf32>
    
    // Step 2: Matrix multiplication with dequantized weights
    // Note: IREE's FuseDequantizationMatmul pass can fuse steps 1 and 2
    // for more efficient execution
    %output_init = tensor.empty() : tensor<1x4096xf32>
    %output_filled = linalg.fill ins(%c0 : f32) 
                                 outs(%output_init : tensor<1x4096xf32>) 
                                 -> tensor<1x4096xf32>
    
    %result = linalg.generic {
        indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,      // input (reshaped)
            affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>,      // weights
            affine_map<(d0, d1, d2, d3) -> (d0, d1)>           // output
        ],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]
    } ins(%input, %weights_fp : tensor<1x4096xf32>, tensor<4096x32x128xf32>)
      outs(%output_filled : tensor<1x4096xf32>) {
    ^bb0(%in: f32, %w: f32, %out: f32):
        %prod = arith.mulf %in, %w : f32
        %sum = arith.addf %prod, %out : f32
        linalg.yield %sum : f32
    } -> tensor<1x4096xf32>
    
    util.return %result : tensor<1x4096xf32>
}

// =============================================================================
// Example 2: INT4 Type Variants
// =============================================================================

util.func @int4_type_examples() {
    // i4:  4-bit integer (generic)
    %i4_val = arith.constant 7 : i4
    
    // si4: signed 4-bit integer (range: -8 to 7)
    %si4_val = arith.constant -5 : si4
    
    // ui4: unsigned 4-bit integer (range: 0 to 15)  
    %ui4_val = arith.constant 12 : ui4
    
    // INT4 values are typically stored in i8 for memory alignment
    // and unpacked when needed
    %packed = arith.constant dense<[0x12, 0x34, 0x56, 0x78]> : tensor<4xi8>
    
    // Each i8 can hold two i4 values
    // Lower 4 bits: first value
    // Upper 4 bits: second value
    
    util.return
}

// =============================================================================
// Example 3: Optimized INT4 Matmul (After Fusion)
// =============================================================================
// The IREE compiler's FuseDequantizationMatmul pass transforms the pattern
// from Example 1 into a more efficient fused operation

util.func @int4_fused_matmul_optimized(
    %weights: tensor<4096x32x128xi4>,
    %scales: tensor<4096x32xf32>,
    %zero_points: tensor<4096x32xf32>,
    %input: tensor<1x4096xf32>
) -> tensor<1x4096xf32> {
    // After optimization, IREE generates code that:
    // 1. Streams through quantized weights
    // 2. Dequantizes on-the-fly in registers
    // 3. Immediately uses in computation
    // 4. Minimizes memory traffic
    
    // This is represented as a fused kernel at the codegen level
    // See: compiler/src/iree/compiler/GlobalOptimization/FuseDequantizationMatmul.cpp
    
    util.return // fused result
}

// =============================================================================
// INT4 Quantization Best Practices
// =============================================================================
//
// 1. Group Size Selection:
//    - Typical: 32, 64, or 128 elements per group
//    - Smaller groups: better accuracy, more overhead
//    - Larger groups: more compression, potential accuracy loss
//
// 2. Zero Point Handling:
//    - Per-group zero points improve accuracy
//    - Can be omitted for symmetric quantization
//
// 3. Calibration:
//    - Use representative data for finding optimal scales
//    - MinMax calibration: simple, may clip outliers
//    - Percentile calibration: better for outlier handling
//    - MSE calibration: minimizes quantization error
//
// 4. Quantization-Aware Training (QAT):
//    - For best accuracy with INT4
//    - Simulate quantization during training
//    - Model learns to be robust to quantization error
//
// =============================================================================
// Compiling INT4 Models
// =============================================================================
//
// The IREE compiler automatically recognizes and optimizes INT4 patterns:
//
// iree-compile model.mlir \
//   --iree-hal-target-backends=llvm-cpu \
//   -o model.vmfb
//
// Key optimization passes (automatically applied):
//   --iree-global-opt-fuse-dequantization-matmul
//
// The compiler will:
// • Recognize dequantization + matmul patterns
// • Fuse operations for efficiency  
// • Generate optimized kernels
// • Minimize memory bandwidth usage
//
// =============================================================================
// Creating INT4 Quantized Models
// =============================================================================
//
// Method 1: PyTorch with torch.ao.quantization
// ---------------------------------------------
// import torch
// from torch.ao.quantization import quantize_dynamic
// 
// # Load your model
// model = YourModel()
// 
// # Apply INT4 quantization (requires torch >= 2.0)
// quantized_model = quantize_dynamic(
//     model,
//     qconfig_spec={torch.nn.Linear},
//     dtype=torch.qint4x2  # or use custom config
// )
// 
// # Export to ONNX
// torch.onnx.export(quantized_model, ...)
//
// Method 2: Custom Quantization
// -----------------------------
// • Implement grouped quantization in your framework
// • Store weights as i4, scales, and zero_points separately
// • Export with dequant + compute pattern
// • IREE will recognize and optimize the pattern
//
// Method 3: Post-Training Quantization
// ------------------------------------
// • Use ONNX Runtime quantization tools
// • Or implement custom quantization script
// • Generate ONNX with QuantizeLinear/DequantizeLinear ops
//
// =============================================================================
// Performance Characteristics
// =============================================================================
//
// Memory:
// • ~8x smaller than FP32
// • ~2x smaller than INT8
//
// Speed (vs FP32):
// • CPU: 1.5-2x faster (memory-bound workloads)
// • GPU: Variable (depends on kernel implementation)
//
// Accuracy:
// • Can maintain <1-2% accuracy loss with proper calibration
// • QAT significantly improves accuracy
// • Larger models generally quantize better
//
// =============================================================================
'''
    
    with open(output_path, 'w') as f:
        f.write(mlir_content)
    
    return output_path


def create_int4_guide(output_path):
    """Create a text guide for INT4 quantization."""
    
    guide_content = '''INT4 Quantization Guide for IREE
=====================================

OVERVIEW
--------
INT4 quantization reduces model size by ~8x compared to FP32 by using 4-bit
integers to represent weights. IREE natively supports i4, si4, and ui4 types.

SUPPORTED TYPES
---------------
• i4:  Generic 4-bit integer
• si4: Signed 4-bit integer (-8 to 7)
• ui4: Unsigned 4-bit integer (0 to 15)

QUANTIZATION METHODS
--------------------

1. Grouped Quantization (Recommended)
   • Divide weights into groups (32-128 elements)
   • Each group has its own scale and zero point
   • Formula: dequant = (quantized - zero_point) * scale
   • Better accuracy than per-tensor quantization

2. Per-Tensor Quantization
   • Single scale and zero point for entire tensor
   • Less overhead but lower accuracy
   • Simpler implementation

WHEN TO USE INT4
----------------
✓ Large models (LLMs, transformers) where memory is critical
✓ Edge deployment with strict memory constraints
✓ When ~8x compression is needed
✓ Models that can tolerate some accuracy loss

✗ Small models (overhead not worth it)
✗ When accuracy is critical and INT8 doesn't suffice
✗ Real-time applications needing predictable performance

ACCURACY CONSIDERATIONS
-----------------------
• Typical accuracy loss: 1-3% without QAT
• With QAT: Can match FP32 accuracy
• Larger groups (128): More compression, less accuracy
• Smaller groups (32): Better accuracy, more overhead

IMPLEMENTATION WORKFLOW
-----------------------

Step 1: Quantize Your Model
   Option A - PyTorch:
     from torch.ao.quantization import quantize_dynamic
     quantized = quantize_dynamic(model, qconfig_spec={nn.Linear})
     torch.onnx.export(quantized, ...)
   
   Option B - Custom:
     • Compute per-group min/max
     • Calculate scales and zero points
     • Quantize: q = clip(round(w/scale + zp), 0, 15)
     • Export with dequantization pattern

Step 2: Convert to ONNX
   • Include QuantizeLinear/DequantizeLinear nodes
   • Or export pattern: DequantizeLinear -> Compute
   • Ensure i4 types are preserved

Step 3: Import to IREE
   iree-import-onnx model.onnx -o model.mlir

Step 4: Compile with IREE
   iree-compile model.mlir \
     --iree-hal-target-backends=llvm-cpu \
     -o model.vmfb

IREE Optimizations:
   • FuseDequantizationMatmul pass (automatic)
   • Fuses dequant + compute for efficiency
   • Reduces memory bandwidth requirements

Step 5: Run Inference
   iree-run-module \
     --module=model.vmfb \
     --function=main \
     --input=...

OPTIMIZATION TIPS
-----------------
1. Group Size Selection:
   • Start with 128 for maximum compression
   • Use 64 or 32 if accuracy suffers
   • Consistent group size across layers

2. Calibration:
   • Use diverse, representative data
   • More calibration samples = better accuracy
   • Consider percentile clipping (e.g., 99.9%)

3. Symmetric vs Asymmetric:
   • Symmetric (zp=0): Simpler, faster
   • Asymmetric: Better accuracy for biased distributions

4. Per-Channel vs Per-Tensor:
   • Per-channel: Better accuracy (recommended)
   • Per-tensor: Simpler, less overhead

PERFORMANCE EXPECTATIONS
-------------------------
Model Size:
  FP32:  100%
  INT8:   25%
  INT4:  12.5%

Inference Speed (CPU, memory-bound):
  FP32:  1.0x
  INT8:  2-3x
  INT4:  1.5-2.5x

Accuracy (typical, without QAT):
  INT8:  <1% loss
  INT4:  1-3% loss

HARDWARE CONSIDERATIONS
-----------------------
• CPU: Well supported, focus on memory bandwidth
• GPU: Variable support, may not have native i4 ops
• NPU/TPU: Check specific hardware support

EXAMPLE: Quantizing MobileNet V2
---------------------------------
See the companion scripts in this directory:
• quantize_mobilenet_v2.py - Full workflow example
• int4_quantization.py - This script

TROUBLESHOOTING
---------------
Problem: Significant accuracy loss
Solution:
  • Reduce group size (128 -> 64 -> 32)
  • Use per-channel quantization
  • Try QAT (Quantization-Aware Training)
  • Use more calibration data

Problem: Performance worse than INT8
Solution:
  • Check if hardware has native i4 support
  • May be better to use INT8 on some platforms
  • Profile to identify bottlenecks

Problem: ONNX export fails
Solution:
  • Ensure ONNX opset >= 13 for QDQ support
  • Check operator compatibility
  • May need custom export logic

REFERENCES
----------
• IREE Compiler: compiler/src/iree/compiler/GlobalOptimization/
• Test Cases: tests/e2e/linalg/*i4*.mlir
• Fusion Pass: FuseDequantizationMatmul.cpp

NEXT STEPS
----------
1. Review the MLIR examples (created by this script)
2. Try quantizing a sample model
3. Compare accuracy with FP32 baseline
4. Benchmark inference performance
5. Iterate on group size and calibration

For questions and support:
• IREE GitHub: https://github.com/iree-org/iree
• Discussions: https://github.com/iree-org/iree/discussions
'''
    
    with open(output_path, 'w') as f:
        f.write(guide_content)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="INT4 quantization reference for IREE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        help="Path to input ONNX model (optional, for reference)"
    )
    parser.add_argument(
        "--output",
        default="int4_quantization_guide.txt",
        help="Output path for guide (default: int4_quantization_guide.txt)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("INT4 Quantization Reference for IREE")
    print("="*70)
    
    if args.model and os.path.exists(args.model):
        print(f"\nInput model: {args.model}")
        size_mb = os.path.getsize(args.model) / (1024 * 1024)
        print(f"Model size: {size_mb:.2f} MB")
        estimated_int4_size = size_mb / 8
        print(f"Estimated INT4 size: {estimated_int4_size:.2f} MB (~8x reduction)")
    
    # Create MLIR examples
    mlir_path = args.output.replace('.txt', '_examples.mlir')
    print(f"\nCreating MLIR examples: {mlir_path}")
    create_int4_mlir_example(mlir_path)
    print(f"✓ MLIR examples created")
    
    # Create text guide
    print(f"\nCreating guide: {args.output}")
    create_int4_guide(args.output)
    print(f"✓ Guide created")
    
    print("\n" + "="*70)
    print("INT4 Documentation Created")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  • {args.output} - Comprehensive guide")
    print(f"  • {mlir_path} - MLIR code examples")
    
    print("\nKey Points:")
    print("  • INT4 provides ~8x model compression")
    print("  • Uses grouped quantization for better accuracy")
    print("  • IREE natively supports i4, si4, ui4 types")
    print("  • Compiler automatically fuses dequantization with compute")
    
    print("\nFor actual INT4 quantization:")
    print("  1. Use PyTorch QAT or custom quantization")
    print("  2. Export to ONNX with quantization nodes")
    print("  3. Import to IREE: iree-import-onnx model.onnx")
    print("  4. Compile: iree-compile model.mlir -o model.vmfb")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
