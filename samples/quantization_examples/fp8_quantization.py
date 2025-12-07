#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
FP8 quantization reference and examples for IREE.

FP8 (8-bit floating point) quantization is optimized for modern GPU architectures
with hardware acceleration for FP8 tensor operations.

IREE supports multiple FP8 formats:
• E4M3FNUZ / E4M3FN: 4 exponent bits, 3 mantissa bits (for activations)
• E5M2FNUZ / E5M2:  5 exponent bits, 2 mantissa bits (wider range, for weights)

Usage:
    python fp8_quantization.py --format e4m3fn --output fp8_e4m3_guide.txt
    python fp8_quantization.py --format e5m2 --output fp8_e5m2_guide.txt
"""

import argparse
import os
import sys


def create_fp8_mlir_example(format_type, output_path):
    """
    Create MLIR examples showing FP8 quantization patterns in IREE.
    
    Args:
        format_type: FP8 format ('e4m3fn', 'e4m3fnuz', 'e5m2', 'e5m2fnuz')
        output_path: Path to save MLIR example file
    """
    
    format_info = {
        'e4m3fn': {
            'mlir_type': 'f8E4M3FN',
            'hardware': 'NVIDIA Hopper (SM 90+)',
            'target': 'cuda',
            'chip': 'sm_90',
            'desc': '4 exp bits, 3 mantissa bits - IEEE-like, good for activations'
        },
        'e4m3fnuz': {
            'mlir_type': 'f8E4M3FNUZ',
            'hardware': 'AMD MI300 (gfx942, gfx950)',
            'target': 'rocm',
            'chip': 'gfx942',
            'desc': '4 exp bits, 3 mantissa bits - AMD variant with different NaN/Inf'
        },
        'e5m2': {
            'mlir_type': 'f8E5M2',
            'hardware': 'NVIDIA Hopper (SM 90+)',
            'target': 'cuda',
            'chip': 'sm_90',
            'desc': '5 exp bits, 2 mantissa bits - wider range, good for weights'
        },
        'e5m2fnuz': {
            'mlir_type': 'f8E5M2FNUZ',
            'hardware': 'AMD MI300 (gfx942, gfx950)',
            'target': 'rocm',
            'chip': 'gfx942',
            'desc': '5 exp bits, 2 mantissa bits - AMD variant, wider range'
        }
    }
    
    info = format_info.get(format_type, format_info['e4m3fn'])
    mlir_type = info['mlir_type']
    
    mlir_content = f'''// FP8 {format_type.upper()} Quantization in IREE - Comprehensive Example
// Copyright 2024 The IREE Authors
// Licensed under the Apache License v2.0 with LLVM Exceptions.

// This file demonstrates {format_type.upper()} quantization patterns supported by IREE.
// Format: {info['desc']}
// Hardware: {info['hardware']}

// =============================================================================
// FP8 Format Details: {format_type.upper()}
// =============================================================================
// Type: {mlir_type}
// Precision: {info['desc']}
// Hardware acceleration: {info['hardware']}
//
// Compared to FP32:
// • Memory: 4x reduction
// • Performance: Hardware accelerated matmul on supported GPUs
// • Accuracy: Better than INT8 for many models
//
// Compared to INT8:
// • Better representation of floating point distributions
// • No need for zero points (direct float representation)
// • Hardware acceleration on modern GPUs

// =============================================================================
// Example 1: Basic FP8 Matrix Multiplication
// =============================================================================

func.func @fp8_matmul_basic(
    %lhs: tensor<1024x2048x{mlir_type}>,
    %rhs: tensor<2048x4096x{mlir_type}>
) -> tensor<1024x4096xf32> {{
    
    // Initialize output in FP32 (accumulation precision)
    %init = tensor.empty() : tensor<1024x4096xf32>
    %c0 = arith.constant 0.0 : f32
    %output = linalg.fill ins(%c0 : f32) outs(%init : tensor<1024x4096xf32>) 
                         -> tensor<1024x4096xf32>
    
    // FP8 matrix multiplication with FP32 accumulation
    // IREE will map this to hardware-accelerated kernels
    %result = linalg.matmul 
        ins(%lhs, %rhs : tensor<1024x2048x{mlir_type}>, tensor<2048x4096x{mlir_type}>)
        outs(%output : tensor<1024x4096xf32>) 
        -> tensor<1024x4096xf32>
    
    // On {info['hardware']}, this becomes:
    // • FP8 tensor core operations
    // • High throughput (e.g., 2-4x FP16 performance)
    // • FP32 accumulation for accuracy
    
    return %result : tensor<1024x4096xf32>
}}

// =============================================================================
// Example 2: FP8 with Scaling
// =============================================================================
// FP8 quantization typically uses per-tensor or per-channel scaling
// to maximize the effective range

func.func @fp8_scaled_matmul(
    %lhs: tensor<1024x2048x{mlir_type}>,
    %rhs: tensor<2048x4096x{mlir_type}>,
    %lhs_scale: f32,
    %rhs_scale: f32
) -> tensor<1024x4096xf32> {{
    
    %c0 = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<1024x4096xf32>
    %output = linalg.fill ins(%c0 : f32) outs(%init : tensor<1024x4096xf32>) 
                         -> tensor<1024x4096xf32>
    
    // Compute: output = (lhs * lhs_scale) @ (rhs * rhs_scale)
    // Can be optimized to: output = lhs @ rhs * (lhs_scale * rhs_scale)
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<1024x2048x{mlir_type}>, tensor<2048x4096x{mlir_type}>)
        outs(%output : tensor<1024x4096xf32>)
        -> tensor<1024x4096xf32>
    
    // Apply combined scaling factor
    %scale_combined = arith.mulf %lhs_scale, %rhs_scale : f32
    %scaled_result = linalg.generic {{
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
    }} ins(%result : tensor<1024x4096xf32>)
       outs(%init : tensor<1024x4096xf32>) {{
    ^bb0(%in: f32, %out: f32):
        %scaled = arith.mulf %in, %scale_combined : f32
        linalg.yield %scaled : f32
    }} -> tensor<1024x4096xf32>
    
    return %scaled_result : tensor<1024x4096xf32>
}}

// =============================================================================
// Example 3: FP32 to FP8 Conversion
// =============================================================================

func.func @quantize_fp32_to_fp8(
    %input: tensor<1024x2048xf32>,
    %scale: f32
) -> tensor<1024x2048x{mlir_type}> {{
    
    %output = tensor.empty() : tensor<1024x2048x{mlir_type}>
    
    %quantized = linalg.generic {{
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
    }} ins(%input : tensor<1024x2048xf32>)
       outs(%output : tensor<1024x2048x{mlir_type}>) {{
    ^bb0(%in: f32, %out: {mlir_type}):
        // Scale input to FP8 range
        %scaled = arith.divf %in, %scale : f32
        // Truncate to FP8 (with rounding)
        %fp8_val = arith.truncf %scaled : f32 to {mlir_type}
        linalg.yield %fp8_val : {mlir_type}
    }} -> tensor<1024x2048x{mlir_type}>
    
    return %quantized : tensor<1024x2048x{mlir_type}>
}}

// =============================================================================
// Example 4: FP8 to FP32 Conversion (Dequantization)
// =============================================================================

func.func @dequantize_fp8_to_fp32(
    %input: tensor<1024x2048x{mlir_type}>,
    %scale: f32
) -> tensor<1024x2048xf32> {{
    
    %output = tensor.empty() : tensor<1024x2048xf32>
    
    %dequantized = linalg.generic {{
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
    }} ins(%input : tensor<1024x2048x{mlir_type}>)
       outs(%output : tensor<1024x2048xf32>) {{
    ^bb0(%in: {mlir_type}, %out: f32):
        // Extend FP8 to FP32
        %fp32_val = arith.extf %in : {mlir_type} to f32
        // Apply scale
        %scaled = arith.mulf %fp32_val, %scale : f32
        linalg.yield %scaled : f32
    }} -> tensor<1024x2048xf32>
    
    return %dequantized : tensor<1024x2048xf32>
}}

// =============================================================================
// Compilation for {info['hardware']}
// =============================================================================
//
// To compile this code for {info['hardware']}:
//
// iree-compile fp8_example.mlir \\
//   --iree-hal-target-backends={info['target']} \\
//   --iree-{info['target']}-target={info['chip']} \\
//   -o model.vmfb
//
// The IREE compiler will:
// • Recognize FP8 types ({mlir_type})
// • Map to hardware-accelerated kernels
// • Optimize memory layouts for tensor cores
// • Generate efficient {info['target'].upper()} code

// =============================================================================
// FP8 Quantization Workflow
// =============================================================================
//
// Step 1: Prepare Model
//   • Train in FP32/FP16/BF16
//   • Optional: FP8-aware training (better accuracy)
//
// Step 2: Determine Scaling Factors
//   • Per-tensor: One scale per tensor (simpler)
//   • Per-channel: One scale per output channel (better accuracy)
//   • Use calibration data to find optimal scales
//
// Step 3: Convert Weights
//   • weight_fp8 = clip(round(weight_fp32 / scale), fp8_min, fp8_max)
//   • Store as {mlir_type} tensors
//
// Step 4: Export to ONNX/MLIR
//   • Include Cast operations: FP32 -> {mlir_type}
//   • Include scaling metadata
//
// Step 5: Compile with IREE
//   • Import: iree-import-onnx model.onnx -o model.mlir
//   • Compile for target GPU
//   • IREE optimizes FP8 operations automatically
//
// Step 6: Run Inference
//   • IREE runtime handles FP8 tensors efficiently
//   • Hardware acceleration on supported GPUs

// =============================================================================
// Performance Characteristics
// =============================================================================
//
// Memory Bandwidth:
//   FP32: 4 bytes/element
//   {mlir_type}: 1 byte/element (4x reduction)
//
// Compute Performance ({info['hardware']}):
//   FP32: Baseline
//   FP16: ~2x faster
//   {mlir_type}: ~2-4x faster than FP16 (hardware dependent)
//
// Accuracy:
//   • Better than INT8 for most models
//   • E4M3: Better for activations (more precision)
//   • E5M2: Better for weights (wider range)
//   • Typical accuracy loss: <0.5% vs FP16

// =============================================================================
// Best Practices
// =============================================================================
//
// 1. Format Selection:
//    • E4M3 for activations (better precision)
//    • E5M2 for weights (wider range)
//    • Mixed precision: E5M2 weights, E4M3 activations
//
// 2. Scaling:
//    • Use per-channel for better accuracy
//    • Calibrate with representative data
//    • Consider delayed scaling for training
//
// 3. Hardware Targeting:
//    • Verify GPU supports FP8 ({info['hardware']})
//    • Match FP8 format to hardware (FNUZ for AMD, FN for NVIDIA)
//    • Profile to confirm speedup
//
// 4. Testing:
//    • Compare accuracy against FP32 baseline
//    • Test on diverse inputs
//    • Monitor for numerical instability

// =============================================================================
// References
// =============================================================================
//
// • IREE GPU Dialect: compiler/src/iree/compiler/Codegen/Dialect/GPU/
// • ROCM FP8 Kernels: compiler/plugins/target/ROCM/builtins/mlir_ukernel/
// • FP8 Tests: compiler/plugins/target/ROCM/test/*fp8*.mlir
// • FP8 Specification: https://arxiv.org/abs/2209.05433
'''
    
    with open(output_path, 'w') as f:
        f.write(mlir_content)
    
    return output_path


def create_fp8_guide(format_type, output_path):
    """Create a comprehensive text guide for FP8 quantization."""
    
    format_info = {
        'e4m3fn': {
            'name': 'FP8 E4M3FN',
            'hardware': 'NVIDIA Hopper (H100, H200)',
            'precision': '4 exponent bits, 3 mantissa bits',
            'use_case': 'Activations and gradients'
        },
        'e4m3fnuz': {
            'name': 'FP8 E4M3FNUZ',
            'hardware': 'AMD MI300 series',
            'precision': '4 exponent bits, 3 mantissa bits',
            'use_case': 'Activations and gradients'
        },
        'e5m2': {
            'name': 'FP8 E5M2',
            'hardware': 'NVIDIA Hopper (H100, H200)',
            'precision': '5 exponent bits, 2 mantissa bits',
            'use_case': 'Weights (wider range)'
        },
        'e5m2fnuz': {
            'name': 'FP8 E5M2FNUZ',
            'hardware': 'AMD MI300 series',
            'precision': '5 exponent bits, 2 mantissa bits',
            'use_case': 'Weights (wider range)'
        }
    }
    
    info = format_info.get(format_type, format_info['e4m3fn'])
    
    guide_content = f'''{info['name']} Quantization Guide for IREE
{'='*70}

OVERVIEW
--------
{info['name']} is an 8-bit floating point format optimized for modern GPU
architectures with hardware acceleration for FP8 tensor operations.

Format: {info['precision']}
Hardware: {info['hardware']}
Best for: {info['use_case']}

WHY FP8?
--------
✓ 4x memory reduction vs FP32
✓ 2-4x faster inference on supported GPUs
✓ Better accuracy than INT8 for many models
✓ No zero-point offset needed (unlike INT quantization)
✓ Native hardware acceleration on modern GPUs

FP8 FORMAT COMPARISON
---------------------

E4M3 (4 exponent, 3 mantissa):
• Range: ±240
• Precision: Higher (3 mantissa bits)
• Best for: Activations, gradients
• More precise representation of small values

E5M2 (5 exponent, 2 mantissa):
• Range: ±57344
• Precision: Lower (2 mantissa bits)
• Best for: Weights (need wider range)
• Can represent larger absolute values

HARDWARE VARIANTS
-----------------

NVIDIA (FN - Finite, No NaN):
• f8E4M3FN, f8E5M2
• Used on Hopper architecture (H100, H200)
• Standard IEEE-like representation

AMD (FNUZ - Finite, No NaN, Unsigned Zero):
• f8E4M3FNUZ, f8E5M2FNUZ
• Used on MI300 series (gfx942, gfx950)
• Different NaN/Inf encoding

WHEN TO USE {info['name']}
{'='*70}

Ideal Scenarios:
✓ Large models (transformers, LLMs)
✓ GPU deployment on supported hardware
✓ When accuracy is important (better than INT8)
✓ Memory bandwidth is bottleneck
✓ Have access to calibration data

Not Recommended:
✗ CPUs (limited FP8 support)
✗ Older GPUs without FP8 tensor cores
✗ Real-time systems (INT8 more predictable)
✗ When model is already small

QUANTIZATION WORKFLOW
---------------------

Step 1: Assess Hardware Compatibility
   • Check GPU architecture
   • {info['hardware']} required for hardware acceleration
   • Older GPUs: will emulate (slower than INT8)

Step 2: Collect Calibration Data
   • Representative samples from training/validation set
   • ~100-1000 samples typically sufficient
   • Diversity matters more than quantity

Step 3: Compute Scaling Factors
   Option A - Per-Tensor (simpler):
     scale = max(abs(tensor)) / fp8_max
     
   Option B - Per-Channel (better accuracy):
     scale[i] = max(abs(tensor[:, i])) / fp8_max
     
   Option C - Percentile Clipping (robust):
     scale = percentile(abs(tensor), 99.9) / fp8_max

Step 4: Quantize Weights
   # Pseudocode
   quantized = clip(
       round(weight / scale),
       fp8_min,
       fp8_max
   ).to(fp8)

Step 5: Export Model
   • Include FP8 Cast operations
   • Store scaling factors as metadata
   • Export to ONNX or directly to MLIR

Step 6: Compile with IREE
   For NVIDIA GPU:
     iree-compile model.mlir \\
       --iree-hal-target-backends=cuda \\
       --iree-cuda-target=sm_90 \\
       -o model.vmfb
   
   For AMD GPU:
     iree-compile model.mlir \\
       --iree-hal-target-backends=rocm \\
       --iree-rocm-target-chip=gfx942 \\
       -o model.vmfb

Step 7: Validate Accuracy
   • Compare outputs vs FP32 baseline
   • Check on diverse test cases
   • Monitor for numerical instability

IMPLEMENTATION EXAMPLE (PyTorch)
---------------------------------

import torch
import torch.nn as nn

# Assume model is defined and loaded
model = YourModel()
model.eval()

# Step 1: Collect calibration data
calibration_data = []
for batch in calibration_loader:
    calibration_data.append(batch)

# Step 2: Compute scales (per-tensor example)
def compute_scale(tensor):
    return torch.max(torch.abs(tensor)) / 240.0  # E4M3 max

scales = {{}}
for name, param in model.named_parameters():
    scales[name] = compute_scale(param.data)

# Step 3: Quantize (in practice, use framework tools)
def quantize_to_fp8(tensor, scale):
    quantized = torch.clamp(
        torch.round(tensor / scale),
        -240, 240  # E4M3 range
    )
    return quantized

# Step 4: Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model_fp8.onnx",
    opset_version=13
)

PERFORMANCE EXPECTATIONS
-------------------------

Model Size:
  FP32:  100%
  FP16:  50%
  {info['name']}: 25%

Inference Speed ({info['hardware']}):
  FP32:  1.0x
  FP16:  ~2x
  {info['name']}: ~4-6x (memory-bound models)

Accuracy (typical):
  vs FP32: <0.5% loss
  vs FP16: <0.2% loss
  Better than INT8 for most models

Memory Bandwidth:
  4x reduction vs FP32
  2x reduction vs FP16
  Critical for large models

MIXED PRECISION STRATEGIES
---------------------------

Strategy 1: FP8 Weights, FP16/FP32 Activations
• Reduces model size
• Fast weight loading
• Maintains activation precision

Strategy 2: E5M2 Weights, E4M3 Activations
• Optimized for each data type
• Best balance of range and precision
• Recommended for production

Strategy 3: Selective FP8
• Keep sensitive layers in FP16
• Use FP8 for larger layers
• Profile-guided optimization

TROUBLESHOOTING
---------------

Problem: No speedup observed
Solution:
  • Verify GPU supports FP8 ({info['hardware']})
  • Check if bottleneck is compute (not memory)
  • Profile to confirm tensor core usage
  • May need INT8 on unsupported hardware

Problem: Accuracy degradation
Solution:
  • Use per-channel quantization
  • Increase calibration data diversity
  • Try mixed precision (FP8/FP16)
  • Consider FP8-aware training

Problem: Numerical instability
Solution:
  • Check for overflow (use E5M2 for wider range)
  • Adjust scaling factors
  • Add gradient clipping during inference
  • Use higher precision for critical operations

Problem: IREE compilation fails
Solution:
  • Verify ONNX opset compatibility
  • Check operator support in IREE
  • Ensure proper FP8 Cast operations
  • May need custom import logic

BEST PRACTICES
--------------

1. Format Selection:
   • E4M3 for activations (need precision)
   • E5M2 for weights (need range)
   • Test both for your specific model

2. Scaling Strategy:
   • Per-channel > Per-tensor accuracy
   • Percentile clipping for outliers
   • Validate scales with histograms

3. Calibration:
   • Use representative data
   • Include edge cases
   • More diversity > more samples

4. Validation:
   • Compare against FP32 baseline
   • Test on diverse inputs
   • Monitor worst-case accuracy

5. Hardware Targeting:
   • Match format to hardware (FN vs FNUZ)
   • Verify tensor core usage
   • Profile actual performance

ADVANCED: FP8 TRAINING
-----------------------

For best accuracy, train with FP8-aware training:

1. Simulate quantization during forward pass
2. Use higher precision for gradients
3. Delayed scaling: adjust scales periodically
4. Stochastic rounding for better convergence

Frameworks with FP8 training support:
• PyTorch (torch.float8 types)
• JAX (with custom dtypes)
• NVIDIA Transformer Engine
• Microsoft DeepSpeed

FP8 vs INT8 COMPARISON
----------------------

{info['name']}:
  ✓ Better accuracy (floating point)
  ✓ No zero-point offset needed
  ✓ Natural for neural networks
  ✓ Hardware accelerated (modern GPUs)
  ✗ Limited hardware support
  ✗ Newer, less tooling

INT8:
  ✓ Wide hardware support
  ✓ Mature tooling
  ✓ Good CPU performance
  ✗ Needs zero points
  ✗ Quantization can be tricky
  ✗ Lower accuracy for some models

RESOURCES
---------

• FP8 Formats Paper: https://arxiv.org/abs/2209.05433
• IREE ROCM FP8: compiler/plugins/target/ROCM/builtins/
• IREE GPU Dialect: compiler/src/iree/compiler/Codegen/Dialect/GPU/
• NVIDIA FP8: https://developer.nvidia.com/blog/fp8-training/

NEXT STEPS
----------

1. Verify hardware compatibility ({info['hardware']})
2. Review MLIR examples (created by this script)
3. Collect calibration data for your model
4. Quantize and validate accuracy
5. Benchmark inference performance
6. Iterate on scaling strategy if needed

For questions and support:
• IREE GitHub: https://github.com/iree-org/iree
• Discussions: https://github.com/iree-org/iree/discussions
• GPU-specific forums for hardware questions
'''
    
    with open(output_path, 'w') as f:
        f.write(guide_content)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="FP8 quantization reference for IREE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--format",
        choices=['e4m3fn', 'e4m3fnuz', 'e5m2', 'e5m2fnuz'],
        default='e4m3fn',
        help="FP8 format type (default: e4m3fn)"
    )
    parser.add_argument(
        "--output",
        help="Output path for guide (default: fp8_<format>_guide.txt)"
    )
    
    args = parser.parse_args()
    
    if not args.output:
        args.output = f"fp8_{args.format}_guide.txt"
    
    print("="*70)
    print(f"FP8 {args.format.upper()} Quantization Reference for IREE")
    print("="*70)
    
    # Create MLIR examples
    mlir_path = args.output.replace('.txt', '_examples.mlir')
    print(f"\nCreating MLIR examples: {mlir_path}")
    create_fp8_mlir_example(args.format, mlir_path)
    print(f"✓ MLIR examples created")
    
    # Create text guide
    print(f"\nCreating guide: {args.output}")
    create_fp8_guide(args.format, args.output)
    print(f"✓ Guide created")
    
    print("\n" + "="*70)
    print(f"FP8 {args.format.upper()} Documentation Created")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  • {args.output} - Comprehensive guide")
    print(f"  • {mlir_path} - MLIR code examples")
    
    format_desc = {
        'e4m3fn': 'NVIDIA Hopper - activations/gradients',
        'e4m3fnuz': 'AMD MI300 - activations/gradients',
        'e5m2': 'NVIDIA Hopper - weights (wider range)',
        'e5m2fnuz': 'AMD MI300 - weights (wider range)'
    }
    
    print(f"\nFormat: {args.format.upper()}")
    print(f"  {format_desc.get(args.format, '')}")
    print("\nKey Benefits:")
    print("  • 4x memory reduction vs FP32")
    print("  • Hardware accelerated on modern GPUs")
    print("  • Better accuracy than INT8")
    print("  • No zero-point quantization needed")
    
    print("\nFor actual FP8 quantization:")
    print("  1. Calibrate scales with representative data")
    print("  2. Quantize weights: q = clip(round(w/scale), min, max)")
    print("  3. Export to ONNX with Cast operations")
    print("  4. Compile: iree-compile model.mlir -o model.vmfb")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
