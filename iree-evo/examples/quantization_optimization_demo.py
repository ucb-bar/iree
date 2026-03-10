#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
IREE-Evo Use Case: Integer-Only Quantization Optimization

This demonstrates how IREE-Evo can be used to optimize quantized neural networks
by exploring different compilation strategies for integer-only requantization.

Problem Statement:
-----------------
Current quantization flows use inefficient float-based scaling:
  i32 accumulator -> f32 -> scale -> bias -> requant -> i8

Optimization Goal:
-----------------
Use integer-only requantization with fixed-point math:
  i32 accumulator -> fixed-point scale -> bias (i32) -> i8

Benefits:
--------
1. Eliminate float conversions (sitofp, mulf, divf)
2. Fuse dequant/requant into single integer operation
3. Better hardware utilization (integer units instead of FPUs)
4. Easier pattern matching for custom kernels
5. Potential 1.5-3x speedup for quantized inference

How IREE-Evo Helps:
------------------
IREE-Evo can automatically:
1. Parse the quantization pattern in MLIR
2. Select appropriate optimization strategies for quantized ops
3. Generate compilation flags that enable quantization fusion
4. Verify correctness of optimized variants
5. Benchmark and compare different approaches
6. Evolve the best configuration across generations
"""

import sys
from pathlib import Path

# Add iree_evo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from iree_evo.config import IREEEvoConfig
from iree_evo.orchestrator import Orchestrator
from iree_evo.agents.optimization_menu import OptimizationStrategy


def create_quantization_strategies():
    """Create custom optimization strategies for quantization."""
    
    strategies = {
        "quantization_fusion_basic": OptimizationStrategy(
            name="quantization_fusion_basic",
            description="Enable basic dequantization fusion with matmul",
            applicable_ops=["linalg.quantized_matmul"],
            applicable_backends=["*"],
            flags=["--iree-flow-enable-fuse-dequantization-matmul"],
            complexity="low",
            expected_speedup="1.3-1.8x",
        ),
        
        "quantization_int8_ukernels": OptimizationStrategy(
            name="quantization_int8_ukernels",
            description="Use optimized INT8 micro-kernels for quantized operations",
            applicable_ops=["linalg.quantized_matmul"],
            applicable_backends=["llvm-cpu"],
            flags=[
                "--iree-llvmcpu-enable-ukernels=all",
                "--iree-flow-enable-fuse-dequantization-matmul",
            ],
            complexity="medium",
            expected_speedup="1.8-2.5x",
        ),
        
        "quantization_vectorized": OptimizationStrategy(
            name="quantization_vectorized",
            description="Vectorize quantized operations for SIMD",
            applicable_ops=["linalg.quantized_matmul"],
            applicable_backends=["llvm-cpu"],
            flags=[
                "--iree-llvmcpu-enable-ukernels=all",
                "--iree-llvmcpu-target-cpu-features=+avx2,+fma",
                "--iree-flow-enable-fuse-dequantization-matmul",
            ],
            complexity="medium",
            expected_speedup="2.0-3.0x",
        ),
        
        "quantization_gpu_tensor_cores": OptimizationStrategy(
            name="quantization_gpu_tensor_cores",
            description="Use Tensor Cores for INT8 quantized matmul on GPUs",
            applicable_ops=["linalg.quantized_matmul"],
            applicable_backends=["cuda"],
            flags=[
                "--iree-codegen-llvmgpu-enable-transform-dialect-jit",
                "--iree-flow-enable-fuse-dequantization-matmul",
            ],
            complexity="high",
            expected_speedup="2.5-4.0x",
        ),
    }
    
    return strategies


def main():
    """Demonstrate IREE-Evo for quantization optimization."""
    
    print("=" * 80)
    print("IREE-Evo Use Case: Integer-Only Quantization Optimization")
    print("=" * 80)
    print()
    
    # Check if example files exist
    float_mlir = Path(__file__).parent / "quantized_matmul_float.mlir"
    integer_mlir = Path(__file__).parent / "quantized_matmul_integer.mlir"
    
    if not float_mlir.exists():
        print(f"❌ Example file not found: {float_mlir}")
        print("Please ensure you're running from the examples directory.")
        return 1
    
    print("📋 Problem Overview")
    print("-" * 80)
    print("Current quantization flow (INEFFICIENT):")
    print("  1. Quantized MatMul (i8 × i8 -> i32) ✓ Fast")
    print("  2. Convert i32 -> f32                ✗ Slow (sitofp)")
    print("  3. Scale in f32                      ✗ Slow (mulf)")
    print("  4. Add bias in f32                   ✗ Slow (addf)")
    print("  5. Requantize f32 -> i8              ✗ Slow (divf, fptosi)")
    print()
    print("Optimized flow (EFFICIENT):")
    print("  1. Quantized MatMul (i8 × i8 -> i32) ✓ Fast")
    print("  2. Add bias in i32                   ✓ Fast (addi)")
    print("  3. Fixed-point scale (i32 -> i8)     ✓ Fast (muli, shrsi)")
    print("  4. Clamp to i8 range                 ✓ Fast (maxsi, minsi)")
    print()
    
    print("🎯 IREE-Evo Optimization Strategy")
    print("-" * 80)
    print("IREE-Evo will:")
    print("  1. Parse the quantized MLIR to identify linalg.quantized_matmul")
    print("  2. Select applicable quantization fusion strategies")
    print("  3. Generate variants with different optimization flags")
    print("  4. Compile each variant with iree-compile")
    print("  5. Verify correctness against baseline")
    print("  6. Benchmark and measure speedup")
    print("  7. Evolve the best configuration across generations")
    print()
    
    # Display custom strategies
    print("📚 Custom Quantization Strategies")
    print("-" * 80)
    custom_strategies = create_quantization_strategies()
    for name, strategy in custom_strategies.items():
        print(f"\n{name}:")
        print(f"  Description: {strategy.description}")
        print(f"  Expected speedup: {strategy.expected_speedup}")
        print(f"  Flags:")
        for flag in strategy.flags:
            print(f"    - {flag}")
    print()
    
    # Configuration
    print("⚙️  IREE-Evo Configuration")
    print("-" * 80)
    
    config = IREEEvoConfig(
        # Target configuration
        target_backend="llvm-cpu",  # or "cuda" for GPU
        target_device="local-task",
        
        # Evolutionary parameters (small for demonstration)
        population_size=6,
        num_generations=3,
        selection_top_k=2,
        
        # Objectives
        optimize_latency=True,
        optimize_size=False,
        latency_weight=1.0,
        
        # Working directory
        work_dir=Path("/tmp/iree_evo_quantization"),
        
        # Verbosity
        verbose=True,
    )
    
    print(f"Target: {config.target_backend} / {config.target_device}")
    print(f"Generations: {config.num_generations}")
    print(f"Population per generation: {config.population_size}")
    print(f"Optimization objective: {'Latency' if config.optimize_latency else 'Size'}")
    print()
    
    # Create orchestrator
    orchestrator = Orchestrator(config)
    
    # Add custom strategies to the optimization menu
    print("📝 Adding custom quantization strategies...")
    for name, strategy in custom_strategies.items():
        orchestrator.optimization_menu.strategies[name] = strategy
    print(f"✓ Added {len(custom_strategies)} custom strategies")
    print()
    
    print("🚀 Starting Optimization")
    print("=" * 80)
    print()
    print("NOTE: This will attempt to compile and benchmark the MLIR.")
    print("If you don't have iree-compile and iree-benchmark-module installed,")
    print("the demonstration will show the workflow but skip actual compilation.")
    print()
    
    try:
        # Optimize the float-based version
        print("Optimizing float-based quantization pattern...")
        print(f"Input: {float_mlir}")
        print()
        
        best_variant = orchestrator.optimize(float_mlir)
        
        if best_variant:
            print()
            print("=" * 80)
            print("✅ OPTIMIZATION COMPLETE")
            print("=" * 80)
            print()
            print(f"Best Variant: {best_variant.variant_id}")
            print(f"Generation: {best_variant.generation}")
            print(f"Mean Latency: {best_variant.mean_latency_ms:.3f} ms")
            print(f"Binary Size: {best_variant.binary_size_bytes / (1024*1024):.2f} MB")
            print(f"Fitness Score: {best_variant.fitness_score:.3f}")
            print()
            print("Compilation Flags Used:")
            for flag in best_variant.compilation_flags:
                print(f"  {flag}")
            print()
            
            if best_variant.vmfb_path:
                print(f"Optimized Binary: {best_variant.vmfb_path}")
            print()
            
            print("💡 Key Insights:")
            print("-" * 80)
            print("1. The optimized variant should show improved latency")
            print("2. Look for flags like --iree-flow-enable-fuse-dequantization-matmul")
            print("3. Integer-only operations avoid FPU bottlenecks")
            print("4. Consider the integer-only MLIR pattern for even better results")
            print()
            print("📝 Next Steps:")
            print("-" * 80)
            print("1. Compare with the integer-only version (quantized_matmul_integer.mlir)")
            print("2. Create a compiler pass to transform float patterns to integer patterns")
            print("3. Benchmark on target hardware (CPU with AVX-512, GPU with INT8 Tensor Cores)")
            print("4. Integrate the best configuration into your model compilation pipeline")
            
            return 0
        else:
            print()
            print("⚠️  No successful optimization found")
            print()
            print("This is expected if IREE tools are not installed.")
            print("The framework demonstrated the workflow:")
            print("  ✓ MLIR parsing and analysis")
            print("  ✓ Strategy selection")
            print("  ✓ Flag generation")
            print("  ✓ Evolutionary loop structure")
            print()
            print("To run with actual compilation:")
            print("  pip install iree-compiler iree-runtime")
            return 1
    
    except Exception as e:
        print()
        print(f"❌ Error: {e}")
        print()
        print("Note: This demonstration requires IREE compiler tools.")
        print("Install with: pip install iree-compiler iree-runtime")
        print()
        print("However, this example shows how IREE-Evo can be used for")
        print("quantization optimization in production environments!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
