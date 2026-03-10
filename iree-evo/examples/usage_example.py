#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Example usage of IREE-Evo programmatically."""

from pathlib import Path
from iree_evo.config import IREEEvoConfig
from iree_evo.orchestrator import Orchestrator


def main():
    """Example: Optimize a matmul MLIR file."""
    
    # Configure IREE-Evo
    config = IREEEvoConfig(
        # Target configuration
        target_backend="llvm-cpu",
        target_device="local-task",
        
        # Evolutionary parameters
        population_size=5,
        num_generations=3,
        selection_top_k=2,
        
        # Objectives
        optimize_latency=True,
        optimize_size=False,
        latency_weight=1.0,
        size_weight=0.0,
        
        # Working directory
        work_dir=Path("/tmp/iree_evo_example"),
        
        # Verbosity
        verbose=True,
    )
    
    # Create orchestrator
    orchestrator = Orchestrator(config)
    
    # Run optimization
    mlir_file = Path("examples/matmul.mlir")
    
    print("Starting IREE-Evo optimization...")
    print(f"Input file: {mlir_file}")
    print(f"Target: {config.target_backend} / {config.target_device}")
    print(f"Generations: {config.num_generations}")
    print(f"Population: {config.population_size}")
    print()
    
    try:
        best_variant = orchestrator.optimize(mlir_file)
        
        if best_variant:
            print("\n" + "=" * 70)
            print("✓ Optimization Complete!")
            print("=" * 70)
            print(f"\nBest variant: {best_variant.variant_id}")
            print(f"Mean latency: {best_variant.mean_latency_ms:.3f} ms")
            print(f"Binary size: {best_variant.binary_size_bytes / (1024*1024):.2f} MB")
            print(f"Fitness score: {best_variant.fitness_score:.3f}")
            print(f"\nCompilation flags used:")
            for flag in best_variant.compilation_flags:
                print(f"  {flag}")
            
            if best_variant.vmfb_path:
                print(f"\nOptimized binary: {best_variant.vmfb_path}")
        else:
            print("\n✗ No successful optimization found")
            print("This may be because:")
            print("  - IREE tools (iree-compile, iree-benchmark-module) are not available")
            print("  - The MLIR file has compilation errors")
            print("  - All variants failed verification")
    
    except Exception as e:
        print(f"\n✗ Error during optimization: {e}")
        print("\nNote: This example requires IREE compiler and runtime tools.")
        print("Install with: pip install iree-compiler iree-runtime")


if __name__ == "__main__":
    main()
