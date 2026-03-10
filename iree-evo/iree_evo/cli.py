#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Command-line interface for IREE-Evo."""

import argparse
import sys
from pathlib import Path

from iree_evo.config import IREEEvoConfig
from iree_evo.orchestrator import Orchestrator


def main():
    """Main entry point for IREE-Evo CLI."""
    parser = argparse.ArgumentParser(
        description="IREE-Evo: Autonomous Agentic Compiler Optimization Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize a matmul for CPU
  iree-evo --input matmul.mlir --backend llvm-cpu --device local-task
  
  # Optimize for NVIDIA GPU with 10 generations
  iree-evo --input model.mlir --backend cuda --device cuda --generations 10
  
  # Quick optimization with small population
  iree-evo --input conv.mlir --backend llvm-cpu --population 5 --generations 3
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input MLIR file to optimize",
    )
    
    # Target configuration
    parser.add_argument(
        "--backend",
        type=str,
        default="llvm-cpu",
        choices=["llvm-cpu", "cuda", "rocm", "vulkan", "metal"],
        help="Target backend (default: llvm-cpu)",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="local-task",
        help="Target device (default: local-task)",
    )
    
    # Evolutionary parameters
    parser.add_argument(
        "--generations",
        type=int,
        default=5,
        help="Number of generations (default: 5)",
    )
    
    parser.add_argument(
        "--population",
        type=int,
        default=10,
        help="Population size per generation (default: 10)",
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top variants to keep for next generation (default: 3)",
    )
    
    # Paths
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/iree_evo_work"),
        help="Working directory for outputs (default: /tmp/iree_evo_work)",
    )
    
    parser.add_argument(
        "--iree-compile",
        type=str,
        default="iree-compile",
        help="Path to iree-compile (default: iree-compile)",
    )
    
    parser.add_argument(
        "--iree-benchmark",
        type=str,
        default="iree-benchmark-module",
        help="Path to iree-benchmark-module (default: iree-benchmark-module)",
    )
    
    # Optimization objectives
    parser.add_argument(
        "--optimize-latency",
        action="store_true",
        default=True,
        help="Optimize for latency (default: True)",
    )
    
    parser.add_argument(
        "--optimize-size",
        action="store_true",
        help="Also optimize for binary size",
    )
    
    parser.add_argument(
        "--latency-weight",
        type=float,
        default=1.0,
        help="Weight for latency in fitness (default: 1.0)",
    )
    
    parser.add_argument(
        "--size-weight",
        type=float,
        default=0.0,
        help="Weight for binary size in fitness (default: 0.0)",
    )
    
    # Timeouts
    parser.add_argument(
        "--compile-timeout",
        type=int,
        default=300,
        help="Compilation timeout in seconds (default: 300)",
    )
    
    parser.add_argument(
        "--benchmark-timeout",
        type=int,
        default=60,
        help="Benchmark timeout in seconds (default: 60)",
    )
    
    # Output control
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output (default: True)",
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1
    
    # Create configuration
    config = IREEEvoConfig(
        iree_compile_path=args.iree_compile,
        iree_benchmark_path=args.iree_benchmark,
        work_dir=args.work_dir,
        target_backend=args.backend,
        target_device=args.device,
        population_size=args.population,
        num_generations=args.generations,
        selection_top_k=args.top_k,
        optimize_latency=args.optimize_latency,
        optimize_size=args.optimize_size,
        latency_weight=args.latency_weight,
        size_weight=args.size_weight,
        compile_timeout=args.compile_timeout,
        benchmark_timeout=args.benchmark_timeout,
        verbose=args.verbose and not args.quiet,
    )
    
    # Run optimization
    try:
        orchestrator = Orchestrator(config)
        best_variant = orchestrator.optimize(args.input)
        
        if best_variant:
            print("\n" + "=" * 70)
            print("✓ Optimization Complete!")
            print("=" * 70)
            print(f"\nBest variant: {best_variant.variant_id}")
            print(f"Mean latency: {best_variant.mean_latency_ms:.3f} ms")
            print(f"Binary size: {best_variant.binary_size_bytes / (1024*1024):.2f} MB")
            print(f"VMFB location: {best_variant.vmfb_path}")
            print(f"\nFlags used:")
            for flag in best_variant.compilation_flags:
                print(f"  {flag}")
            
            return 0
        else:
            print("\n✗ No successful optimization found", file=sys.stderr)
            return 1
    
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user", file=sys.stderr)
        return 130
    
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
