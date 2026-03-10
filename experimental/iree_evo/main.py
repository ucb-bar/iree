#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Main entry point for IREE-EVO optimization system.

This module initializes and runs the evolutionary optimization loop
for IREE compiler configurations using OpenEvolve.

Usage:
    python -m iree_evo.main --mlir-path <path> --backend <backend>

Example:
    python -m iree_evo.main --mlir-path model.mlir --backend llvm-cpu
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, Optional

from .evaluator import IREEEvaluator, CompilationError
from .knowledge_base import KnowledgeBase
from .prompts import PLANNER_PROMPT, CODER_PROMPT, EVOLUTION_PROMPT
from .slicer import MLIRSlicer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def mock_planner_llm(mlir_summary: Dict, backend: str) -> Dict:
    """Mock planner LLM for testing without API access.

    This function simulates the planner LLM's decision-making process.
    In production, this would be replaced with actual API calls.

    Args:
        mlir_summary: Summary of the MLIR content.
        backend: Target backend.

    Returns:
        A dictionary containing the optimization strategy and constraints.
    """
    # Check for quantization-related operations
    quant_ops = mlir_summary.get("quantization_ops", [])
    compute_ops = mlir_summary.get("compute_ops", [])

    # Determine strategy based on IR content
    if any("sitofp" in op or "uitofp" in op for op in quant_ops):
        strategy = "IntegerRequantization"
        rationale = (
            "Detected float dequantization operations (sitofp/uitofp). "
            "These can be converted to pure integer arithmetic for better performance."
        )
    elif any("matmul" in op.lower() for op in compute_ops):
        strategy = "QuantizationFusion"
        rationale = (
            "Detected matmul operations with potential quantization. "
            "Fusing quantization with compute can reduce memory bandwidth."
        )
    elif any("conv" in op.lower() for op in compute_ops):
        strategy = "Tiling"
        rationale = (
            "Detected convolution operations. "
            "Tiling can improve cache utilization for large convolutions."
        )
    else:
        strategy = "Vectorization"
        rationale = (
            "No specific optimization pattern detected. "
            "Applying vectorization for general performance improvement."
        )

    return {
        "analysis": {
            "compute_bottleneck": "quantization overhead" if quant_ops else "compute",
            "memory_pattern": "sequential access",
            "quantization_status": "float dequant present" if quant_ops else "none",
        },
        "strategy": strategy,
        "rationale": rationale,
        "constraints": [
            "Preserve numerical precision",
            "Ensure output matches baseline within tolerance",
        ],
        "priority_ops": compute_ops[:5] if compute_ops else ["all"],
    }


def mock_coder_llm(strategy: str, backend: str, constraints: list) -> str:
    """Mock coder LLM for testing without API access.

    This function generates compiler flags based on the strategy.
    In production, this would be replaced with actual API calls.

    Args:
        strategy: The optimization strategy.
        backend: Target backend.
        constraints: List of constraints from the planner.

    Returns:
        A string containing compiler flags.
    """
    base_flags = [
        f"# Optimization: {strategy}",
        f"# Target: {backend}",
        "",
    ]

    # Get strategy-specific flags
    try:
        strategy_flags = KnowledgeBase.get_optimization_strategy_flags(strategy)
    except ValueError:
        strategy_flags = []

    # Add backend-specific flags
    if backend == "llvm-cpu":
        base_flags.extend([
            "--iree-opt-const-eval=true",
            "--iree-opt-const-expr-hoisting=true",
        ])
    elif backend == "cuda":
        base_flags.extend([
            "--iree-codegen-llvmgpu-enable-transform-dialect-jit-default=true",
        ])

    # Add strategy flags
    base_flags.extend(strategy_flags)

    return "\n".join(base_flags)


def run_optimization(
    mlir_path: str,
    backend: str,
    work_dir: Optional[str] = None,
    max_iterations: int = 10,
    use_openevolve: bool = True,
) -> Dict:
    """Runs the IREE-EVO optimization loop.

    Args:
        mlir_path: Path to the input MLIR file.
        backend: Target backend.
        work_dir: Working directory for intermediate files.
        max_iterations: Maximum number of evolution iterations.
        use_openevolve: Whether to use OpenEvolve (if available).

    Returns:
        A dictionary containing the optimization results.
    """
    logger.info(f"Starting IREE-EVO optimization for {mlir_path}")
    logger.info(f"Target backend: {backend}")

    # Initialize evaluator
    evaluator = IREEEvaluator(
        baseline_mlir_path=mlir_path,
        target_backend=backend,
        work_dir=work_dir,
        strategy="IntegerRequantization",  # Will be updated by planner
    )

    # Step 1: Analyze MLIR and run planner
    logger.info("Step 1: Analyzing MLIR and running planner...")
    mlir_summary = evaluator.get_baseline_summary()
    logger.info(f"MLIR Summary: {json.dumps(mlir_summary, indent=2)}")

    planner_result = mock_planner_llm(mlir_summary, backend)
    logger.info(f"Planner decision: {planner_result['strategy']}")
    logger.info(f"Rationale: {planner_result['rationale']}")

    # Update evaluator strategy
    evaluator.strategy = planner_result["strategy"]

    # Step 2: Generate initial configuration
    logger.info("Step 2: Generating initial configuration...")
    initial_config = mock_coder_llm(
        planner_result["strategy"],
        backend,
        planner_result["constraints"],
    )
    logger.info(f"Initial configuration:\n{initial_config}")

    # Step 3: Run OpenEvolve loop (or mock loop)
    best_config = initial_config
    best_score = float("-inf")

    if use_openevolve:
        try:
            # Try to use OpenEvolve if available
            from openevolve import Controller

            logger.info("Step 3: Starting OpenEvolve optimization loop...")

            controller = Controller(
                evaluator=evaluator,
                initial_population=[initial_config],
                system_prompt=EVOLUTION_PROMPT,
                max_iterations=max_iterations,
            )

            best_config, best_score = controller.run()
            logger.info(f"OpenEvolve completed. Best score: {best_score}")

        except ImportError:
            logger.warning(
                "OpenEvolve not installed. Running mock optimization loop."
            )
            use_openevolve = False

    if not use_openevolve:
        # Mock optimization loop
        logger.info("Step 3: Running mock optimization loop...")

        for iteration in range(max_iterations):
            logger.info(f"Iteration {iteration + 1}/{max_iterations}")

            try:
                score = evaluator.evaluate(best_config)
                logger.info(f"Score: {score}")

                if score > best_score:
                    best_score = score
                    logger.info(f"New best score: {best_score}")

                # Simple mutation: toggle a flag
                mutated_config = mutate_config(best_config, iteration)
                try:
                    mutated_score = evaluator.evaluate(mutated_config)
                    if mutated_score > best_score:
                        best_config = mutated_config
                        best_score = mutated_score
                        logger.info(f"Mutation improved score to: {best_score}")
                except CompilationError as e:
                    logger.warning(f"Mutation failed to compile: {e.message}")

            except CompilationError as e:
                logger.error(f"Compilation error: {e.message}")
                if e.debug_info:
                    logger.debug(f"Debug info: {e.debug_info[:500]}...")

    # Cleanup
    evaluator.cleanup()

    return {
        "strategy": planner_result["strategy"],
        "best_config": best_config,
        "best_score": best_score,
        "iterations": max_iterations,
    }


def mutate_config(config: str, iteration: int) -> str:
    """Simple mutation function for testing.

    Args:
        config: Current configuration.
        iteration: Current iteration number.

    Returns:
        Mutated configuration.
    """
    lines = config.split("\n")

    # Add a comment about the mutation
    lines.insert(0, f"# Mutation iteration {iteration + 1}")

    # Simple mutation: add or remove a flag based on iteration
    mutations = [
        "--iree-opt-data-tiling=true",
        "--iree-llvmcpu-enable-pad-consumer-fusion=true",
        "--iree-global-opt-enable-fuse-horizontal-contractions=true",
    ]

    mutation_flag = mutations[iteration % len(mutations)]

    if mutation_flag in config:
        # Remove the flag
        lines = [l for l in lines if mutation_flag not in l]
    else:
        # Add the flag
        lines.append(mutation_flag)

    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="IREE-EVO: Evolutionary Optimization for IREE Compiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize a model for CPU
  python -m iree_evo.main --mlir-path model.mlir --backend llvm-cpu

  # Run with more iterations
  python -m iree_evo.main --mlir-path model.mlir --backend cuda --max-iterations 50

  # Specify working directory
  python -m iree_evo.main --mlir-path model.mlir --backend llvm-cpu --work-dir /tmp/iree_evo
        """,
    )

    parser.add_argument(
        "--mlir-path",
        type=str,
        required=True,
        help="Path to the input MLIR file",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="llvm-cpu",
        choices=["llvm-cpu", "cuda", "rocm"],
        help="Target backend (default: llvm-cpu)",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Working directory for intermediate files",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum number of evolution iterations (default: 10)",
    )
    parser.add_argument(
        "--no-openevolve",
        action="store_true",
        help="Disable OpenEvolve and use mock optimization loop",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file for results (JSON format)",
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input file
    if not os.path.exists(args.mlir_path):
        logger.error(f"MLIR file not found: {args.mlir_path}")
        sys.exit(1)

    # Run optimization
    try:
        results = run_optimization(
            mlir_path=args.mlir_path,
            backend=args.backend,
            work_dir=args.work_dir,
            max_iterations=args.max_iterations,
            use_openevolve=not args.no_openevolve,
        )

        # Print results
        print("\n" + "=" * 60)
        print("IREE-EVO Optimization Results")
        print("=" * 60)
        print(f"Strategy: {results['strategy']}")
        print(f"Best Score: {results['best_score']}")
        print(f"Iterations: {results['iterations']}")
        print("\nBest Configuration:")
        print("-" * 40)
        print(results["best_config"])
        print("-" * 40)

        # Save results if output file specified
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
