# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Knowledge base for IREE compiler flags and MLIR Transform operations.

This module provides information about valid IREE compilation flags
and MLIR Transform Dialect operations that can be used for optimization.
"""

from typing import Dict, List


class KnowledgeBase:
    """Knowledge base for IREE compiler flags and Transform Dialect operations.

    This class provides methods to retrieve valid compiler flags for different
    backends and available Transform Dialect operations for code generation
    optimization.
    """

    # Valid IREE flags for llvm-cpu backend relevant to quantization and tiling
    _LLVM_CPU_FLAGS: List[str] = [
        # Quantization-related flags
        "--iree-opt-data-tiling",
        "--iree-opt-const-expr-hoisting",
        "--iree-opt-const-eval",
        "--iree-opt-numeric-precision-reduction",
        # Tiling and vectorization flags
        "--iree-llvmcpu-target-triple",
        "--iree-llvmcpu-target-cpu",
        "--iree-llvmcpu-target-cpu-features",
        "--iree-llvmcpu-enable-ukernels",
        # Code generation flags
        "--iree-codegen-llvm-generic-ops-workgroup-size",
        "--iree-llvmcpu-enable-pad-consumer-fusion",
        # Transform dialect flags
        "--iree-codegen-transform-dialect-library",
        # Debug flags (useful for error analysis)
        "--mlir-print-ir-after-all",
        "--mlir-print-ir-before-all",
        "--mlir-print-op-on-diagnostic",
        # Global optimization flags
        "--iree-global-opt-enable-fuse-horizontal-contractions",
        "--iree-global-opt-enable-quantized-matmul-reassociation",
        "--iree-opt-demote-f32-to-f16",
        "--iree-opt-demote-f64-to-f32",
    ]

    # Valid IREE flags for CUDA backend
    _CUDA_FLAGS: List[str] = [
        # Target configuration
        "--iree-hal-target-backends=cuda",
        "--iree-hal-cuda-llvm-target-arch",
        # Quantization flags
        "--iree-opt-data-tiling",
        "--iree-opt-const-expr-hoisting",
        "--iree-opt-const-eval",
        # Code generation flags
        "--iree-codegen-llvmgpu-enable-transform-dialect-jit-default",
        "--iree-codegen-transform-dialect-library",
        # Debug flags
        "--mlir-print-ir-after-all",
        "--mlir-print-ir-before-all",
    ]

    # Valid IREE flags for ROCM/AMD GPU backend
    _ROCM_FLAGS: List[str] = [
        # Target configuration
        "--iree-hal-target-backends=rocm",
        "--iree-rocm-target-chip",
        # Quantization flags (FP8 support)
        "--iree-opt-data-tiling",
        "--iree-opt-const-expr-hoisting",
        # Code generation flags
        "--iree-codegen-llvmgpu-enable-transform-dialect-jit-default",
        # Debug flags
        "--mlir-print-ir-after-all",
        "--mlir-print-ir-before-all",
    ]

    # Valid MLIR Transform Dialect operations
    _TRANSFORM_OPS: List[str] = [
        # Structured transform ops for tiling
        "transform.structured.tile_using_forall",
        "transform.structured.tile_using_for",
        "transform.structured.tile_reduction_using_for",
        "transform.structured.tile_reduction_using_forall",
        # Fusion operations
        "transform.structured.fuse_into_containing_op",
        "transform.structured.fuse",
        # Vectorization operations
        "transform.structured.vectorize",
        "transform.structured.vectorize_children_and_apply_patterns",
        # Pattern application
        "transform.apply_patterns.canonicalization",
        "transform.apply_patterns.linalg.tiling_canonicalization",
        "transform.apply_patterns.iree.fold_fill_into_pad",
        "transform.apply_patterns.scf.for_loop_canonicalization",
        # CSE and cleanup
        "transform.apply_cse",
        "transform.structured.match",
        # IREE-specific transforms
        "transform.iree.populate_workgroup_count_region_using_num_threads_slice",
        "transform.iree.bufferize",
        "transform.iree.forall_to_workgroup",
        # Loop transforms
        "transform.loop.unroll",
        "transform.loop.peel",
        # Lowering transforms
        "transform.iree.apply_lowering_strategy",
    ]

    # Common flag values for different optimization strategies
    _OPTIMIZATION_STRATEGIES: Dict[str, List[str]] = {
        "IntegerRequantization": [
            "--iree-global-opt-enable-quantized-matmul-reassociation=true",
            "--iree-opt-const-eval=true",
            "--iree-opt-const-expr-hoisting=true",
        ],
        "Tiling": [
            "--iree-opt-data-tiling=true",
            "--iree-llvmcpu-enable-pad-consumer-fusion=true",
        ],
        "Vectorization": [
            "--iree-opt-data-tiling=true",
        ],
        "QuantizationFusion": [
            "--iree-global-opt-enable-fuse-horizontal-contractions=true",
            "--iree-global-opt-enable-quantized-matmul-reassociation=true",
        ],
    }

    @classmethod
    def get_valid_flags(cls, backend: str) -> List[str]:
        """Returns a list of valid IREE flags for a given backend.

        Args:
            backend: The target backend (e.g., 'llvm-cpu', 'cuda', 'rocm').

        Returns:
            A list of valid compiler flags for the specified backend.

        Raises:
            ValueError: If the backend is not recognized.
        """
        backend_lower = backend.lower()
        if backend_lower == "llvm-cpu":
            return cls._LLVM_CPU_FLAGS.copy()
        elif backend_lower == "cuda":
            return cls._CUDA_FLAGS.copy()
        elif backend_lower == "rocm":
            return cls._ROCM_FLAGS.copy()
        else:
            raise ValueError(
                f"Unknown backend: {backend}. "
                f"Supported backends: llvm-cpu, cuda, rocm"
            )

    @classmethod
    def get_transform_ops(cls) -> List[str]:
        """Returns a list of valid MLIR Transform Dialect operations.

        Returns:
            A list of available Transform Dialect operations.
        """
        return cls._TRANSFORM_OPS.copy()

    @classmethod
    def get_optimization_strategy_flags(cls, strategy: str) -> List[str]:
        """Returns recommended flags for a specific optimization strategy.

        Args:
            strategy: The optimization strategy name.

        Returns:
            A list of recommended flags for the strategy.

        Raises:
            ValueError: If the strategy is not recognized.
        """
        if strategy not in cls._OPTIMIZATION_STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Available strategies: {list(cls._OPTIMIZATION_STRATEGIES.keys())}"
            )
        return cls._OPTIMIZATION_STRATEGIES[strategy].copy()

    @classmethod
    def get_all_strategies(cls) -> List[str]:
        """Returns all available optimization strategy names.

        Returns:
            A list of available strategy names.
        """
        return list(cls._OPTIMIZATION_STRATEGIES.keys())
