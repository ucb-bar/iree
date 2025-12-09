#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Optimization Menu: Predefined optimization strategies for IREE."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class OptimizationStrategy:
    """Represents an optimization strategy."""
    name: str
    description: str
    applicable_ops: List[str]
    applicable_backends: List[str]
    flags: List[str]
    transform_dialect_required: bool = False
    complexity: str = "medium"  # low, medium, high
    expected_speedup: str = "1.2-2.0x"


class OptimizationMenu:
    """Catalog of predefined optimization strategies."""
    
    def __init__(self):
        self.strategies = self._initialize_strategies()
    
    def _initialize_strategies(self) -> Dict[str, OptimizationStrategy]:
        """Initialize the menu of optimization strategies."""
        return {
            "baseline": OptimizationStrategy(
                name="baseline",
                description="No optimizations, baseline compilation",
                applicable_ops=["*"],
                applicable_backends=["*"],
                flags=[],
                complexity="low",
                expected_speedup="1.0x (baseline)",
            ),
            
            "enable_ukernels": OptimizationStrategy(
                name="enable_ukernels",
                description="Enable micro-kernels for optimized CPU operations",
                applicable_ops=["linalg.matmul", "linalg.conv"],
                applicable_backends=["llvm-cpu"],
                flags=["--iree-llvmcpu-enable-ukernels=all"],
                complexity="low",
                expected_speedup="1.5-3.0x",
            ),
            
            "aggressive_fusion": OptimizationStrategy(
                name="aggressive_fusion",
                description="Enable aggressive operation fusion",
                applicable_ops=["linalg.matmul", "linalg.generic", "linalg.fill"],
                applicable_backends=["*"],
                flags=["--iree-flow-enable-aggressive-fusion"],
                complexity="medium",
                expected_speedup="1.2-1.8x",
            ),
            
            "fuse_dequantization": OptimizationStrategy(
                name="fuse_dequantization",
                description="Fuse dequantization with matmul for quantized models",
                applicable_ops=["linalg.matmul"],
                applicable_backends=["*"],
                flags=["--iree-flow-enable-fuse-dequantization-matmul"],
                complexity="medium",
                expected_speedup="1.3-2.0x",
            ),
            
            "tensor_core": OptimizationStrategy(
                name="tensor_core",
                description="Optimize for NVIDIA Tensor Cores",
                applicable_ops=["linalg.matmul", "linalg.batch_matmul"],
                applicable_backends=["cuda"],
                flags=[
                    "--iree-codegen-llvmgpu-enable-transform-dialect-jit",
                    "--iree-codegen-gpu-native-math-precision=true",
                ],
                complexity="high",
                expected_speedup="2.0-4.0x",
            ),
            
            "gpu_vectorize": OptimizationStrategy(
                name="gpu_vectorize",
                description="Enable GPU vectorization optimizations",
                applicable_ops=["linalg.matmul", "linalg.conv"],
                applicable_backends=["cuda", "rocm", "vulkan"],
                flags=[
                    "--iree-codegen-gpu-native-math-precision=true",
                ],
                complexity="medium",
                expected_speedup="1.3-2.0x",
            ),
            
            "software_pipelining": OptimizationStrategy(
                name="software_pipelining",
                description="Enable software pipelining for GPU",
                applicable_ops=["linalg.matmul"],
                applicable_backends=["cuda", "rocm"],
                flags=[
                    "--iree-codegen-llvmgpu-use-transform-dialect=true",
                ],
                transform_dialect_required=True,
                complexity="high",
                expected_speedup="1.5-2.5x",
            ),
            
            "amd_wmma": OptimizationStrategy(
                name="amd_wmma",
                description="Use AMD Wave Matrix Multiply-Accumulate (WMMA)",
                applicable_ops=["linalg.matmul", "linalg.batch_matmul"],
                applicable_backends=["rocm"],
                flags=[
                    "--iree-codegen-llvmgpu-enable-transform-dialect-jit",
                ],
                complexity="high",
                expected_speedup="2.0-3.5x",
            ),
            
            "cpu_data_tiling": OptimizationStrategy(
                name="cpu_data_tiling",
                description="Apply data tiling for better CPU cache utilization",
                applicable_ops=["linalg.matmul", "linalg.conv"],
                applicable_backends=["llvm-cpu"],
                flags=[
                    "--iree-llvmcpu-target-cpu-features=+avx512f",
                ],
                transform_dialect_required=True,
                complexity="high",
                expected_speedup="1.5-2.5x",
            ),
            
            "pad_operations": OptimizationStrategy(
                name="pad_operations",
                description="Pad tensor operations for better alignment",
                applicable_ops=["linalg.matmul", "linalg.conv"],
                applicable_backends=["*"],
                flags=[
                    "--iree-flow-enable-padding",
                ],
                complexity="medium",
                expected_speedup="1.1-1.5x",
            ),
        }
    
    def get_strategy(self, name: str) -> Optional[OptimizationStrategy]:
        """Get a specific optimization strategy by name."""
        return self.strategies.get(name)
    
    def get_applicable_strategies(
        self,
        op_names: List[str],
        backend: str,
    ) -> List[OptimizationStrategy]:
        """Get strategies applicable to given operations and backend.
        
        Args:
            op_names: List of MLIR operation names
            backend: Target backend (llvm-cpu, cuda, rocm, etc.)
            
        Returns:
            List of applicable OptimizationStrategy objects
        """
        applicable = []
        
        for strategy in self.strategies.values():
            # Check if strategy applies to any of the operations
            if strategy.applicable_ops == ["*"]:
                ops_match = True
            else:
                ops_match = any(
                    op in strategy.applicable_ops
                    for op in op_names
                )
            
            # Check if strategy applies to the backend
            if strategy.applicable_backends == ["*"]:
                backend_match = True
            else:
                backend_match = backend in strategy.applicable_backends
            
            if ops_match and backend_match:
                applicable.append(strategy)
        
        return applicable
    
    def get_all_strategies(self) -> List[OptimizationStrategy]:
        """Get all available strategies."""
        return list(self.strategies.values())
    
    def strategy_to_flags(self, strategy: OptimizationStrategy) -> List[str]:
        """Convert a strategy to compilation flags."""
        return strategy.flags.copy()
    
    def combine_strategies(
        self,
        strategies: List[OptimizationStrategy],
    ) -> List[str]:
        """Combine multiple strategies into a single flag list.
        
        Note: This is a simple combination. In practice, some flags
        may conflict and require more sophisticated merging.
        """
        all_flags = []
        seen_flags = set()
        
        for strategy in strategies:
            for flag in strategy.flags:
                # Extract flag name (before '=')
                flag_name = flag.split('=')[0]
                if flag_name not in seen_flags:
                    all_flags.append(flag)
                    seen_flags.add(flag_name)
        
        return all_flags
