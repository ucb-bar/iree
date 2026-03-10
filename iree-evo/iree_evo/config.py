#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Configuration and constants for IREE-Evo."""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class IREEEvoConfig:
    """Main configuration for IREE-Evo system."""
    
    # Paths
    iree_compile_path: str = "iree-compile"
    iree_benchmark_path: str = "iree-benchmark-module"
    iree_run_module_path: str = "iree-run-module"
    work_dir: Path = field(default_factory=lambda: Path("/tmp/iree_evo_work"))
    
    # Target configuration
    target_backend: str = "llvm-cpu"  # llvm-cpu, cuda, rocm, vulkan, metal
    target_device: str = "local-task"  # local-task, local-sync, cuda, rocm, vulkan
    
    # Evolutionary algorithm parameters
    population_size: int = 10
    num_generations: int = 5
    mutation_rate: float = 0.3
    selection_top_k: int = 3
    
    # Optimization objectives
    optimize_latency: bool = True
    optimize_size: bool = False
    latency_weight: float = 1.0
    size_weight: float = 0.0
    
    # Verification settings
    correctness_rtol: float = 1e-5
    correctness_atol: float = 1e-8
    num_test_runs: int = 3
    
    # LLM API settings (placeholders for future integration)
    planner_model: str = "gemini-2.0-flash"  # or gpt-4o
    coder_model: str = "gemini-2.0-flash"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000
    
    # Timeout settings (in seconds)
    compile_timeout: int = 300
    benchmark_timeout: int = 60
    test_timeout: int = 60
    
    # Logging
    verbose: bool = True
    log_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Initialize work and log directories."""
        self.work_dir = Path(self.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        if self.log_dir:
            self.log_dir = Path(self.log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)


# Predefined optimization strategies
OPTIMIZATION_STRATEGIES = {
    "vectorize": {
        "description": "Enable vectorization for SIMD operations",
        "applicable_ops": ["linalg.matmul", "linalg.conv", "linalg.generic"],
        "flags": ["--iree-llvmcpu-enable-ukernels=all"],
    },
    "tile": {
        "description": "Apply tiling transformation for better cache locality",
        "applicable_ops": ["linalg.matmul", "linalg.conv", "linalg.batch_matmul"],
        "transform_required": True,
    },
    "fuse": {
        "description": "Fuse operations to reduce memory traffic",
        "applicable_ops": ["linalg.matmul", "linalg.generic", "linalg.fill"],
        "flags": ["--iree-flow-enable-aggressive-fusion"],
    },
    "pad": {
        "description": "Pad tensors to improve alignment",
        "applicable_ops": ["linalg.matmul", "linalg.conv"],
    },
    "pipeline": {
        "description": "Enable software pipelining",
        "applicable_ops": ["linalg.matmul"],
        "flags": ["--iree-codegen-llvmgpu-use-transform-dialect=true"],
    },
    "tensorcore": {
        "description": "Use Tensor Cores on NVIDIA GPUs",
        "applicable_ops": ["linalg.matmul", "linalg.batch_matmul"],
        "backends": ["cuda"],
        "flags": [
            "--iree-codegen-llvmgpu-enable-transform-dialect-jit",
            "--iree-codegen-gpu-native-math-precision=true",
        ],
    },
    "quantize": {
        "description": "Apply quantization optimizations",
        "applicable_ops": ["linalg.matmul"],
        "flags": ["--iree-flow-enable-fuse-dequantization-matmul"],
    },
}


# Hardware profiles
HARDWARE_PROFILES = {
    "nvidia_a100": {
        "vendor": "nvidia",
        "arch": "ampere",
        "tensor_cores": True,
        "compute_capability": "8.0",
        "l1_cache_kb": 192,
        "shared_mem_kb": 164,
        "preferred_tile_sizes": [128, 128, 16],
    },
    "nvidia_h100": {
        "vendor": "nvidia",
        "arch": "hopper",
        "tensor_cores": True,
        "fp8_support": True,
        "compute_capability": "9.0",
        "l1_cache_kb": 256,
        "shared_mem_kb": 228,
        "preferred_tile_sizes": [128, 128, 16],
    },
    "amd_mi300": {
        "vendor": "amd",
        "arch": "cdna3",
        "matrix_cores": True,
        "fp8_support": True,
        "fp8_variants": ["E4M3FNUZ", "E5M2FNUZ"],
        "lds_kb": 64,
        "preferred_tile_sizes": [128, 128, 32],
    },
    "cpu_avx512": {
        "vendor": "intel",
        "arch": "x86_64",
        "simd": "avx512",
        "l1_cache_kb": 32,
        "l2_cache_kb": 1024,
        "l3_cache_kb": 36864,
        "preferred_tile_sizes": [64, 64, 8],
    },
}


# MLIR operation categories
MLIR_OP_CATEGORIES = {
    "compute_intensive": [
        "linalg.matmul",
        "linalg.batch_matmul",
        "linalg.conv_2d_nhwc_hwcf",
        "linalg.conv_2d_nchw_fchw",
        "linalg.pooling_nhwc_sum",
        "linalg.pooling_nhwc_max",
    ],
    "elementwise": [
        "linalg.generic",
        "linalg.map",
        "arith.addf",
        "arith.mulf",
        "math.exp",
        "math.tanh",
    ],
    "memory": [
        "linalg.fill",
        "linalg.copy",
        "tensor.extract",
        "tensor.insert",
    ],
    "control_flow": [
        "scf.for",
        "scf.while",
        "scf.if",
    ],
}
