#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Baseline Manager: Establishes and manages baseline compilation and performance."""

import subprocess
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class BaselineResult:
    """Results from baseline compilation and benchmarking."""
    mlir_file: str
    vmfb_file: str
    compilation_flags: List[str]
    compilation_success: bool
    compilation_time: float
    binary_size_bytes: int
    benchmark_success: bool = False
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput: float = 0.0
    error_message: Optional[str] = None


class BaselineManager:
    """Manages baseline compilation and benchmarking."""
    
    def __init__(
        self,
        iree_compile_path: str = "iree-compile",
        iree_benchmark_path: str = "iree-benchmark-module",
        target_backend: str = "llvm-cpu",
        target_device: str = "local-task",
    ):
        self.iree_compile_path = iree_compile_path
        self.iree_benchmark_path = iree_benchmark_path
        self.target_backend = target_backend
        self.target_device = target_device
    
    def establish_baseline(
        self,
        mlir_file: Path,
        output_dir: Path,
        timeout: int = 300,
        benchmark: bool = True,
    ) -> BaselineResult:
        """Establish baseline by compiling and optionally benchmarking.
        
        Args:
            mlir_file: Path to input MLIR file
            output_dir: Directory for output files
            timeout: Compilation timeout in seconds
            benchmark: Whether to run benchmarking
            
        Returns:
            BaselineResult with compilation and benchmark data
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output path
        vmfb_file = output_dir / f"{mlir_file.stem}_baseline.vmfb"
        
        # Basic compilation flags
        flags = [
            f"--iree-hal-target-backends={self.target_backend}",
        ]
        
        # Compile
        import time
        start_time = time.time()
        success, error = self._compile(mlir_file, vmfb_file, flags, timeout)
        compile_time = time.time() - start_time
        
        # Get binary size
        binary_size = vmfb_file.stat().st_size if success else 0
        
        result = BaselineResult(
            mlir_file=str(mlir_file),
            vmfb_file=str(vmfb_file),
            compilation_flags=flags,
            compilation_success=success,
            compilation_time=compile_time,
            binary_size_bytes=binary_size,
            error_message=error,
        )
        
        # Benchmark if requested and compilation succeeded
        if benchmark and success:
            bench_result = self._benchmark(vmfb_file)
            if bench_result:
                result.benchmark_success = True
                result.mean_latency_ms = bench_result.get('mean_latency_ms', 0.0)
                result.p50_latency_ms = bench_result.get('p50_latency_ms', 0.0)
                result.p99_latency_ms = bench_result.get('p99_latency_ms', 0.0)
                result.throughput = bench_result.get('throughput', 0.0)
        
        return result
    
    def _compile(
        self,
        mlir_file: Path,
        output_file: Path,
        flags: List[str],
        timeout: int,
    ) -> tuple[bool, Optional[str]]:
        """Compile MLIR file to VMFB.
        
        Returns:
            Tuple of (success, error_message)
        """
        cmd = [
            self.iree_compile_path,
            str(mlir_file),
            "-o", str(output_file),
        ] + flags
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            if result.returncode == 0:
                return True, None
            else:
                return False, result.stderr
        
        except subprocess.TimeoutExpired:
            return False, f"Compilation timed out after {timeout}s"
        except FileNotFoundError:
            return False, f"iree-compile not found at: {self.iree_compile_path}"
        except Exception as e:
            return False, f"Compilation error: {str(e)}"
    
    def _benchmark(
        self,
        vmfb_file: Path,
        timeout: int = 60,
    ) -> Optional[Dict[str, float]]:
        """Benchmark a compiled module.
        
        Returns:
            Dictionary with benchmark metrics or None if failed
        """
        cmd = [
            self.iree_benchmark_path,
            f"--module={vmfb_file}",
            f"--device={self.target_device}",
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            if result.returncode == 0:
                return self._parse_benchmark_output(result.stdout)
            else:
                return None
        
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return None
    
    def _parse_benchmark_output(self, output: str) -> Dict[str, float]:
        """Parse benchmark output to extract metrics."""
        metrics = {}
        
        # Parse common benchmark output patterns
        import re
        
        # Look for timing information
        # Example: "BM_main/process_time/real_time_mean 1.23 ms"
        mean_pattern = r'mean[:\s]+([0-9.]+)\s*(ms|us|s)'
        match = re.search(mean_pattern, output, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            # Convert to ms
            if unit == 'us':
                value /= 1000.0
            elif unit == 's':
                value *= 1000.0
            metrics['mean_latency_ms'] = value
        
        # Look for p50, p99, etc.
        percentile_pattern = r'p(\d+)[:\s]+([0-9.]+)\s*(ms|us|s)'
        for match in re.finditer(percentile_pattern, output, re.IGNORECASE):
            percentile = match.group(1)
            value = float(match.group(2))
            unit = match.group(3)
            # Convert to ms
            if unit == 'us':
                value /= 1000.0
            elif unit == 's':
                value *= 1000.0
            metrics[f'p{percentile}_latency_ms'] = value
        
        return metrics
    
    def save_baseline(self, baseline: BaselineResult, output_file: Path):
        """Save baseline result to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(asdict(baseline), f, indent=2)
    
    def load_baseline(self, input_file: Path) -> BaselineResult:
        """Load baseline result from JSON file."""
        with open(input_file, 'r') as f:
            data = json.load(f)
        return BaselineResult(**data)
