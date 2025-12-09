#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Benchmark Runner: Executes and parses iree-benchmark-module output."""

import subprocess
import time
import re
from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Result from benchmarking a module."""
    success: bool
    mean_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    std_dev_ms: float = 0.0
    throughput_items_per_sec: float = 0.0
    binary_size_bytes: int = 0
    error_message: Optional[str] = None
    raw_output: str = ""


class BenchmarkRunner:
    """Runs benchmarks on compiled IREE modules."""
    
    def __init__(
        self,
        iree_benchmark_path: str = "iree-benchmark-module",
        target_device: str = "local-task",
    ):
        self.iree_benchmark_path = iree_benchmark_path
        self.target_device = target_device
    
    def benchmark(
        self,
        vmfb_file: Path,
        timeout: int = 60,
        num_iterations: int = 100,
    ) -> BenchmarkResult:
        """Benchmark a compiled module.
        
        Args:
            vmfb_file: Path to compiled VMFB module
            timeout: Timeout in seconds
            num_iterations: Number of benchmark iterations
            
        Returns:
            BenchmarkResult with performance metrics
        """
        if not vmfb_file.exists():
            return BenchmarkResult(
                success=False,
                error_message=f"VMFB file not found: {vmfb_file}"
            )
        
        # Get binary size
        binary_size = vmfb_file.stat().st_size
        
        cmd = [
            self.iree_benchmark_path,
            f"--module={vmfb_file}",
            f"--device={self.target_device}",
            f"--benchmark_repetitions={num_iterations}",
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            if result.returncode == 0:
                metrics = self._parse_output(result.stdout)
                return BenchmarkResult(
                    success=True,
                    mean_latency_ms=metrics.get('mean_latency_ms', 0.0),
                    median_latency_ms=metrics.get('median_latency_ms', 0.0),
                    p99_latency_ms=metrics.get('p99_latency_ms', 0.0),
                    min_latency_ms=metrics.get('min_latency_ms', 0.0),
                    max_latency_ms=metrics.get('max_latency_ms', 0.0),
                    std_dev_ms=metrics.get('std_dev_ms', 0.0),
                    throughput_items_per_sec=metrics.get('throughput', 0.0),
                    binary_size_bytes=binary_size,
                    raw_output=result.stdout,
                )
            else:
                return BenchmarkResult(
                    success=False,
                    binary_size_bytes=binary_size,
                    error_message=result.stderr,
                    raw_output=result.stdout,
                )
        
        except subprocess.TimeoutExpired:
            return BenchmarkResult(
                success=False,
                binary_size_bytes=binary_size,
                error_message=f"Benchmark timed out after {timeout}s",
            )
        
        except FileNotFoundError:
            return BenchmarkResult(
                success=False,
                error_message=f"iree-benchmark-module not found at: {self.iree_benchmark_path}",
            )
        
        except Exception as e:
            return BenchmarkResult(
                success=False,
                binary_size_bytes=binary_size,
                error_message=f"Unexpected error: {str(e)}",
            )
    
    def _parse_output(self, output: str) -> Dict[str, float]:
        """Parse benchmark output to extract metrics."""
        metrics = {}
        
        # Pattern for various time units and their conversions to ms
        time_conversions = {
            'ns': 1e-6,
            'us': 1e-3,
            'ms': 1.0,
            's': 1000.0,
        }
        
        # Parse mean time
        patterns = {
            'mean': [
                r'mean[:\s]+([0-9.]+)\s*(ns|us|ms|s)\b',
                r'Time\s+\(mean\s+±\s+σ\):[^\d]*([0-9.]+)\s*(ns|us|ms|s)\b',
            ],
            'median': [
                r'median[:\s]+([0-9.]+)\s*(ns|us|ms|s)\b',
            ],
            'p99': [
                r'p99[:\s]+([0-9.]+)\s*(ns|us|ms|s)\b',
                r'99th percentile[:\s]+([0-9.]+)\s*(ns|us|ms|s)\b',
            ],
            'min': [
                r'min[:\s]+([0-9.]+)\s*(ns|us|ms|s)\b',
                r'Range \(min[^\d]+([0-9.]+)\s*(ns|us|ms|s)\b',
            ],
            'max': [
                r'max[:\s]+([0-9.]+)\s*(ns|us|ms|s)\b',
                r'…\s+max\):[^\d]+([0-9.]+)\s*(ns|us|ms|s)\b',
            ],
            'std_dev': [
                r'std[:\s]+([0-9.]+)\s*(ns|us|ms|s)\b',
                r'±\s+([0-9.]+)\s*(ns|us|ms|s)\b',
            ],
        }
        
        for metric_name, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, output, re.IGNORECASE)
                if match:
                    value = float(match.group(1))
                    unit = match.group(2).lower()
                    # Convert to milliseconds
                    value_ms = value * time_conversions.get(unit, 1.0)
                    metrics[f'{metric_name}_latency_ms'] = value_ms
                    break
        
        # Calculate throughput (items/sec) from mean latency
        if 'mean_latency_ms' in metrics and metrics['mean_latency_ms'] > 0:
            metrics['throughput'] = 1000.0 / metrics['mean_latency_ms']
        
        return metrics
    
    def compare_results(
        self, 
        baseline: BenchmarkResult,
        candidate: BenchmarkResult,
    ) -> Dict[str, Any]:
        """Compare two benchmark results.
        
        Returns:
            Dictionary with comparison metrics
        """
        if not baseline.success or not candidate.success:
            return {
                'valid': False,
                'error': 'One or both benchmarks failed'
            }
        
        speedup = baseline.mean_latency_ms / candidate.mean_latency_ms if candidate.mean_latency_ms > 0 else 0
        latency_reduction = baseline.mean_latency_ms - candidate.mean_latency_ms
        latency_reduction_pct = (latency_reduction / baseline.mean_latency_ms * 100) if baseline.mean_latency_ms > 0 else 0
        
        size_reduction = baseline.binary_size_bytes - candidate.binary_size_bytes
        size_reduction_pct = (size_reduction / baseline.binary_size_bytes * 100) if baseline.binary_size_bytes > 0 else 0
        
        return {
            'valid': True,
            'speedup': speedup,
            'latency_reduction_ms': latency_reduction,
            'latency_reduction_pct': latency_reduction_pct,
            'baseline_latency_ms': baseline.mean_latency_ms,
            'candidate_latency_ms': candidate.mean_latency_ms,
            'size_reduction_bytes': size_reduction,
            'size_reduction_pct': size_reduction_pct,
            'baseline_size_bytes': baseline.binary_size_bytes,
            'candidate_size_bytes': candidate.binary_size_bytes,
        }
