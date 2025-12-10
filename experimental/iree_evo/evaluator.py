# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""IREE Evaluator for OpenEvolve integration.

This module provides the evaluator class that interfaces with OpenEvolve
for evolutionary optimization of IREE compiler configurations.
"""

import os
import re
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .slicer import MLIRSlicer
from .verification import LitGen


class CompilationError(Exception):
    """Exception raised when compilation fails.

    Attributes:
        message: The error message.
        debug_info: Optional debug information from the compiler.
        stderr: The raw stderr output from the compiler.
    """

    def __init__(
        self,
        message: str,
        debug_info: Optional[str] = None,
        stderr: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.debug_info = debug_info
        self.stderr = stderr

    def __str__(self):
        parts = [self.message]
        if self.debug_info:
            parts.append(f"\nDebug Info:\n{self.debug_info}")
        return "\n".join(parts)


class IREEEvaluator:
    """Evaluator for IREE compiler optimization using OpenEvolve.

    This class implements the evaluation logic for evolutionary optimization
    of IREE compiler configurations. It inherits from openevolve.Evaluator
    when openevolve is available.

    Attributes:
        baseline_mlir_path: Path to the baseline MLIR file.
        target_backend: The target backend (e.g., 'llvm-cpu', 'cuda').
        work_dir: Working directory for intermediate files.
        strategy: The optimization strategy name.
    """

    # Score constants for different outcomes
    COMPILATION_FAILURE_SCORE = -100.0
    STRUCTURAL_FAILURE_SCORE = -50.0
    CORRECTNESS_FAILURE_SCORE = -10.0

    def __init__(
        self,
        baseline_mlir_path: str,
        target_backend: str = "llvm-cpu",
        work_dir: Optional[str] = None,
        strategy: str = "IntegerRequantization",
        iree_compile_path: str = "iree-compile",
        iree_run_module_path: str = "iree-run-module",
        iree_benchmark_module_path: str = "iree-benchmark-module",
        lit_executable: str = "llvm-lit",
        baseline_inputs: Optional[List[str]] = None,
        baseline_expected_outputs: Optional[np.ndarray] = None,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ):
        """Initializes the IREEEvaluator.

        Args:
            baseline_mlir_path: Path to the baseline MLIR file.
            target_backend: The target backend for compilation.
            work_dir: Working directory for intermediate files.
            strategy: The optimization strategy name.
            iree_compile_path: Path to iree-compile executable.
            iree_run_module_path: Path to iree-run-module executable.
            iree_benchmark_module_path: Path to iree-benchmark-module executable.
            lit_executable: Path to llvm-lit executable.
            baseline_inputs: Optional list of input specifications for correctness.
            baseline_expected_outputs: Optional expected outputs for correctness.
            rtol: Relative tolerance for output comparison.
            atol: Absolute tolerance for output comparison.
        """
        if not os.path.exists(baseline_mlir_path):
            raise FileNotFoundError(
                f"Baseline MLIR file not found: {baseline_mlir_path}"
            )

        self.baseline_mlir_path = baseline_mlir_path
        self.target_backend = target_backend
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="iree_evo_")
        self.strategy = strategy
        self.iree_compile_path = iree_compile_path
        self.iree_run_module_path = iree_run_module_path
        self.iree_benchmark_module_path = iree_benchmark_module_path
        self.lit_executable = lit_executable
        self.baseline_inputs = baseline_inputs or []
        self.baseline_expected_outputs = baseline_expected_outputs
        self.rtol = rtol
        self.atol = atol

        # Load baseline MLIR content
        with open(baseline_mlir_path, "r") as f:
            self.baseline_mlir_content = f.read()

        # Ensure work directory exists
        os.makedirs(self.work_dir, exist_ok=True)

    def evaluate(self, individual: str) -> float:
        """Evaluates an individual (compiler configuration).

        This method runs the full evaluation pipeline:
        1. Parse the individual string for flags/scripts
        2. Compile with the given configuration
        3. Verify structural correctness using LIT
        4. Verify numerical correctness
        5. Benchmark performance

        Args:
            individual: A string containing compiler flags and/or transform script.

        Returns:
            A fitness score (higher is better). Returns negative scores for failures:
                -100: Compilation failure
                -50: Structural verification failure
                -10: Correctness failure

        Raises:
            CompilationError: If compilation fails and error details are available.
        """
        # Phase 1: Write individual to file
        flags_path = os.path.join(self.work_dir, "flags.txt")
        with open(flags_path, "w") as f:
            f.write(individual)

        # Parse flags from individual
        flags = self._parse_flags(individual)

        # Phase 2: Compile
        try:
            artifact_path = self._compile(flags)
        except CompilationError as e:
            # Re-raise with debug info for OpenEvolve feedback
            raise

        # Phase 3: Structural Verification
        if not self._verify_structure():
            return self.STRUCTURAL_FAILURE_SCORE

        # Phase 4: Correctness
        if self.baseline_expected_outputs is not None:
            if not self._verify_correctness(artifact_path):
                return self.CORRECTNESS_FAILURE_SCORE

        # Phase 5: Benchmark
        try:
            mean_latency_ms = self._benchmark(artifact_path)
            if mean_latency_ms <= 0:
                return 0.0
            return 1000.0 / mean_latency_ms
        except Exception:
            # If benchmarking fails, return a neutral score
            return 0.0

    def _parse_flags(self, individual: str) -> List[str]:
        """Parses compiler flags from the individual string.

        Args:
            individual: The individual string containing flags.

        Returns:
            A list of compiler flags.
        """
        flags = []
        lines = individual.strip().split("\n")
        for line in lines:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            # Handle flags that may be on separate lines or space-separated
            parts = line.split()
            for part in parts:
                if part.startswith("--") or part.startswith("-"):
                    flags.append(part)
        return flags

    def _compile(self, flags: List[str]) -> str:
        """Compiles the MLIR with given flags.

        Args:
            flags: List of compiler flags.

        Returns:
            Path to the compiled artifact.

        Raises:
            CompilationError: If compilation fails.
        """
        artifact_path = os.path.join(self.work_dir, "module.vmfb")

        # Build compile command
        cmd = [
            self.iree_compile_path,
            self.baseline_mlir_path,
            f"--iree-hal-target-backends={self.target_backend}",
            f"-o={artifact_path}",
        ]
        cmd.extend(flags)

        # First attempt: compile without debug output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            # Compilation failed - get debug dump
            debug_dump = self._get_debug_dump(flags)
            error_summary = MLIRSlicer.parse_compilation_error(result.stderr)

            raise CompilationError(
                message="Compilation failed",
                debug_info=f"{error_summary}\n\n=== DEBUG DUMP ===\n{debug_dump}",
                stderr=result.stderr,
            )

        return artifact_path

    def _get_debug_dump(self, flags: List[str]) -> str:
        """Gets debug dump from a failing compilation.

        Args:
            flags: The original compiler flags.

        Returns:
            Debug dump output from the compiler.
        """
        debug_artifact_path = os.path.join(self.work_dir, "debug_module.vmfb")

        cmd = [
            self.iree_compile_path,
            self.baseline_mlir_path,
            f"--iree-hal-target-backends={self.target_backend}",
            f"-o={debug_artifact_path}",
            "--mlir-print-ir-after-all",
            "--mlir-print-op-on-diagnostic",
        ]
        cmd.extend(flags)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        # Return the combined output (debug info is in stderr), truncated if needed
        return _truncate_output(result.stderr)

    def _verify_structure(self) -> bool:
        """Verifies structural correctness using LIT tests.

        Returns:
            True if structural verification passes, False otherwise.
        """
        result = LitGen.create_and_run_test(
            strategy=self.strategy,
            original_mlir=self.baseline_mlir_content,
            compile_flags=[f"--iree-hal-target-backends={self.target_backend}"],
            work_dir=self.work_dir,
            lit_executable=self.lit_executable,
        )
        return result.get("passed", False)

    def _verify_correctness(self, artifact_path: str) -> bool:
        """Verifies numerical correctness of the compiled module.

        Args:
            artifact_path: Path to the compiled artifact.

        Returns:
            True if outputs match expected values, False otherwise.
        """
        if not os.path.exists(artifact_path):
            return False

        if self.baseline_expected_outputs is None:
            # No expected outputs provided, skip correctness check
            return True

        # Build run command
        cmd = [
            self.iree_run_module_path,
            f"--module={artifact_path}",
        ]

        # Add input specifications
        for input_spec in self.baseline_inputs:
            cmd.append(f"--input={input_spec}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                return False

            # Parse output and compare
            actual_outputs = self._parse_module_output(result.stdout)
            if actual_outputs is None:
                return False

            return np.allclose(
                actual_outputs,
                self.baseline_expected_outputs,
                rtol=self.rtol,
                atol=self.atol,
            )

        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False

    def _parse_module_output(self, stdout: str) -> Optional[np.ndarray]:
        """Parses module output from iree-run-module.

        Args:
            stdout: The stdout from iree-run-module.

        Returns:
            Parsed output as numpy array, or None if parsing fails.
        """
        try:
            # Pattern to match tensor output like "3x3xf32=[1 2 3][4 5 6][7 8 9]"
            pattern = r"(\d+(?:x\d+)*x\w+)=\[([\d\s\.\-e\[\]]+)\]"
            match = re.search(pattern, stdout)
            if not match:
                return None

            shape_str = match.group(1)
            values_str = match.group(2)

            # Parse shape
            parts = shape_str.rsplit("x", 1)
            shape_parts = parts[0].split("x")
            shape = tuple(int(s) for s in shape_parts)

            # Parse values (handle nested brackets)
            values_str = values_str.replace("][", " ")
            values_str = values_str.replace("[", " ").replace("]", " ")
            values = [float(v) for v in values_str.split()]

            return np.array(values).reshape(shape)

        except Exception:
            return None

    def _benchmark(self, artifact_path: str) -> float:
        """Benchmarks the compiled module.

        Args:
            artifact_path: Path to the compiled artifact.

        Returns:
            Mean latency in milliseconds.
        """
        if not os.path.exists(artifact_path):
            return float("inf")

        cmd = [
            self.iree_benchmark_module_path,
            f"--module={artifact_path}",
            "--benchmark_repetitions=5",
        ]

        # Add input specifications if available
        for input_spec in self.baseline_inputs:
            cmd.append(f"--input={input_spec}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                return float("inf")

            # Parse benchmark output for mean latency
            return self._parse_benchmark_output(result.stdout)

        except subprocess.TimeoutExpired:
            return float("inf")
        except Exception:
            return float("inf")

    def _parse_benchmark_output(self, stdout: str) -> float:
        """Parses benchmark output for mean latency.

        Args:
            stdout: The stdout from iree-benchmark-module.

        Returns:
            Mean latency in milliseconds.
        """
        try:
            # Look for mean time in benchmark output
            # Pattern matches lines like: "mean: 1.234 ms"
            patterns = [
                r"mean[:\s]+(\d+\.?\d*)\s*ms",
                r"BM_\w+\s+(\d+\.?\d*)\s*ms",
                r"(\d+\.?\d*)\s*ms\s+mean",
            ]

            for pattern in patterns:
                match = re.search(pattern, stdout, re.IGNORECASE)
                if match:
                    return float(match.group(1))

            # Alternative: look for time values and compute mean
            time_pattern = r"(\d+\.?\d*)\s*ms"
            times = re.findall(time_pattern, stdout)
            if times:
                return sum(float(t) for t in times) / len(times)

            return float("inf")

        except Exception:
            return float("inf")

    def get_baseline_summary(self) -> Dict[str, Any]:
        """Returns a summary of the baseline MLIR for the LLM.

        Returns:
            A dictionary containing the MLIR summary.
        """
        return MLIRSlicer.extract_summary(self.baseline_mlir_content)

    def cleanup(self):
        """Cleans up temporary files in the work directory."""
        import shutil

        if os.path.exists(self.work_dir) and self.work_dir.startswith(
            tempfile.gettempdir()
        ):
            shutil.rmtree(self.work_dir, ignore_errors=True)


def _truncate_output(output: str, max_length: int = 10000) -> str:
    """Truncates output string to a maximum length with ellipsis.

    Args:
        output: The output string to truncate.
        max_length: Maximum allowed length.

    Returns:
        Truncated string if necessary, otherwise the original string.
    """
    if len(output) <= max_length:
        return output
    half = max_length // 2
    truncated_chars = len(output) - max_length
    return (
        output[:half]
        + f"\n... [{truncated_chars} characters truncated] ...\n"
        + output[-half:]
    )


# Try to inherit from openevolve.Evaluator if available
try:
    from openevolve.evaluator import Evaluator as OpenEvolveEvaluator

    class IREEOpenEvolveEvaluator(IREEEvaluator, OpenEvolveEvaluator):
        """IREE Evaluator with OpenEvolve integration.

        This class combines the IREEEvaluator functionality with the
        OpenEvolve Evaluator base class for evolutionary optimization.
        """

        pass

    # Make it available as the preferred evaluator when openevolve is present
    OpenEvolveCompatibleEvaluator = IREEOpenEvolveEvaluator

except ImportError:
    # OpenEvolve not installed, use standalone class
    OpenEvolveCompatibleEvaluator = IREEEvaluator
