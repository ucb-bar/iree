# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Structural verification using LIT tests.

This module provides functionality to generate and run LLVM LIT tests
for verifying the structural properties of compiled MLIR.
"""

import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional


class LitGen:
    """Generator for LLVM LIT structural tests.

    This class creates and runs LIT tests to verify that compiler
    optimizations produce the expected structural changes in the IR.
    """

    # Verification patterns for different optimization strategies
    _VERIFICATION_PATTERNS: Dict[str, Dict[str, List[str]]] = {
        "IntegerRequantization": {
            # Operations that should NOT appear after optimization
            "check_not": [
                "arith.sitofp",
                "arith.mulf",
                "arith.uitofp",
                "arith.subf",
            ],
            # Operations that SHOULD appear after optimization
            "check": [
                "arith.muli",
                "arith.shrsi",
            ],
        },
        "QuantizationFusion": {
            "check_not": [
                "linalg.generic",  # Should be fused
            ],
            "check": [
                "linalg.matmul",  # Or fused matmul variant
            ],
        },
        "Tiling": {
            "check_not": [],
            "check": [
                "scf.forall",  # Tiled loops should appear
            ],
        },
        "Vectorization": {
            "check_not": [],
            "check": [
                "vector.transfer_read",
                "vector.transfer_write",
            ],
        },
    }

    @classmethod
    def generate_structural_test(
        cls,
        strategy: str,
        original_mlir: str,
        compile_flags: Optional[List[str]] = None,
    ) -> str:
        """Creates a .mlir file content ready for llvm-lit.

        Args:
            strategy: The optimization strategy name (e.g., 'IntegerRequantization').
            original_mlir: The original MLIR content to test.
            compile_flags: Optional list of compiler flags to include in RUN line.

        Returns:
            A string containing the complete LIT test file content.
        """
        # Get verification patterns for the strategy
        if strategy in cls._VERIFICATION_PATTERNS:
            patterns = cls._VERIFICATION_PATTERNS[strategy]
        else:
            # Default to empty patterns if strategy not recognized
            patterns = {"check_not": [], "check": []}

        # Build the RUN line
        flags_str = " ".join(compile_flags) if compile_flags else ""
        run_line = f"// RUN: iree-opt {flags_str} %s | FileCheck %s"

        # Build CHECK lines
        check_lines: List[str] = []

        # Add CHECK-NOT lines for operations that should be eliminated
        for op in patterns.get("check_not", []):
            check_lines.append(f"// CHECK-NOT: {op}")

        # Add CHECK lines for operations that should be present
        for op in patterns.get("check", []):
            check_lines.append(f"// CHECK: {op}")

        # Combine all parts
        test_content_parts = [
            run_line,
            "",
        ]
        test_content_parts.extend(check_lines)
        test_content_parts.extend(["", original_mlir])

        return "\n".join(test_content_parts)

    @classmethod
    def generate_custom_test(
        cls,
        original_mlir: str,
        check_patterns: List[str],
        check_not_patterns: List[str],
        compile_flags: Optional[List[str]] = None,
    ) -> str:
        """Creates a custom LIT test with specified patterns.

        Args:
            original_mlir: The original MLIR content to test.
            check_patterns: List of patterns that should be present.
            check_not_patterns: List of patterns that should not be present.
            compile_flags: Optional list of compiler flags.

        Returns:
            A string containing the complete LIT test file content.
        """
        # Build the RUN line
        flags_str = " ".join(compile_flags) if compile_flags else ""
        run_line = f"// RUN: iree-opt {flags_str} %s | FileCheck %s"

        # Build CHECK lines
        check_lines: List[str] = []

        for pattern in check_not_patterns:
            check_lines.append(f"// CHECK-NOT: {pattern}")

        for pattern in check_patterns:
            check_lines.append(f"// CHECK: {pattern}")

        # Combine all parts
        test_content_parts = [
            run_line,
            "",
        ]
        test_content_parts.extend(check_lines)
        test_content_parts.extend(["", original_mlir])

        return "\n".join(test_content_parts)

    @classmethod
    def run_lit(
        cls,
        test_filepath: str,
        lit_executable: str = "llvm-lit",
        timeout: int = 60,
    ) -> bool:
        """Runs llvm-lit on a test file and returns the result.

        Args:
            test_filepath: Path to the LIT test file.
            lit_executable: Path to the llvm-lit executable.
            timeout: Timeout in seconds for the test.

        Returns:
            True if the test passed, False otherwise.
        """
        if not os.path.exists(test_filepath):
            raise FileNotFoundError(f"Test file not found: {test_filepath}")

        try:
            result = subprocess.run(
                [lit_executable, "-v", test_filepath],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except FileNotFoundError:
            raise FileNotFoundError(
                f"llvm-lit executable not found: {lit_executable}. "
                "Please ensure LLVM/LIT is installed and in PATH."
            )

    @classmethod
    def run_lit_with_output(
        cls,
        test_filepath: str,
        lit_executable: str = "llvm-lit",
        timeout: int = 60,
    ) -> Dict[str, Any]:
        """Runs llvm-lit and returns detailed output.

        Args:
            test_filepath: Path to the LIT test file.
            lit_executable: Path to the llvm-lit executable.
            timeout: Timeout in seconds for the test.

        Returns:
            A dictionary containing:
                - 'passed': bool indicating if the test passed
                - 'stdout': stdout from llvm-lit
                - 'stderr': stderr from llvm-lit
                - 'returncode': the return code
        """
        if not os.path.exists(test_filepath):
            raise FileNotFoundError(f"Test file not found: {test_filepath}")

        try:
            result = subprocess.run(
                [lit_executable, "-v", test_filepath],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "passed": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "stdout": "",
                "stderr": f"Test timed out after {timeout} seconds",
                "returncode": -1,
            }
        except FileNotFoundError:
            return {
                "passed": False,
                "stdout": "",
                "stderr": f"llvm-lit executable not found: {lit_executable}",
                "returncode": -1,
            }

    @classmethod
    def create_and_run_test(
        cls,
        strategy: str,
        original_mlir: str,
        compile_flags: Optional[List[str]] = None,
        work_dir: Optional[str] = None,
        lit_executable: str = "llvm-lit",
    ) -> Dict[str, Any]:
        """Creates a test file, runs it, and returns results.

        This is a convenience method that combines test generation and execution.

        Args:
            strategy: The optimization strategy name.
            original_mlir: The original MLIR content to test.
            compile_flags: Optional list of compiler flags.
            work_dir: Optional working directory for the test file.
            lit_executable: Path to the llvm-lit executable.

        Returns:
            A dictionary containing the test results.
        """
        # Generate test content
        test_content = cls.generate_structural_test(
            strategy, original_mlir, compile_flags
        )

        # Create temporary file for the test
        if work_dir:
            os.makedirs(work_dir, exist_ok=True)
            test_file = os.path.join(work_dir, "test.mlir")
            with open(test_file, "w") as f:
                f.write(test_content)
        else:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".mlir", delete=False
            ) as f:
                f.write(test_content)
                test_file = f.name

        try:
            # Run the test
            result = cls.run_lit_with_output(test_file, lit_executable)
            result["test_file"] = test_file
            result["test_content"] = test_content
            return result
        finally:
            # Clean up if using temp file
            if not work_dir and os.path.exists(test_file):
                os.remove(test_file)

    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Returns the list of available verification strategies.

        Returns:
            A list of strategy names that have defined verification patterns.
        """
        return list(cls._VERIFICATION_PATTERNS.keys())
