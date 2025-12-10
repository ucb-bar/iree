# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""MLIR IR analysis and error parsing utilities.

This module provides functionality to extract summaries from MLIR content
and parse compilation errors for LLM-based debugging.
"""

import re
from typing import Any, Dict, List, Optional


class MLIRSlicer:
    """Utilities for slicing and analyzing MLIR IR content.

    This class provides methods to extract structural summaries from MLIR IR
    and parse compilation errors for debugging purposes.
    """

    # Patterns for extracting compute operations
    _COMPUTE_OP_PATTERNS = [
        # Linalg named operations
        r"linalg\.(matmul|conv_2d|conv_2d_nchw_fchw|conv_2d_nhwc_hwcf|"
        r"batch_matmul|dot|pooling_nhwc_sum|pooling_nchw_sum|"
        r"generic|fill|copy|transpose|reduce|broadcast)",
        # Tensor operations
        r"tensor\.(empty|extract|extract_slice|insert|insert_slice|"
        r"pad|collapse_shape|expand_shape|concat|generate)",
        # Arithmetic operations for quantization
        r"arith\.(sitofp|fptosi|extui|extsi|trunci|muli|mulf|addi|addf|"
        r"subi|subf|shrsi|shrui|shli|divsi|divui|andi|ori|xori)",
        # Math operations
        r"math\.(exp|log|sqrt|rsqrt|tanh|absf|ceil|floor|round)",
        # Vector operations
        r"vector\.(transfer_read|transfer_write|broadcast|contract|"
        r"reduction|fma|outerproduct)",
        # SCF operations
        r"scf\.(for|forall|if|while|yield)",
    ]

    # Pattern for extracting tensor shapes
    _TENSOR_SHAPE_PATTERN = r"tensor<([^>]+)>"

    # Pattern for extracting memref shapes
    _MEMREF_SHAPE_PATTERN = r"memref<([^>]+)>"

    # Error patterns for parsing compilation output
    _ERROR_PATTERNS = [
        r"error:\s*(.+)",
        r"note:\s*(.+)",
        r"warning:\s*(.+)",
        r"failed to legalize operation\s*'([^']+)'",
        r"'([^']+)' op\s+(.+)",
    ]

    # Pattern for extracting line numbers from error messages
    _LINE_NUMBER_PATTERN = r":(\d+):(\d+):"

    @classmethod
    def extract_summary(cls, mlir_content: str) -> Dict[str, List[str]]:
        """Extracts a structural summary from MLIR content.

        Args:
            mlir_content: The MLIR IR content as a string.

        Returns:
            A dictionary containing:
                - 'compute_ops': List of compute operations found in the IR.
                - 'tensor_shapes': List of tensor shapes found in the IR.
                - 'memref_shapes': List of memref shapes found in the IR.
                - 'quantization_ops': List of quantization-related operations.
        """
        compute_ops: List[str] = []
        tensor_shapes: List[str] = []
        memref_shapes: List[str] = []
        quantization_ops: List[str] = []

        # Extract compute operations
        for pattern in cls._COMPUTE_OP_PATTERNS:
            matches = re.findall(pattern, mlir_content)
            for match in matches:
                if isinstance(match, tuple):
                    # Extract the first non-empty element from the match tuple
                    op_name = ""
                    for element in match:
                        if element:
                            op_name = element
                            break
                else:
                    op_name = match
                if op_name:
                    full_op = re.search(rf"\b\w+\.{re.escape(op_name)}\b", mlir_content)
                    if full_op:
                        compute_ops.append(full_op.group())

        # Extract tensor shapes
        tensor_matches = re.findall(cls._TENSOR_SHAPE_PATTERN, mlir_content)
        tensor_shapes = list(set(tensor_matches))

        # Extract memref shapes
        memref_matches = re.findall(cls._MEMREF_SHAPE_PATTERN, mlir_content)
        memref_shapes = list(set(memref_matches))

        # Identify quantization-related operations
        quant_patterns = [
            r"arith\.sitofp",
            r"arith\.fptosi",
            r"arith\.extui",
            r"arith\.extsi",
            r"arith\.trunci",
            r"arith\.muli",
            r"arith\.shrsi",
            r"arith\.shrui",
            r"arith\.uitofp",
            r"arith\.subf",
            r"arith\.mulf",
        ]
        for pattern in quant_patterns:
            if re.search(pattern, mlir_content):
                quantization_ops.append(pattern.replace(r"\.", "."))

        # Remove duplicates while preserving order
        compute_ops = list(dict.fromkeys(compute_ops))

        return {
            "compute_ops": compute_ops,
            "tensor_shapes": tensor_shapes,
            "memref_shapes": memref_shapes,
            "quantization_ops": quantization_ops,
        }

    @classmethod
    def parse_compilation_error(
        cls, stderr: str, context_lines: int = 5
    ) -> str:
        """Parses compilation error output for LLM debugging.

        Args:
            stderr: The standard error output from a failed iree-compile.
            context_lines: Number of lines of context to include around errors.

        Returns:
            A clean string summarizing the error for the LLM.
        """
        if not stderr:
            return "No error output provided."

        lines = stderr.split("\n")
        error_summary_parts: List[str] = []
        error_locations: List[Dict[str, Any]] = []

        # Find all error lines and their locations
        for i, line in enumerate(lines):
            for pattern in cls._ERROR_PATTERNS:
                match = re.search(pattern, line)
                if match:
                    # Try to extract line number
                    loc_match = re.search(cls._LINE_NUMBER_PATTERN, line)
                    error_locations.append({
                        "index": i,
                        "line": line,
                        "match": match.group(),
                        "line_number": int(loc_match.group(1)) if loc_match else None,
                        "column": int(loc_match.group(2)) if loc_match else None,
                    })
                    break

        if not error_locations:
            # No specific errors found, return the full stderr truncated
            truncated = "\n".join(lines[:50])
            if len(lines) > 50:
                truncated += f"\n... ({len(lines) - 50} more lines)"
            return f"Compilation failed with output:\n{truncated}"

        # Build error summary with context
        error_summary_parts.append("=== COMPILATION ERROR SUMMARY ===\n")

        for error in error_locations[:5]:  # Limit to first 5 errors
            idx = error["index"]
            start = max(0, idx - context_lines)
            end = min(len(lines), idx + context_lines + 1)

            error_summary_parts.append(f"--- Error at line {idx + 1} ---")
            if error["line_number"]:
                error_summary_parts.append(
                    f"Source location: line {error['line_number']}, "
                    f"column {error['column']}"
                )

            # Add context
            context = "\n".join(lines[start:end])
            error_summary_parts.append(f"Context:\n{context}\n")

        # Add primary error message
        primary_errors = [e["match"] for e in error_locations if "error:" in e["line"]]
        if primary_errors:
            error_summary_parts.append("=== PRIMARY ERRORS ===")
            for err in primary_errors[:3]:
                error_summary_parts.append(f"  - {err}")

        return "\n".join(error_summary_parts)

    @classmethod
    def extract_op_at_location(
        cls, mlir_content: str, line_number: int, context_lines: int = 5
    ) -> Optional[str]:
        """Extracts the operation and context at a specific line number.

        Args:
            mlir_content: The MLIR IR content.
            line_number: The line number to extract (1-indexed).
            context_lines: Number of lines of context to include.

        Returns:
            A string with the operation and surrounding context, or None if
            the line number is out of range.
        """
        lines = mlir_content.split("\n")
        if line_number < 1 or line_number > len(lines):
            return None

        idx = line_number - 1  # Convert to 0-indexed
        start = max(0, idx - context_lines)
        end = min(len(lines), idx + context_lines + 1)

        result_lines = []
        for i in range(start, end):
            prefix = ">>> " if i == idx else "    "
            result_lines.append(f"{i + 1:4d} {prefix}{lines[i]}")

        return "\n".join(result_lines)

    @classmethod
    def identify_quantization_pattern(cls, mlir_content: str) -> Optional[str]:
        """Identifies the quantization pattern present in the MLIR.

        Args:
            mlir_content: The MLIR IR content.

        Returns:
            A string identifying the quantization pattern, or None if no
            recognized pattern is found.
        """
        # Check for dequantization pattern (int -> float -> multiply)
        dequant_pattern = (
            r"arith\.extui.*i\d+.*i\d+"
            r".*arith\.uitofp"
            r".*arith\.subf"
            r".*arith\.mulf"
        )
        if re.search(dequant_pattern, mlir_content, re.DOTALL):
            return "GroupedDequantization"

        # Check for requantization pattern (float -> int)
        requant_pattern = r"arith\.fptosi.*arith\.trunci"
        if re.search(requant_pattern, mlir_content, re.DOTALL):
            return "Requantization"

        # Check for integer-only operations
        int_only_pattern = r"arith\.muli.*arith\.shrsi"
        if re.search(int_only_pattern, mlir_content, re.DOTALL):
            return "IntegerOnly"

        # Check for float quantization conversion
        if re.search(r"arith\.sitofp", mlir_content) and re.search(
            r"arith\.mulf", mlir_content
        ):
            return "FloatDequantization"

        return None
