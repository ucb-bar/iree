#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Error Log Cleaner: Extracts relevant error information from compilation logs."""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ErrorInfo:
    """Structured error information."""
    error_message: str
    mlir_context: List[str]
    error_location: Optional[str] = None
    error_type: Optional[str] = None


class ErrorLogCleaner:
    """Cleans and extracts relevant error information from compiler output."""
    
    def __init__(self):
        # Patterns for error detection
        self.error_patterns = [
            re.compile(r'error:\s*(.+)'),
            re.compile(r'ERROR:\s*(.+)'),
            re.compile(r'failed to\s+(.+)', re.IGNORECASE),
            re.compile(r'unable to\s+(.+)', re.IGNORECASE),
        ]
        
        # Pattern for location information
        self.location_pattern = re.compile(
            r'(\w+\.mlir):(\d+):(\d+):'
        )
        
        # Pattern for MLIR operation lines
        self.mlir_op_pattern = re.compile(r'^\s*%\w+\s*=\s*\S+')
    
    def clean_stderr(self, stderr: str) -> ErrorInfo:
        """Extract clean error information from stderr output.
        
        Args:
            stderr: Raw stderr output from compilation
            
        Returns:
            ErrorInfo with extracted error details
        """
        lines = stderr.split('\n')
        
        # Find the main error message
        error_msg = self._extract_error_message(lines)
        
        # Extract location information
        error_location = self._extract_location(stderr)
        
        # Extract MLIR context around the error
        mlir_context = self._extract_mlir_context(lines, error_location)
        
        # Determine error type
        error_type = self._classify_error(error_msg)
        
        return ErrorInfo(
            error_message=error_msg,
            mlir_context=mlir_context,
            error_location=error_location,
            error_type=error_type
        )
    
    def _extract_error_message(self, lines: List[str]) -> str:
        """Extract the primary error message from log lines."""
        error_messages = []
        
        for line in lines:
            for pattern in self.error_patterns:
                match = pattern.search(line)
                if match:
                    error_messages.append(match.group(1).strip())
        
        if error_messages:
            # Return the first substantial error message
            for msg in error_messages:
                if len(msg) > 20:  # Filter out very short messages
                    return msg
            return error_messages[0]
        
        return "Unknown compilation error"
    
    def _extract_location(self, stderr: str) -> Optional[str]:
        """Extract file location from error output."""
        match = self.location_pattern.search(stderr)
        if match:
            filename = match.group(1)
            line = match.group(2)
            col = match.group(3)
            return f"{filename}:{line}:{col}"
        return None
    
    def _extract_mlir_context(
        self, lines: List[str], location: Optional[str]
    ) -> List[str]:
        """Extract MLIR code context around the error.
        
        Args:
            lines: All lines from stderr
            location: Error location string
            
        Returns:
            List of MLIR lines providing context (up to 5 lines before and after)
        """
        mlir_lines = []
        context_window = 5
        
        # If we have a location, try to find the specific line
        if location and ':' in location:
            try:
                # Try to extract line number from location
                parts = location.split(':')
                if len(parts) >= 2:
                    target_line = int(parts[1])
                    
                    # Collect lines around the error
                    for i, line in enumerate(lines, 1):
                        if abs(i - target_line) <= context_window:
                            if self._is_mlir_line(line):
                                mlir_lines.append(line.strip())
            except (ValueError, IndexError):
                pass
        
        # If we couldn't extract based on location, look for MLIR-like lines
        if not mlir_lines:
            for line in lines:
                if self._is_mlir_line(line):
                    mlir_lines.append(line.strip())
                    if len(mlir_lines) >= 10:  # Limit context
                        break
        
        return mlir_lines
    
    def _is_mlir_line(self, line: str) -> bool:
        """Check if a line looks like MLIR code."""
        # Look for common MLIR patterns
        mlir_indicators = [
            '%', '=', 'func.func', 'linalg.', 'tensor<',
            'arith.', 'scf.', 'flow.', 'hal.'
        ]
        return any(indicator in line for indicator in mlir_indicators)
    
    def _classify_error(self, error_msg: str) -> str:
        """Classify the type of error based on the message."""
        error_msg_lower = error_msg.lower()
        
        if 'type' in error_msg_lower or 'mismatch' in error_msg_lower:
            return "type_error"
        elif 'shape' in error_msg_lower or 'dimension' in error_msg_lower:
            return "shape_error"
        elif 'unknown' in error_msg_lower or 'not found' in error_msg_lower:
            return "symbol_error"
        elif 'syntax' in error_msg_lower or 'parse' in error_msg_lower:
            return "syntax_error"
        elif 'flag' in error_msg_lower or 'option' in error_msg_lower:
            return "flag_error"
        elif 'backend' in error_msg_lower or 'target' in error_msg_lower:
            return "backend_error"
        else:
            return "unknown_error"
    
    def format_error_report(self, error_info: ErrorInfo) -> str:
        """Format error information as a readable report."""
        lines = []
        lines.append("=" * 70)
        lines.append("COMPILATION ERROR REPORT")
        lines.append("=" * 70)
        
        if error_info.error_type:
            lines.append(f"\nError Type: {error_info.error_type}")
        
        if error_info.error_location:
            lines.append(f"Location: {error_info.error_location}")
        
        lines.append(f"\nError Message:")
        lines.append(f"  {error_info.error_message}")
        
        if error_info.mlir_context:
            lines.append(f"\nMLIR Context:")
            for i, context_line in enumerate(error_info.mlir_context, 1):
                lines.append(f"  {i:2d}| {context_line}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
