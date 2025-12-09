#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compiler Wrapper: Safe subprocess-based compilation with timeout handling."""

import subprocess
import time
from typing import List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass


@dataclass
class CompilationResult:
    """Result of a compilation attempt."""
    success: bool
    output_file: Optional[Path]
    stdout: str
    stderr: str
    compilation_time: float
    exit_code: int
    timed_out: bool = False


class CompilerWrapper:
    """Wrapper for iree-compile with safe subprocess handling."""
    
    def __init__(self, iree_compile_path: str = "iree-compile"):
        self.iree_compile_path = iree_compile_path
    
    def compile(
        self,
        mlir_file: Path,
        output_file: Path,
        flags: List[str],
        timeout: int = 300,
    ) -> CompilationResult:
        """Compile MLIR to VMFB with given flags.
        
        Args:
            mlir_file: Input MLIR file
            output_file: Output VMFB file
            flags: Compilation flags
            timeout: Timeout in seconds
            
        Returns:
            CompilationResult with compilation status and outputs
        """
        cmd = [
            self.iree_compile_path,
            str(mlir_file),
            "-o", str(output_file),
        ] + flags
        
        start_time = time.time()
        stdout_str = ""
        stderr_str = ""
        timed_out = False
        exit_code = -1
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            stdout_str = result.stdout
            stderr_str = result.stderr
            exit_code = result.returncode
            success = (exit_code == 0)
            
        except subprocess.TimeoutExpired as e:
            timed_out = True
            success = False
            stderr_str = f"Compilation timed out after {timeout}s"
            if e.stdout:
                stdout_str = e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout
            if e.stderr:
                stderr_str += f"\n{e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr}"
        
        except FileNotFoundError:
            success = False
            stderr_str = f"iree-compile not found at: {self.iree_compile_path}"
        
        except Exception as e:
            success = False
            stderr_str = f"Unexpected error: {str(e)}"
        
        compilation_time = time.time() - start_time
        
        return CompilationResult(
            success=success,
            output_file=output_file if success else None,
            stdout=stdout_str,
            stderr=stderr_str,
            compilation_time=compilation_time,
            exit_code=exit_code,
            timed_out=timed_out,
        )
    
    def compile_with_transform_dialect(
        self,
        mlir_file: Path,
        transform_script: Path,
        output_file: Path,
        base_flags: List[str],
        timeout: int = 300,
    ) -> CompilationResult:
        """Compile with Transform Dialect script.
        
        Args:
            mlir_file: Input MLIR file
            transform_script: Transform dialect script file
            output_file: Output VMFB file
            base_flags: Base compilation flags
            timeout: Timeout in seconds
            
        Returns:
            CompilationResult
        """
        flags = base_flags + [
            f"--iree-codegen-transform-dialect-library={transform_script}",
        ]
        
        return self.compile(mlir_file, output_file, flags, timeout)
    
    def validate_flags(self, flags: List[str]) -> Tuple[bool, Optional[str]]:
        """Validate compilation flags by checking help output.
        
        Args:
            flags: List of flags to validate
            
        Returns:
            Tuple of (valid, error_message)
        """
        # Get help output
        try:
            result = subprocess.run(
                [self.iree_compile_path, "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            help_text = result.stdout + result.stderr
            
            # Check each flag
            for flag in flags:
                # Extract flag name (remove value)
                flag_name = flag.split('=')[0]
                
                # Check if flag exists in help
                if flag_name not in help_text:
                    return False, f"Unknown flag: {flag_name}"
            
            return True, None
        
        except Exception as e:
            return False, f"Could not validate flags: {str(e)}"
