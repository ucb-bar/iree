#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Execution module initialization."""

from iree_evo.execution.compiler_wrapper import CompilerWrapper, CompilationResult
from iree_evo.execution.benchmark_runner import BenchmarkRunner, BenchmarkResult

__all__ = [
    "CompilerWrapper",
    "CompilationResult",
    "BenchmarkRunner",
    "BenchmarkResult",
]
