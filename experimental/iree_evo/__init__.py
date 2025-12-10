# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""IREE Evolutionary Optimization Package.

This package provides an autonomous optimization system for the IREE compiler
using Evolutionary Strategies via the `openevolve` library.

The system optimizes compilation flags and Transform Dialect scripts,
specifically for Integer-Only Requantization (fusing float dequant/requant
operations into integer math).
"""

from .knowledge_base import KnowledgeBase
from .slicer import MLIRSlicer
from .verification import LitGen
from .evaluator import IREEEvaluator, CompilationError, OpenEvolveCompatibleEvaluator
from .prompts import PLANNER_PROMPT, CODER_PROMPT

__all__ = [
    "KnowledgeBase",
    "MLIRSlicer",
    "LitGen",
    "IREEEvaluator",
    "OpenEvolveCompatibleEvaluator",
    "CompilationError",
    "PLANNER_PROMPT",
    "CODER_PROMPT",
]
