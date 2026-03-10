#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""IREE-Evo: Autonomous Agentic Compiler Optimization Framework.

A deeply integrated, autonomous agentic flow for the IREE compiler with deep 
semantic understanding and optimization capabilities.
"""

__version__ = "0.1.0"

from iree_evo.orchestrator import Orchestrator
from iree_evo.state_manager import StateManager

__all__ = ["Orchestrator", "StateManager", "__version__"]
