#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from setuptools import setup, find_packages

setup(
    name="iree-evo",
    packages=find_packages(),
    package_data={
        "iree_evo": ["py.typed"],
    },
)
