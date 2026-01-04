# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.16.3)

# 1. Dependency Checks
if(NOT IREE_HAL_DRIVER_LOCAL_TASK)
  return()
endif()

set(_NAME "multi_core_runner")

# 2. Define the Executable
add_executable(${_NAME}
  main.c
)

set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "multi-core-runner")

target_compile_definitions(${_NAME} PRIVATE
  _GNU_SOURCE=1
)

# 3. Link IREE Libraries
target_link_libraries(${_NAME}
  PRIVATE
    # High-level runtime
    iree::runtime::runtime

    # The HAL Module
    iree::modules::hal

    # The specific HAL driver (Note the ::task_driver suffix)
    iree::hal::drivers::local_task::task_driver

    # The Task System API
    iree::task::api

    # Internal threading (for affinity masking)
    iree::base::internal::threading

    # Common base
    iree::base::base
)