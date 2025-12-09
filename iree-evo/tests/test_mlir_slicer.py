#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for MLIR Slicer."""

import pytest
from pathlib import Path

from iree_evo.perception.mlir_slicer import MLIRSlicer, TensorInfo


def test_parse_simple_matmul():
    """Test parsing a simple matmul MLIR."""
    mlir_content = """
    func.func @matmul(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> {
      %0 = tensor.empty() : tensor<128x128xf32>
      %cst = arith.constant 0.000000e+00 : f32
      %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
      %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>)
                         outs(%1 : tensor<128x128xf32>) -> tensor<128x128xf32>
      return %2 : tensor<128x128xf32>
    }
    """
    
    slicer = MLIRSlicer()
    summary = slicer.parse_string(mlir_content)
    
    assert summary.entry_point == "@matmul"
    assert len(summary.input_signature) == 2
    assert len(summary.output_signature) == 1
    assert summary.input_signature[0].shape == [128, 128]
    assert summary.input_signature[0].element_type == "f32"


def test_parse_dynamic_shapes():
    """Test parsing MLIR with dynamic shapes."""
    mlir_content = """
    func.func @dynamic(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
      return %arg0 : tensor<?x?xf32>
    }
    """
    
    slicer = MLIRSlicer()
    summary = slicer.parse_string(mlir_content)
    
    assert summary.entry_point == "@dynamic"
    assert len(summary.input_signature) == 1
    assert summary.input_signature[0].is_dynamic == True
    assert -1 in summary.input_signature[0].shape


def test_dialect_analysis():
    """Test dialect usage analysis."""
    mlir_content = """
    func.func @test(%arg0: tensor<10xf32>) -> tensor<10xf32> {
      %0 = arith.mulf %arg0, %arg0 : tensor<10xf32>
      %1 = math.exp %0 : tensor<10xf32>
      return %1 : tensor<10xf32>
    }
    """
    
    slicer = MLIRSlicer()
    summary = slicer.parse_string(mlir_content)
    
    assert "arith" in summary.dialect_usage
    assert "math" in summary.dialect_usage


def test_to_concise_summary():
    """Test concise summary generation."""
    mlir_content = """
    func.func @test(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
      return %arg0 : tensor<64x64xf32>
    }
    """
    
    slicer = MLIRSlicer()
    summary = slicer.parse_string(mlir_content)
    concise = slicer.to_concise_summary(summary)
    
    assert "@test" in concise
    assert "64x64xf32" in concise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
