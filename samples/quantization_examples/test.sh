#!/bin/bash
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Simple test script to demonstrate quantization examples

set -e

echo "=============================================="
echo "IREE Quantization Examples - Test Script"
echo "=============================================="
echo ""

# Test INT4 documentation generation
echo "1. Testing INT4 quantization documentation..."
python3 int4_quantization.py --output test_int4.txt
if [ -f "test_int4.txt" ] && [ -f "test_int4_examples.mlir" ]; then
    echo "   ✓ INT4 documentation generated successfully"
    rm test_int4.txt test_int4_examples.mlir
else
    echo "   ✗ INT4 documentation generation failed"
    exit 1
fi
echo ""

# Test FP8 documentation generation
echo "2. Testing FP8 quantization documentation..."
python3 fp8_quantization.py --format e4m3fn --output test_fp8.txt
if [ -f "test_fp8.txt" ] && [ -f "test_fp8_examples.mlir" ]; then
    echo "   ✓ FP8 documentation generated successfully"
    rm test_fp8.txt test_fp8_examples.mlir
else
    echo "   ✗ FP8 documentation generation failed"
    exit 1
fi
echo ""

# Test help messages
echo "3. Testing help messages..."
python3 quantize_mobilenet_v2.py --help > /dev/null
python3 int8_quantization.py --help > /dev/null
python3 int4_quantization.py --help > /dev/null
python3 fp8_quantization.py --help > /dev/null
echo "   ✓ All help messages work"
echo ""

echo "=============================================="
echo "All tests passed!"
echo "=============================================="
echo ""
echo "To use these scripts:"
echo "  1. Run quantize_mobilenet_v2.py --download to get a model"
echo "  2. Run quantize_mobilenet_v2.py --model mobilenet_v2.onnx --all"
echo "  3. Check the generated quantization examples"
echo ""
