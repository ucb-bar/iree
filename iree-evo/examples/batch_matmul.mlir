// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Batch matmul example for IREE-Evo optimization

func.func @batch_matmul(%arg0: tensor<8x256x512xf32>, %arg1: tensor<8x512x256xf32>) -> tensor<8x256x256xf32> {
  %0 = tensor.empty() : tensor<8x256x256xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<8x256x256xf32>) -> tensor<8x256x256xf32>
  %2 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<8x256x512xf32>, tensor<8x512x256xf32>)
                           outs(%1 : tensor<8x256x256xf32>) -> tensor<8x256x256xf32>
  return %2 : tensor<8x256x256xf32>
}
