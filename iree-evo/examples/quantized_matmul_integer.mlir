// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Quantized MatMul with Integer-Only Scaling (After Optimization)
// This demonstrates the efficient pattern: i32 -> fixed-point scale -> bias -> i8

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>

module {
  func.func @quantized_matmul_integer_only(
    %input: tensor<16x1024xi8>,
    %weight: tensor<128x1024xi8>
  ) -> tensor<16x128xi8> {
    
    // Constants
    %c0_i32 = arith.constant 0 : i32
    %cst_bias = arith.constant dense<[-776, -1859, -2961, -4696, -5358, -2828, 1339, -3986, -3006, -1438]> : tensor<128xi32>
    
    // Fixed-point scale parameters
    // M = (S_input * S_weight) / S_output = 1.04389628E-5 / 0.0168602373 ≈ 0.0006191469
    // M = M0 * 2^(-n), where n=31 (Q31 format)
    // M0 = floor(0.0006191469 * 2^31) = 1329634
    %c_multiplier = arith.constant 1329634 : i32
    %c_shift = arith.constant 31 : i32
    %c_zp_out = arith.constant 0 : i32
    %c_min = arith.constant -128 : i32
    %c_max = arith.constant 127 : i32
    
    // Step 1: Transpose weight
    %weight_transposed_empty = tensor.empty() : tensor<1024x128xi8>
    %weight_transposed = linalg.transpose 
      ins(%weight : tensor<128x1024xi8>) 
      outs(%weight_transposed_empty : tensor<1024x128xi8>) 
      permutation = [1, 0]
    
    // Step 2: Quantized MatMul (Integer Core)
    %matmul_empty = tensor.empty() : tensor<16x128xi32>
    %matmul_filled = linalg.fill ins(%c0_i32 : i32) outs(%matmul_empty : tensor<16x128xi32>) -> tensor<16x128xi32>
    %matmul_result = linalg.quantized_matmul 
      ins(%input, %weight_transposed, %c0_i32, %c0_i32 : tensor<16x1024xi8>, tensor<1024x128xi8>, i32, i32) 
      outs(%matmul_filled : tensor<16x128xi32>) -> tensor<16x128xi32>
    
    // Step 3: Add bias in i32 (Efficient!)
    %biased_empty = tensor.empty() : tensor<16x128xi32>
    %biased_result = linalg.generic {
      indexing_maps = [#map1, #map2, #map1], 
      iterator_types = ["parallel", "parallel"]
    } ins(%matmul_result, %cst_bias : tensor<16x128xi32>, tensor<128xi32>) 
      outs(%biased_empty : tensor<16x128xi32>) {
    ^bb0(%acc: i32, %bias: i32, %out: i32):
      %sum = arith.addi %acc, %bias : i32
      linalg.yield %sum : i32
    } -> tensor<16x128xi32>
    
    // Step 4: Integer-only requantization with fixed-point math
    %requant_empty = tensor.empty() : tensor<16x128xi8>
    %requant_result = linalg.generic {
      indexing_maps = [#map1, #map1], 
      iterator_types = ["parallel", "parallel"]
    } ins(%biased_result : tensor<16x128xi32>) outs(%requant_empty : tensor<16x128xi8>) {
    ^bb0(%in_val: i32, %out: i8):
      // Fixed-point multiplication (extend to i64 to avoid overflow)
      %val_i64 = arith.extsi %in_val : i32 to i64
      %mult_i64 = arith.extsi %c_multiplier : i32 to i64
      %prod = arith.muli %val_i64, %mult_i64 : i64
      
      // Arithmetic right shift
      %shift_i64 = arith.extsi %c_shift : i32 to i64
      %scaled_i64 = arith.shrsi %prod, %shift_i64 : i64
      %scaled = arith.trunci %scaled_i64 : i64 to i32
      
      // Zero-point addition (0 for symmetric quantization)
      %final_val = arith.addi %scaled, %c_zp_out : i32
      
      // Clamp to i8 range
      %clamped_low = arith.maxsi %final_val, %c_min : i32
      %clamped = arith.minsi %clamped_low, %c_max : i32
      
      // Cast to i8
      %res_i8 = arith.trunci %clamped : i32 to i8
      
      linalg.yield %res_i8 : i8
    } -> tensor<16x128xi8>
    
    return %requant_result : tensor<16x128xi8>
  }
}
