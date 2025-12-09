// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Quantized MatMul with Float-Based Scaling (Before Optimization)
// This demonstrates the inefficient pattern: i32 -> f32 -> scale -> bias -> requant -> i8

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>

module {
  func.func @quantized_matmul_with_float_scaling(
    %input: tensor<16x1024xi8>,
    %weight: tensor<128x1024xi8>
  ) -> tensor<16x128xi8> {
    
    // Constants
    %c0_i32 = arith.constant 0 : i32
    %cst_bias = arith.constant dense<[-776, -1859, -2961, -4696, -5358, -2828, 1339, -3986, -3006, -1438]> : tensor<128xi32>
    %cst_input_scale = arith.constant 1.04389628E-5 : f32  // S_input * S_weight
    %cst_output_scale = arith.constant 0.0168602373 : f32  // S_output
    %cst_min = arith.constant -1.280000e+02 : f32
    %cst_max = arith.constant 1.270000e+02 : f32
    %cst_zp = arith.constant 0.000000e+00 : f32
    
    // Step 1: Convert bias from i32 to f32 (INEFFICIENT!)
    %bias_f32_empty = tensor.empty() : tensor<128xf32>
    %bias_f32 = linalg.generic {
      indexing_maps = [#map, #map], 
      iterator_types = ["parallel"]
    } ins(%cst_bias : tensor<128xi32>) outs(%bias_f32_empty : tensor<128xf32>) {
    ^bb0(%in: i32, %out: f32):
      %f = arith.sitofp %in : i32 to f32
      %scaled = arith.mulf %f, %cst_input_scale : f32
      linalg.yield %scaled : f32
    } -> tensor<128xf32>
    
    // Step 2: Transpose weight
    %weight_transposed_empty = tensor.empty() : tensor<1024x128xi8>
    %weight_transposed = linalg.transpose 
      ins(%weight : tensor<128x1024xi8>) 
      outs(%weight_transposed_empty : tensor<1024x128xi8>) 
      permutation = [1, 0]
    
    // Step 3: Quantized MatMul (Integer Core - Good!)
    %matmul_empty = tensor.empty() : tensor<16x128xi32>
    %matmul_filled = linalg.fill ins(%c0_i32 : i32) outs(%matmul_empty : tensor<16x128xi32>) -> tensor<16x128xi32>
    %matmul_result = linalg.quantized_matmul 
      ins(%input, %weight_transposed, %c0_i32, %c0_i32 : tensor<16x1024xi8>, tensor<1024x128xi8>, i32, i32) 
      outs(%matmul_filled : tensor<16x128xi32>) -> tensor<16x128xi32>
    
    // Step 4: Dequantize to f32 (BOTTLENECK!)
    %dequant_empty = tensor.empty() : tensor<16x128xf32>
    %dequant_result = linalg.generic {
      indexing_maps = [#map1, #map1], 
      iterator_types = ["parallel", "parallel"]
    } ins(%matmul_result : tensor<16x128xi32>) outs(%dequant_empty : tensor<16x128xf32>) {
    ^bb0(%in: i32, %out: f32):
      %f = arith.sitofp %in : i32 to f32
      %scaled = arith.mulf %f, %cst_input_scale : f32
      linalg.yield %scaled : f32
    } -> tensor<16x128xf32>
    
    // Step 5: Add bias in f32 (INEFFICIENT!)
    %biased_empty = tensor.empty() : tensor<16x128xf32>
    %biased_result = linalg.generic {
      indexing_maps = [#map1, #map2, #map1], 
      iterator_types = ["parallel", "parallel"]
    } ins(%dequant_result, %bias_f32 : tensor<16x128xf32>, tensor<128xf32>) 
      outs(%biased_empty : tensor<16x128xf32>) {
    ^bb0(%in: f32, %bias: f32, %out: f32):
      %sum = arith.addf %in, %bias : f32
      linalg.yield %sum : f32
    } -> tensor<16x128xf32>
    
    // Step 6: Requantize to i8 (INEFFICIENT!)
    %requant_empty = tensor.empty() : tensor<16x128xi8>
    %requant_result = linalg.generic {
      indexing_maps = [#map1, #map1], 
      iterator_types = ["parallel", "parallel"]
    } ins(%biased_result : tensor<16x128xf32>) outs(%requant_empty : tensor<16x128xi8>) {
    ^bb0(%in: f32, %out: i8):
      %div = arith.divf %in, %cst_output_scale : f32
      %round = math.roundeven %div : f32
      %add_zp = arith.addf %round, %cst_zp : f32
      %clamp_min = arith.maximumf %add_zp, %cst_min : f32
      %clamp_max = arith.minimumf %clamp_min, %cst_max : f32
      %quant = arith.fptosi %clamp_max : f32 to i8
      linalg.yield %quant : i8
    } -> tensor<16x128xi8>
    
    return %requant_result : tensor<16x128xi8>
  }
}
