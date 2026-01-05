func.func @vanilla_matmul(%A: tensor<8x8xf32>, %B: tensor<8x8xf32>, %C: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // 8x8xf32 = operating on 256 bits of data per dimension
  %0 = linalg.matmul ins(%A, %B : tensor<8x8xf32>, tensor<8x8xf32>)
                     outs(%C : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}