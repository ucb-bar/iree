func.func @vanilla_matmul_large(%A: tensor<128x128xi8>, %B: tensor<128x128xi8>, %C: tensor<128x128xi32>) -> tensor<128x128xi32> {
  // 128x128 is the "Goldilocks" size: 
  // Large enough to trigger "Data Tiling" (matmul -> mmt4d).
  // Matches your microkernel's tiling requirements.
  %0 = linalg.matmul ins(%A, %B : tensor<128x128xi8>, tensor<128x128xi8>)
                     outs(%C : tensor<128x128xi32>) -> tensor<128x128xi32>
  return %0 : tensor<128x128xi32>
}