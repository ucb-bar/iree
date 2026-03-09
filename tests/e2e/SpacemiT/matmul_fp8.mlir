func.func @matmul_fp8(%lhs: tensor<32x64xf8E4M3FN>, %rhs: tensor<64x32xf8E4M3FN>) -> tensor<32x32xf16> {
  %c0 = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<32x32xf16>
  %fill = linalg.fill ins(%c0 : f16) outs(%init : tensor<32x32xf16>) -> tensor<32x32xf16>
  %res = linalg.matmul ins(%lhs, %rhs : tensor<32x64xf8E4M3FN>, tensor<64x32xf8E4M3FN>) outs(%fill : tensor<32x32xf16>) -> tensor<32x32xf16>
  return %res : tensor<32x32xf16>
}