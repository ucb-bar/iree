func.func @matmul_i8_quantized(%lhs: tensor<32x64xi8>, %rhs: tensor<64x32xi8>) -> tensor<32x32xi32> {
  %c0 = arith.constant 0 : i32
  %lhs_zp = arith.constant 0 : i32
  %rhs_zp = arith.constant 0 : i32

  %init = tensor.empty() : tensor<32x32xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%init : tensor<32x32xi32>) -> tensor<32x32xi32>
  
  %res = linalg.quantized_matmul 
    ins(%lhs, %rhs, %lhs_zp, %rhs_zp : tensor<32x64xi8>, tensor<64x32xi8>, i32, i32) 
    outs(%fill : tensor<32x32xi32>) -> tensor<32x32xi32>
    
  return %res : tensor<32x32xi32>
}