module {
  func.func @matmul_i8(%lhs: tensor<16x16xi8>, %rhs: tensor<16x16xi8>, %acc: tensor<16x16xi32>) -> tensor<16x16xi32> {
    %0 = linalg.matmul
      ins(%lhs, %rhs : tensor<16x16xi8>, tensor<16x16xi8>)
      outs(%acc : tensor<16x16xi32>)
      -> tensor<16x16xi32>
    return %0 : tensor<16x16xi32>
  }
}
