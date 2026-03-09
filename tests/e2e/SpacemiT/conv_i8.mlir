func.func @conv_i8(%input: tensor<1x34x34x8xi8>, %filter: tensor<3x3x8x16xi8>) -> tensor<1x32x32x16xi32> {
  %c0 = arith.constant 0 : i32
  %init = tensor.empty() : tensor<1x32x32x16xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%init : tensor<1x32x32x16xi32>) -> tensor<1x32x32x16xi32>
  %res = linalg.conv_2d_nhwc_hwcf 
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} 
    ins(%input, %filter : tensor<1x34x34x8xi8>, tensor<3x3x8x16xi8>) 
    outs(%fill : tensor<1x32x32x16xi32>) -> tensor<1x32x32x16xi32>
  return %res : tensor<1x32x32x16xi32>
}