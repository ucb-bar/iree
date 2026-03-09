func.func @conv_fp8(%input: tensor<1x34x34x8xf8E4M3FN>, %filter: tensor<3x3x8x16xf8E4M3FN>) -> tensor<1x32x32x16xf16> {
  %c0 = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<1x32x32x16xf16>
  %fill = linalg.fill ins(%c0 : f16) outs(%init : tensor<1x32x32x16xf16>) -> tensor<1x32x32x16xf16>
  %res = linalg.conv_2d_nhwc_hwcf 
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} 
    ins(%input, %filter : tensor<1x34x34x8xf8E4M3FN>, tensor<3x3x8x16xf8E4M3FN>) 
    outs(%fill : tensor<1x32x32x16xf16>) -> tensor<1x32x32x16xf16>
  return %res : tensor<1x32x32x16xf16>
}