// Model A: Convolutional feature extractor
// This model represents a simple convolutional neural network that processes
// image data and produces features. Output feeds into Model B.
// Simulates a feature extraction network like early layers of MobileNet.

builtin.module @model_a {
  // Input: 1x28x28x3 image
  // Output: 1x14x14x16 feature map
  func.func @extract_features(%input: tensor<1x28x28x3xf32>) -> tensor<1x14x14x16xf32> {
    // Conv2D: 3x3 kernel, stride 1, 16 filters
    %filter = arith.constant dense<1.0> : tensor<3x3x3x16xf32>
    %zero = arith.constant dense<0.0> : tensor<1x26x26x16xf32>
    
    %conv = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x28x28x3xf32>, tensor<3x3x3x16xf32>)
      outs(%zero : tensor<1x26x26x16xf32>) -> tensor<1x26x26x16xf32>
    
    // ReLU activation
    %c0 = arith.constant 0.0 : f32
    %relu = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%conv : tensor<1x26x26x16xf32>)
      outs(%zero : tensor<1x26x26x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %max = arith.maximumf %in, %c0 : f32
      linalg.yield %max : f32
    } -> tensor<1x26x26x16xf32>
    
    // Max pooling 2x2
    %init_pool = tensor.empty() : tensor<1x13x13x16xf32>
    %pool_window = tensor.empty() : tensor<2x2xf32>
    %pooled = linalg.pooling_nhwc_max
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%relu, %pool_window : tensor<1x26x26x16xf32>, tensor<2x2xf32>)
      outs(%init_pool : tensor<1x13x13x16xf32>) -> tensor<1x13x13x16xf32>
    
    // Pad to 14x14
    %padded = tensor.pad %pooled low[0, 0, 0, 0] high[0, 1, 1, 0] {
    ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
      tensor.yield %c0 : f32
    } : tensor<1x13x13x16xf32> to tensor<1x14x14x16xf32>
    
    return %padded : tensor<1x14x14x16xf32>
  }
}
