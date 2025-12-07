// Model C: Independent lightweight processing network
// This model runs independently and simulates a different workload like
// audio processing or sensor fusion. Represents a ResNet-like bottleneck.

builtin.module @model_c {
  // Input: 1x32x32x8 (smaller input, faster processing)
  // Output: 1x32x32x8 (same shape, residual connection)
  func.func @process_data(%input: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
    %c0 = arith.constant 0.0 : f32
    
    // First 1x1 conv to reduce channels: 8 -> 4
    %filter1 = arith.constant dense<0.1> : tensor<1x1x8x4xf32>
    %zero1 = arith.constant dense<0.0> : tensor<1x32x32x4xf32>
    
    %conv1 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter1 : tensor<1x32x32x8xf32>, tensor<1x1x8x4xf32>)
      outs(%zero1 : tensor<1x32x32x4xf32>) -> tensor<1x32x32x4xf32>
    
    // ReLU
    %relu1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%conv1 : tensor<1x32x32x4xf32>)
      outs(%zero1 : tensor<1x32x32x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %max = arith.maximumf %in, %c0 : f32
      linalg.yield %max : f32
    } -> tensor<1x32x32x4xf32>
    
    // 3x3 conv at bottleneck: 4 -> 4
    %filter2 = arith.constant dense<0.1> : tensor<3x3x4x4xf32>
    %zero2 = arith.constant dense<0.0> : tensor<1x30x30x4xf32>
    
    %conv2 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%relu1, %filter2 : tensor<1x32x32x4xf32>, tensor<3x3x4x4xf32>)
      outs(%zero2 : tensor<1x30x30x4xf32>) -> tensor<1x30x30x4xf32>
    
    // ReLU
    %relu2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%conv2 : tensor<1x30x30x4xf32>)
      outs(%zero2 : tensor<1x30x30x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %max = arith.maximumf %in, %c0 : f32
      linalg.yield %max : f32
    } -> tensor<1x30x30x4xf32>
    
    // Pad back to 32x32
    %padded = tensor.pad %relu2 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
      tensor.yield %c0 : f32
    } : tensor<1x30x30x4xf32> to tensor<1x32x32x4xf32>
    
    // Final 1x1 conv to expand: 4 -> 8
    %filter3 = arith.constant dense<0.1> : tensor<1x1x4x8xf32>
    %zero3 = arith.constant dense<0.0> : tensor<1x32x32x8xf32>
    
    %conv3 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%padded, %filter3 : tensor<1x32x32x4xf32>, tensor<1x1x4x8xf32>)
      outs(%zero3 : tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
    
    // Residual connection: add input
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%conv3, %input : tensor<1x32x32x8xf32>, tensor<1x32x32x8xf32>)
      outs(%zero3 : tensor<1x32x32x8xf32>) {
    ^bb0(%in1: f32, %in2: f32, %out: f32):
      %add = arith.addf %in1, %in2 : f32
      linalg.yield %add : f32
    } -> tensor<1x32x32x8xf32>
    
    return %result : tensor<1x32x32x8xf32>
  }
}
