// Model B: Dense classifier 
// This model takes features from Model A and produces classification results.
// Simulates a classifier network like later layers of MobileNet.

builtin.module @model_b {
  // Input: 1x14x14x16 feature map (from Model A)
  // Output: 1x10 class probabilities
  func.func @classify(%input: tensor<1x14x14x16xf32>) -> tensor<1x10xf32> {
    // Flatten: 1x14x14x16 -> 1x3136
    %c1 = arith.constant 1 : index
    %c3136 = arith.constant 3136 : index
    %flattened = tensor.collapse_shape %input [[0], [1, 2, 3]] : tensor<1x14x14x16xf32> into tensor<1x3136xf32>
    
    // Dense layer 1: 3136 -> 64
    %weights1 = arith.constant dense<0.01> : tensor<3136x64xf32>
    %bias1 = arith.constant dense<0.1> : tensor<64xf32>
    %zero1 = arith.constant dense<0.0> : tensor<1x64xf32>
    
    %matmul1 = linalg.matmul
      ins(%flattened, %weights1 : tensor<1x3136xf32>, tensor<3136x64xf32>)
      outs(%zero1 : tensor<1x64xf32>) -> tensor<1x64xf32>
    
    %bias1_broadcast = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%matmul1, %bias1 : tensor<1x64xf32>, tensor<64xf32>)
      outs(%zero1 : tensor<1x64xf32>) {
    ^bb0(%in: f32, %b: f32, %out: f32):
      %add = arith.addf %in, %b : f32
      linalg.yield %add : f32
    } -> tensor<1x64xf32>
    
    // ReLU
    %c0 = arith.constant 0.0 : f32
    %relu1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%bias1_broadcast : tensor<1x64xf32>)
      outs(%zero1 : tensor<1x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %max = arith.maximumf %in, %c0 : f32
      linalg.yield %max : f32
    } -> tensor<1x64xf32>
    
    // Dense layer 2: 64 -> 10
    %weights2 = arith.constant dense<0.01> : tensor<64x10xf32>
    %bias2 = arith.constant dense<0.1> : tensor<10xf32>
    %zero2 = arith.constant dense<0.0> : tensor<1x10xf32>
    
    %matmul2 = linalg.matmul
      ins(%relu1, %weights2 : tensor<1x64xf32>, tensor<64x10xf32>)
      outs(%zero2 : tensor<1x10xf32>) -> tensor<1x10xf32>
    
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%matmul2, %bias2 : tensor<1x10xf32>, tensor<10xf32>)
      outs(%zero2 : tensor<1x10xf32>) {
    ^bb0(%in: f32, %b: f32, %out: f32):
      %add = arith.addf %in, %b : f32
      linalg.yield %add : f32
    } -> tensor<1x10xf32>
    
    // Softmax (simplified)
    return %result : tensor<1x10xf32>
  }
}
