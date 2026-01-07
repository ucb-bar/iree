#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> ()>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>


func.func @quantized_matmul_512x512(
  %A: tensor<512x512xi8>, 
  %B: tensor<512x512xi8>, 
  %AZp: i8, 
  %BZp: i8, 
  %C: tensor<512x512xi32>
) -> tensor<512x512xi32> {
  
  // Wrap scalars in 0-D tensors for the linalg op
  %azp_tensor = tensor.from_elements %AZp : tensor<i8>
  %bzp_tensor = tensor.from_elements %BZp : tensor<i8>

  %0 = linalg.quantized_matmul ins(%A, %B, %azp_tensor, %bzp_tensor : tensor<512x512xi8>, tensor<512x512xi8>, tensor<i8>, tensor<i8>) outs(%C : tensor<512x512xi32>) indexing_maps = [#map0, #map1, #map2, #map2, #map3] iterator_types = ["parallel", "parallel", "reduction"]

  return %0 : tensor<512x512xi32>
}