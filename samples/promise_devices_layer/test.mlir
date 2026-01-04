func.func @main(
  // Input: Starts on Device A (Core 0)
  %input: tensor<4xf32> {iree.abi.affinity = #hal.device.promise<@device_a>}
) -> (
  // Output: We will return the result residing on Device A
  tensor<4xf32> {iree.abi.affinity = #hal.device.promise<@device_a>}
) {
  // ---------------------------------------------------------
  // STAGE 1: Device A (Core 0)
  // ---------------------------------------------------------
  %c1 = arith.constant dense<1.0> : tensor<4xf32>
  %res_a = arith.addf %input, %c1 : tensor<4xf32>
  
  // ---------------------------------------------------------
  // STAGE 2: Transfer Device A -> Device B (Core 1)
  // ---------------------------------------------------------
  // Explicitly move data to Core 1
  %input_b = flow.tensor.transfer %res_a : tensor<4xf32> to #hal.device.promise<@device_b>
  
  %c2 = arith.constant dense<2.0> : tensor<4xf32>
  %res_b = arith.mulf %input_b, %c2 : tensor<4xf32>

  // ---------------------------------------------------------
  // STAGE 3: Transfer Device B -> Device AB (Cluster 0+1)
  // ---------------------------------------------------------
  // Fan out to the device that manages both cores
  %input_ab = flow.tensor.transfer %res_b : tensor<4xf32> to #hal.device.promise<@device_ab>

  %c10 = arith.constant dense<10.0> : tensor<4xf32>
  %res_ab = arith.addf %input_ab, %c10 : tensor<4xf32>

  // ---------------------------------------------------------
  // STAGE 4: Transfer back to Device A (Core 0) for return
  // ---------------------------------------------------------
  // We loop back to the start device to return the value
  %input_final = flow.tensor.transfer %res_ab : tensor<4xf32> to #hal.device.promise<@device_a>
  
  return %input_final : tensor<4xf32>
}