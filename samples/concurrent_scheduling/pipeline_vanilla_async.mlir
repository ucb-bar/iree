// RUN: (iree-compile --iree-execution-model=async-external --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu %p/model_a_conv.mlir -o=%t.model_a.vmfb && \
// RUN:  iree-compile --iree-execution-model=async-external --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu %p/model_b_dense.mlir -o=%t.model_b.vmfb && \
// RUN:  iree-compile --iree-execution-model=async-external --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu %p/model_c_residual.mlir -o=%t.model_c.vmfb && \
// RUN:  iree-compile --iree-execution-model=async-external --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu %s | \
// RUN:  iree-run-module \
// RUN:    --device=local-task \
// RUN:    --module=%t.model_a.vmfb \
// RUN:    --module=%t.model_b.vmfb \
// RUN:    --module=%t.model_c.vmfb \
// RUN:    --module=- --function=pipeline_vanilla) | \
// RUN:  FileCheck %s
// CHECK: VMFB

// Vanilla IREE Concurrent Pipeline Orchestrator
// This orchestrates three models running concurrently:
// - Model A (conv) -> Model B (dense): Dependent pipeline
// - Model C (residual): Independent workload
//
// IREE's built-in scheduler handles all the async execution and dependency
// management automatically using coarse-fences ABI model.

// External module declarations with async execution model
func.func private @model_a.extract_features(%input: tensor<1x28x28x3xf32>) -> tensor<1x14x14x16xf32> attributes {
  iree.abi.model = "coarse-fences"
}

func.func private @model_b.classify(%input: tensor<1x14x14x16xf32>) -> tensor<1x10xf32> attributes {
  iree.abi.model = "coarse-fences"
}

func.func private @model_c.process_data(%input: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> attributes {
  iree.abi.model = "coarse-fences"
}

// Main pipeline function - vanilla IREE scheduling
// IREE automatically handles:
// - Async execution of all models
// - Dependency tracking (A->B)
// - Concurrent execution of independent work (C)
// - Optimal resource allocation
func.func @pipeline_vanilla(
    %input_a: tensor<1x28x28x3xf32>,
    %input_c: tensor<1x32x32x8xf32>
) -> (tensor<1x10xf32>, tensor<1x32x32x8xf32>) {
  
  // Launch Model A - starts immediately
  // Since this is async, execution is scheduled but not blocking
  %features = call @model_a.extract_features(%input_a) 
    : (tensor<1x28x28x3xf32>) -> tensor<1x14x14x16xf32>
  
  // Launch Model C concurrently - independent of A and B
  // IREE scheduler can run this in parallel with A
  %result_c = call @model_c.process_data(%input_c) 
    : (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  
  // Launch Model B - depends on Model A output
  // IREE automatically waits for %features to be ready via fence
  %result_b = call @model_b.classify(%features) 
    : (tensor<1x14x14x16xf32>) -> tensor<1x10xf32>
  
  // Return both results
  // Implicitly waits for both B and C to complete
  return %result_b, %result_c : tensor<1x10xf32>, tensor<1x32x32x8xf32>
}

// Multi-iteration pipeline to simulate different frequencies
// Model A+B runs every iteration (high frequency)
// Model C runs every other iteration (lower frequency)
func.func @pipeline_vanilla_multi_freq(
    %input_a: tensor<1x28x28x3xf32>,
    %input_c: tensor<1x32x32x8xf32>,
    %iterations: i32
) -> (tensor<1x10xf32>, tensor<1x32x32x8xf32>) {
  
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  
  // Initial dummy results
  %init_b = arith.constant dense<0.0> : tensor<1x10xf32>
  %init_c = arith.constant dense<0.0> : tensor<1x32x32x8xf32>
  
  // Loop over iterations
  %final_b, %final_c = scf.for %iter = %c0 to %iterations step %c1 
    iter_args(%result_b = %init_b, %result_c = %init_c) 
    -> (tensor<1x10xf32>, tensor<1x32x32x8xf32>) {
    
    // Always run A->B pipeline (high frequency)
    %features = call @model_a.extract_features(%input_a) 
      : (tensor<1x28x28x3xf32>) -> tensor<1x14x14x16xf32>
    %new_result_b = call @model_b.classify(%features) 
      : (tensor<1x14x14x16xf32>) -> tensor<1x10xf32>
    
    // Run C every other iteration (lower frequency)
    %iter_mod = arith.remsi %iter, %c2 : i32
    %should_run_c = arith.cmpi eq, %iter_mod, %c0 : i32
    %new_result_c = scf.if %should_run_c -> (tensor<1x32x32x8xf32>) {
      %c_out = call @model_c.process_data(%input_c) 
        : (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
      scf.yield %c_out : tensor<1x32x32x8xf32>
    } else {
      scf.yield %result_c : tensor<1x32x32x8xf32>
    }
    
    scf.yield %new_result_b, %new_result_c : tensor<1x10xf32>, tensor<1x32x32x8xf32>
  }
  
  return %final_b, %final_c : tensor<1x10xf32>, tensor<1x32x32x8xf32>
}
