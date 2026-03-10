#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Test Generator: Dynamically generates verification tests for compiled MLIR."""

import textwrap
from typing import List, Optional
from pathlib import Path

from iree_evo.perception.mlir_slicer import MLIRSummary, TensorInfo


class TestGenerator:
    """Generates Python test scripts for verifying compiled MLIR modules."""
    
    def __init__(self, target_device: str = "local-task"):
        self.target_device = target_device
    
    def generate_test_script(
        self,
        mlir_summary: MLIRSummary,
        baseline_vmfb: Path,
        candidate_vmfb: Path,
        output_path: Path,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> str:
        """Generate a test script to compare baseline and candidate modules.
        
        Args:
            mlir_summary: Summary of the MLIR module
            baseline_vmfb: Path to baseline compiled module
            candidate_vmfb: Path to candidate compiled module
            output_path: Where to save the test script
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison
            
        Returns:
            Path to generated test script
        """
        # Generate input creation code
        input_gen_code = self._generate_input_creation(mlir_summary.input_signature)
        
        # Generate the full test script
        script_content = self._create_test_script_template(
            entry_point=mlir_summary.entry_point.lstrip('@'),
            baseline_vmfb=baseline_vmfb,
            candidate_vmfb=candidate_vmfb,
            input_gen_code=input_gen_code,
            num_inputs=len(mlir_summary.input_signature),
            num_outputs=len(mlir_summary.output_signature),
            rtol=rtol,
            atol=atol,
        )
        
        # Write the script
        with open(output_path, 'w') as f:
            f.write(script_content)
        
        # Make it executable
        output_path.chmod(0o755)
        
        return str(output_path)
    
    def _generate_input_creation(self, input_signature: List[TensorInfo]) -> str:
        """Generate code to create random input tensors."""
        lines = []
        lines.append("    # Generate random inputs")
        lines.append("    inputs = []")
        
        for i, tensor_info in enumerate(input_signature):
            shape = tensor_info.shape
            elem_type = tensor_info.element_type
            
            # Convert shape to concrete dimensions (replace -1 with a reasonable size)
            concrete_shape = [dim if dim > 0 else 32 for dim in shape]
            shape_str = str(concrete_shape)
            
            # Generate appropriate random data based on element type
            if elem_type in ['f32', 'f64', 'f16', 'bf16']:
                lines.append(
                    f"    input_{i} = np.random.randn(*{shape_str}).astype(np.{self._numpy_dtype(elem_type)})"
                )
            elif elem_type in ['i8', 'i16', 'i32', 'i64', 'ui8', 'ui16', 'ui32', 'ui64']:
                lines.append(
                    f"    input_{i} = np.random.randint(-10, 10, size={shape_str}, dtype=np.{self._numpy_dtype(elem_type)})"
                )
            else:
                # Default to float32
                lines.append(
                    f"    input_{i} = np.random.randn(*{shape_str}).astype(np.float32)"
                )
            
            lines.append(f"    inputs.append(input_{i})")
        
        return "\n".join(lines)
    
    def _numpy_dtype(self, mlir_type: str) -> str:
        """Convert MLIR type to numpy dtype string."""
        type_map = {
            'f32': 'float32',
            'f64': 'float64',
            'f16': 'float16',
            'bf16': 'float32',  # numpy doesn't have bfloat16, use float32
            'i8': 'int8',
            'i16': 'int16',
            'i32': 'int32',
            'i64': 'int64',
            'ui8': 'uint8',
            'ui16': 'uint16',
            'ui32': 'uint32',
            'ui64': 'uint64',
        }
        return type_map.get(mlir_type, 'float32')
    
    def _create_test_script_template(
        self,
        entry_point: str,
        baseline_vmfb: Path,
        candidate_vmfb: Path,
        input_gen_code: str,
        num_inputs: int,
        num_outputs: int,
        rtol: float,
        atol: float,
    ) -> str:
        """Create the complete test script."""
        template = f'''#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Auto-generated test script for IREE-Evo verification."""

import sys
import numpy as np

try:
    import iree.runtime as rt
except ImportError:
    print("ERROR: iree.runtime not found. Install with: pip install iree-runtime")
    sys.exit(1)


def load_module(vmfb_path: str, device: str = "{self.target_device}"):
    """Load a compiled IREE module."""
    config = rt.Config(device)
    ctx = rt.SystemContext(config=config)
    
    with open(vmfb_path, 'rb') as f:
        vm_module = rt.VmModule.from_flatbuffer(ctx.instance, f.read())
    
    ctx.add_vm_module(vm_module)
    return ctx


def run_module(ctx, entry_point: str, inputs):
    """Run a module with given inputs."""
    # Get the function
    module_name = list(ctx.modules.keys())[-1]
    func = ctx.modules[module_name][entry_point]
    
    # Run inference
    results = func(*inputs)
    
    # Ensure results is a list
    if not isinstance(results, (list, tuple)):
        results = [results]
    
    return results


def main():
    print("=" * 70)
    print("IREE-Evo Correctness Verification")
    print("=" * 70)
    
    baseline_vmfb = "{baseline_vmfb}"
    candidate_vmfb = "{candidate_vmfb}"
    entry_point = "{entry_point}"
    
    print(f"\\nBaseline:  {{baseline_vmfb}}")
    print(f"Candidate: {{candidate_vmfb}}")
    print(f"Entry point: {{entry_point}}")
    
    # Generate inputs
{input_gen_code}
    
    print(f"\\nGenerated {{len(inputs)}} input tensor(s)")
    for i, inp in enumerate(inputs):
        print(f"  input{{i}}: shape={{inp.shape}}, dtype={{inp.dtype}}")
    
    # Load and run baseline
    print("\\nRunning baseline...")
    try:
        baseline_ctx = load_module(baseline_vmfb)
        baseline_outputs = run_module(baseline_ctx, entry_point, inputs)
        print(f"  ✓ Baseline executed successfully ({{len(baseline_outputs)}} outputs)")
    except Exception as e:
        print(f"  ✗ Baseline execution failed: {{e}}")
        sys.exit(1)
    
    # Load and run candidate
    print("\\nRunning candidate...")
    try:
        candidate_ctx = load_module(candidate_vmfb)
        candidate_outputs = run_module(candidate_ctx, entry_point, inputs)
        print(f"  ✓ Candidate executed successfully ({{len(candidate_outputs)}} outputs)")
    except Exception as e:
        print(f"  ✗ Candidate execution failed: {{e}}")
        sys.exit(1)
    
    # Compare outputs
    print("\\nComparing outputs...")
    rtol = {rtol}
    atol = {atol}
    
    if len(baseline_outputs) != len(candidate_outputs):
        print(f"  ✗ Output count mismatch: baseline={{len(baseline_outputs)}}, candidate={{len(candidate_outputs)}}")
        sys.exit(1)
    
    all_match = True
    for i, (baseline, candidate) in enumerate(zip(baseline_outputs, candidate_outputs)):
        baseline_np = np.array(baseline)
        candidate_np = np.array(candidate)
        
        print(f"\\n  Output {{i}}:")
        print(f"    Shape: {{baseline_np.shape}}")
        print(f"    Dtype: {{baseline_np.dtype}}")
        
        try:
            np.testing.assert_allclose(
                baseline_np,
                candidate_np,
                rtol=rtol,
                atol=atol,
            )
            print(f"    ✓ Match (rtol={{rtol}}, atol={{atol}})")
        except AssertionError as e:
            print(f"    ✗ Mismatch: {{e}}")
            all_match = False
            
            # Print some statistics about the difference
            diff = np.abs(baseline_np - candidate_np)
            print(f"    Max absolute difference: {{np.max(diff)}}")
            print(f"    Mean absolute difference: {{np.mean(diff)}}")
    
    print("\\n" + "=" * 70)
    if all_match:
        print("✓ VERIFICATION PASSED: All outputs match!")
        print("=" * 70)
        return 0
    else:
        print("✗ VERIFICATION FAILED: Outputs do not match!")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
'''
        return template
