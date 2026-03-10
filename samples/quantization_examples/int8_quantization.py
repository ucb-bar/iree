#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
INT8 quantization script for ONNX models.

This script applies INT8 dynamic quantization to an ONNX model, which is
the most common and widely supported quantization format.

Usage:
    python int8_quantization.py --model input.onnx --output output_int8.onnx
    python int8_quantization.py --model input.onnx --output output_int8.onnx --static
"""

import argparse
import os
import sys


def quantize_dynamic(model_path, output_path, use_uint8=True):
    """
    Apply dynamic INT8 quantization.
    
    Dynamic quantization quantizes weights at conversion time and activations
    at runtime. This is simpler than static quantization and doesn't require
    calibration data.
    
    Args:
        model_path: Path to input ONNX model
        output_path: Path to save quantized model
        use_uint8: Use uint8 (QUInt8) vs int8 (QInt8) quantization
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        import onnx
        
        print("Applying INT8 dynamic quantization...")
        print(f"  Input: {model_path}")
        print(f"  Output: {output_path}")
        
        weight_type = QuantType.QUInt8 if use_uint8 else QuantType.QInt8
        print(f"  Weight type: {'QUInt8' if use_uint8 else 'QInt8'}")
        
        quantize_dynamic(
            model_input=model_path,
            model_output=output_path,
            weight_type=weight_type,
        )
        
        # Compare sizes
        original_size = os.path.getsize(model_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        reduction = (1 - quantized_size / original_size) * 100
        
        print("\n✓ Dynamic quantization completed!")
        print(f"  Original size: {original_size:.2f} MB")
        print(f"  Quantized size: {quantized_size:.2f} MB")
        print(f"  Size reduction: {reduction:.1f}%")
        
        return True
        
    except ImportError:
        print("\nError: Required packages not installed.")
        print("Install with: pip install onnxruntime onnx")
        return False
    except Exception as e:
        print(f"\nError during quantization: {e}")
        return False


def quantize_static(model_path, output_path, calibration_data_reader=None):
    """
    Apply static INT8 quantization.
    
    Static quantization quantizes both weights and activations at conversion time,
    using calibration data to determine optimal quantization parameters.
    This typically provides better accuracy than dynamic quantization.
    
    Args:
        model_path: Path to input ONNX model
        output_path: Path to save quantized model
        calibration_data_reader: Optional calibration data reader
    """
    try:
        from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader
        import onnx
        import numpy as np
        
        print("Applying INT8 static quantization...")
        print(f"  Input: {model_path}")
        print(f"  Output: {output_path}")
        
        # If no calibration data provided, create a dummy data reader
        # In practice, you should provide real calibration data
        if calibration_data_reader is None:
            print("  Note: Using dummy calibration data (provide real data for best accuracy)")
            
            class DummyDataReader(CalibrationDataReader):
                def __init__(self, model_path):
                    self.data_index = 0
                    self.num_samples = 10
                    
                    # Load model to get input shape
                    model = onnx.load(model_path)
                    input_tensor = model.graph.input[0]
                    
                    # Parse shape
                    shape = []
                    for dim in input_tensor.type.tensor_type.shape.dim:
                        if dim.dim_value:
                            shape.append(dim.dim_value)
                        else:
                            shape.append(1)  # Default for dynamic dimensions
                    
                    self.input_name = input_tensor.name
                    self.input_shape = shape
                    
                def get_next(self):
                    if self.data_index >= self.num_samples:
                        return None
                    
                    # Generate dummy data (should be real calibration data in practice)
                    data = np.random.randn(*self.input_shape).astype(np.float32)
                    self.data_index += 1
                    return {self.input_name: data}
            
            calibration_data_reader = DummyDataReader(model_path)
        
        quantize_static(
            model_input=model_path,
            model_output=output_path,
            calibration_data_reader=calibration_data_reader,
            quant_format=QuantType.QUInt8,
        )
        
        # Compare sizes
        original_size = os.path.getsize(model_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        reduction = (1 - quantized_size / original_size) * 100
        
        print("\n✓ Static quantization completed!")
        print(f"  Original size: {original_size:.2f} MB")
        print(f"  Quantized size: {quantized_size:.2f} MB")
        print(f"  Size reduction: {reduction:.1f}%")
        print("\n  Note: For production use, provide real calibration data")
        print("        that is representative of your actual inference data.")
        
        return True
        
    except ImportError:
        print("\nError: Required packages not installed.")
        print("Install with: pip install onnxruntime onnx numpy")
        return False
    except Exception as e:
        print(f"\nError during quantization: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="INT8 quantization for ONNX models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to input ONNX model"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save quantized model"
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Use static quantization (requires calibration data)"
    )
    parser.add_argument(
        "--use-int8",
        action="store_true",
        help="Use QInt8 instead of QUInt8 (signed vs unsigned)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return 1
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Apply quantization
    if args.static:
        success = quantize_static(args.model, args.output)
    else:
        success = quantize_dynamic(args.model, args.output, use_uint8=not args.use_int8)
    
    if success:
        print("\n" + "="*60)
        print("Next steps: Compile with IREE")
        print("="*60)
        print("\n# Import to IREE MLIR:")
        print(f"iree-import-onnx {args.output} -o model.mlir")
        print("\n# Compile for CPU:")
        print("iree-compile model.mlir --iree-hal-target-backends=llvm-cpu -o model.vmfb")
        print("\n# Compile for GPU:")
        print("iree-compile model.mlir --iree-hal-target-backends=cuda -o model.vmfb")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
