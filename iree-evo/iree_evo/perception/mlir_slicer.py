#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""MLIR Slicer: Extracts critical components from MLIR for LLM consumption."""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class TensorInfo:
    """Information about a tensor in MLIR."""
    shape: List[int]
    element_type: str
    is_dynamic: bool = False


@dataclass
class MLIROperation:
    """Structured representation of an MLIR operation."""
    op_name: str
    location: str
    inputs: List[TensorInfo]
    outputs: List[TensorInfo]
    attributes: Dict[str, Any]
    line_number: Optional[int] = None


@dataclass
class MLIRSummary:
    """High-level summary of MLIR IR."""
    entry_point: str
    input_signature: List[TensorInfo]
    output_signature: List[TensorInfo]
    compute_ops: List[MLIROperation]
    total_ops: int
    has_dispatch_regions: bool
    dialect_usage: Dict[str, int]


class MLIRSlicer:
    """Parser that extracts critical MLIR components for analysis."""
    
    def __init__(self):
        # Regex patterns for parsing MLIR
        self.func_pattern = re.compile(
            r'func\.func\s+(@\w+)\s*\((.*?)\)\s*->\s*\((.*?)\)',
            re.DOTALL
        )
        self.op_pattern = re.compile(
            r'%(\w+)\s*=\s*(\S+)\s*\((.*?)\)',
            re.DOTALL
        )
        self.tensor_pattern = re.compile(
            r'tensor<([\d\?x]+)x([a-z0-9]+)>'
        )
        self.dispatch_pattern = re.compile(
            r'flow\.dispatch'
        )
        
    def parse_file(self, mlir_file: Path) -> MLIRSummary:
        """Parse an MLIR file and extract a summary.
        
        Args:
            mlir_file: Path to the MLIR file
            
        Returns:
            MLIRSummary with extracted information
        """
        with open(mlir_file, 'r') as f:
            mlir_content = f.read()
        
        return self.parse_string(mlir_content)
    
    def parse_string(self, mlir_content: str) -> MLIRSummary:
        """Parse MLIR content string and extract a summary.
        
        Args:
            mlir_content: MLIR IR as a string
            
        Returns:
            MLIRSummary with extracted information
        """
        lines = mlir_content.split('\n')
        
        # Extract function signature
        entry_point, input_sig, output_sig = self._extract_function_signature(mlir_content)
        
        # Extract compute operations
        compute_ops = self._extract_compute_operations(mlir_content, lines)
        
        # Count total operations
        total_ops = self._count_operations(mlir_content)
        
        # Check for dispatch regions
        has_dispatch = bool(self.dispatch_pattern.search(mlir_content))
        
        # Analyze dialect usage
        dialect_usage = self._analyze_dialects(mlir_content)
        
        return MLIRSummary(
            entry_point=entry_point,
            input_signature=input_sig,
            output_signature=output_sig,
            compute_ops=compute_ops,
            total_ops=total_ops,
            has_dispatch_regions=has_dispatch,
            dialect_usage=dialect_usage
        )
    
    def _extract_function_signature(
        self, mlir_content: str
    ) -> Tuple[str, List[TensorInfo], List[TensorInfo]]:
        """Extract function entry point and signature."""
        func_match = self.func_pattern.search(mlir_content)
        
        if not func_match:
            return "unknown", [], []
        
        entry_point = func_match.group(1)
        inputs_str = func_match.group(2)
        outputs_str = func_match.group(3)
        
        input_tensors = self._parse_tensor_list(inputs_str)
        output_tensors = self._parse_tensor_list(outputs_str)
        
        return entry_point, input_tensors, output_tensors
    
    def _parse_tensor_list(self, tensor_str: str) -> List[TensorInfo]:
        """Parse a list of tensor type declarations."""
        tensors = []
        
        # Find all tensor declarations
        for match in self.tensor_pattern.finditer(tensor_str):
            shape_str = match.group(1)
            elem_type = match.group(2)
            
            # Parse shape
            shape_parts = shape_str.rstrip('x').split('x')
            shape = []
            is_dynamic = False
            
            for part in shape_parts:
                if part == '?':
                    shape.append(-1)
                    is_dynamic = True
                elif part.isdigit():
                    shape.append(int(part))
            
            tensors.append(TensorInfo(
                shape=shape,
                element_type=elem_type,
                is_dynamic=is_dynamic
            ))
        
        return tensors
    
    def _extract_compute_operations(
        self, mlir_content: str, lines: List[str]
    ) -> List[MLIROperation]:
        """Extract compute-intensive operations from MLIR."""
        from iree_evo.config import MLIR_OP_CATEGORIES
        
        compute_ops = []
        compute_op_names = MLIR_OP_CATEGORIES.get("compute_intensive", [])
        
        for line_num, line in enumerate(lines, 1):
            # Check for compute-intensive operations
            for op_name in compute_op_names:
                if op_name in line:
                    op_info = self._parse_operation_line(line, line_num)
                    if op_info:
                        compute_ops.append(op_info)
                        break
        
        return compute_ops
    
    def _parse_operation_line(
        self, line: str, line_num: int
    ) -> Optional[MLIROperation]:
        """Parse a single MLIR operation line."""
        # Extract operation name
        op_match = re.search(r'(\w+\.[\w.]+)', line)
        if not op_match:
            return None
        
        op_name = op_match.group(1)
        
        # Extract tensor types
        input_tensors = self._parse_tensor_list(line)
        
        # For now, we'll create a simplified operation
        # In a full implementation, this would parse more details
        return MLIROperation(
            op_name=op_name,
            location=f"line:{line_num}",
            inputs=input_tensors[:len(input_tensors)//2] if input_tensors else [],
            outputs=input_tensors[len(input_tensors)//2:] if input_tensors else [],
            attributes={},
            line_number=line_num
        )
    
    def _count_operations(self, mlir_content: str) -> int:
        """Count total number of operations in MLIR."""
        # Simple heuristic: count lines with '='
        return len([line for line in mlir_content.split('\n') if '=' in line])
    
    def _analyze_dialects(self, mlir_content: str) -> Dict[str, int]:
        """Analyze which dialects are used and how frequently."""
        dialects = {}
        
        # Common IREE/MLIR dialects
        dialect_prefixes = [
            'linalg', 'arith', 'math', 'tensor', 'flow', 'hal', 
            'stream', 'func', 'scf', 'affine', 'vector', 'memref'
        ]
        
        for dialect in dialect_prefixes:
            pattern = rf'\b{dialect}\.\w+'
            count = len(re.findall(pattern, mlir_content))
            if count > 0:
                dialects[dialect] = count
        
        return dialects
    
    def to_json(self, summary: MLIRSummary) -> str:
        """Convert MLIRSummary to JSON string."""
        # Convert dataclasses to dicts
        summary_dict = {
            'entry_point': summary.entry_point,
            'input_signature': [asdict(t) for t in summary.input_signature],
            'output_signature': [asdict(t) for t in summary.output_signature],
            'compute_ops': [
                {
                    'op_name': op.op_name,
                    'location': op.location,
                    'inputs': [asdict(t) for t in op.inputs],
                    'outputs': [asdict(t) for t in op.outputs],
                    'attributes': op.attributes,
                    'line_number': op.line_number
                }
                for op in summary.compute_ops
            ],
            'total_ops': summary.total_ops,
            'has_dispatch_regions': summary.has_dispatch_regions,
            'dialect_usage': summary.dialect_usage
        }
        
        return json.dumps(summary_dict, indent=2)
    
    def to_concise_summary(self, summary: MLIRSummary) -> str:
        """Generate a concise human-readable summary for LLM consumption."""
        lines = []
        lines.append(f"Entry Point: {summary.entry_point}")
        lines.append(f"\nInput Signature:")
        for i, inp in enumerate(summary.input_signature):
            shape_str = "x".join(str(d) if d != -1 else "?" for d in inp.shape)
            lines.append(f"  arg{i}: tensor<{shape_str}x{inp.element_type}>")
        
        lines.append(f"\nOutput Signature:")
        for i, out in enumerate(summary.output_signature):
            shape_str = "x".join(str(d) if d != -1 else "?" for d in out.shape)
            lines.append(f"  result{i}: tensor<{shape_str}x{out.element_type}>")
        
        lines.append(f"\nCompute Operations ({len(summary.compute_ops)}):")
        for op in summary.compute_ops[:10]:  # Limit to first 10
            input_shapes = [
                f"{'x'.join(str(d) for d in t.shape)}x{t.element_type}"
                for t in op.inputs
            ]
            lines.append(f"  {op.op_name} @ {op.location}")
            if input_shapes:
                lines.append(f"    inputs: {', '.join(input_shapes)}")
        
        if len(summary.compute_ops) > 10:
            lines.append(f"  ... and {len(summary.compute_ops) - 10} more")
        
        lines.append(f"\nDialect Usage: {', '.join(summary.dialect_usage.keys())}")
        lines.append(f"Total Operations: {summary.total_ops}")
        lines.append(f"Has Dispatch Regions: {summary.has_dispatch_regions}")
        
        return "\n".join(lines)
