#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Planner Agent: High-level strategy selection using LLM."""

from typing import List, Dict, Any, Optional
import json

from iree_evo.perception.mlir_slicer import MLIRSummary
from iree_evo.agents.optimization_menu import OptimizationMenu, OptimizationStrategy


class PlannerAgent:
    """Plans high-level optimization strategies based on MLIR analysis.
    
    This agent analyzes the MLIR summary and hardware specifications to
    select appropriate optimization strategies. In the full implementation,
    this would use an LLM API (Gemini, GPT-4, etc.).
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.7,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.optimization_menu = OptimizationMenu()
    
    def plan(
        self,
        mlir_summary: MLIRSummary,
        hardware_profile: Dict[str, Any],
        backend: str,
    ) -> List[OptimizationStrategy]:
        """Generate an optimization plan for the given MLIR and hardware.
        
        Args:
            mlir_summary: Parsed MLIR summary
            hardware_profile: Hardware characteristics
            backend: Target backend (llvm-cpu, cuda, etc.)
            
        Returns:
            List of selected OptimizationStrategy objects
        """
        # PLACEHOLDER: In the full implementation, this would call an LLM API
        # For now, we use a heuristic-based approach
        
        # Extract compute operations
        op_names = [op.op_name for op in mlir_summary.compute_ops]
        
        # Get applicable strategies
        applicable_strategies = self.optimization_menu.get_applicable_strategies(
            op_names=op_names,
            backend=backend,
        )
        
        # Apply heuristics to select strategies
        selected_strategies = self._heuristic_selection(
            applicable_strategies,
            mlir_summary,
            hardware_profile,
            backend,
        )
        
        return selected_strategies
    
    def _heuristic_selection(
        self,
        applicable_strategies: List[OptimizationStrategy],
        mlir_summary: MLIRSummary,
        hardware_profile: Dict[str, Any],
        backend: str,
    ) -> List[OptimizationStrategy]:
        """Heuristic-based strategy selection (placeholder for LLM).
        
        This implements basic rules for strategy selection that would
        ideally be learned/reasoned by an LLM in the full system.
        """
        selected = []
        
        # Always start with baseline for comparison
        baseline = self.optimization_menu.get_strategy("baseline")
        if baseline:
            selected.append(baseline)
        
        # Check if we have matmul operations
        has_matmul = any(
            "matmul" in op.op_name.lower()
            for op in mlir_summary.compute_ops
        )
        
        # GPU-specific optimizations
        if backend in ["cuda", "rocm"]:
            # Enable tensor cores if available
            if has_matmul and hardware_profile.get("tensor_cores"):
                tensor_core = self.optimization_menu.get_strategy("tensor_core")
                if tensor_core:
                    selected.append(tensor_core)
            
            # GPU vectorization
            gpu_vec = self.optimization_menu.get_strategy("gpu_vectorize")
            if gpu_vec:
                selected.append(gpu_vec)
        
        # CPU-specific optimizations
        elif backend == "llvm-cpu":
            # Enable micro-kernels for CPU
            if has_matmul:
                ukernels = self.optimization_menu.get_strategy("enable_ukernels")
                if ukernels:
                    selected.append(ukernels)
        
        # General optimizations
        # Aggressive fusion for any backend
        fusion = self.optimization_menu.get_strategy("aggressive_fusion")
        if fusion and fusion not in selected:
            selected.append(fusion)
        
        # Check for quantization patterns
        has_quant = any(
            "quant" in dialect.lower()
            for dialect in mlir_summary.dialect_usage.keys()
        )
        if has_quant and has_matmul:
            dequant_fusion = self.optimization_menu.get_strategy("fuse_dequantization")
            if dequant_fusion:
                selected.append(dequant_fusion)
        
        return selected
    
    def _call_llm_api(
        self,
        prompt: str,
    ) -> Dict[str, Any]:
        """Call LLM API to get strategy recommendations.
        
        PLACEHOLDER: This is where you would integrate with:
        - Google Gemini API
        - OpenAI GPT-4 API
        - Anthropic Claude API
        - Or any other LLM provider
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Dictionary with LLM response
        """
        # TODO: Implement actual LLM API integration
        # Example structure:
        #
        # import google.generativeai as genai
        # genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        # model = genai.GenerativeModel(self.model_name)
        # response = model.generate_content(
        #     prompt,
        #     generation_config=genai.GenerationConfig(
        #         temperature=self.temperature,
        #         max_output_tokens=2000,
        #     )
        # )
        # return json.loads(response.text)
        
        raise NotImplementedError(
            "LLM API integration not yet implemented. "
            "Using heuristic-based planning as fallback."
        )
    
    def _build_planning_prompt(
        self,
        mlir_summary: MLIRSummary,
        hardware_profile: Dict[str, Any],
        backend: str,
    ) -> str:
        """Build a prompt for the LLM planner.
        
        This constructs a detailed prompt that includes:
        - MLIR summary
        - Hardware characteristics
        - Available optimization strategies
        - Request for strategy selection
        """
        from iree_evo.perception.mlir_slicer import MLIRSlicer
        
        slicer = MLIRSlicer()
        mlir_text = slicer.to_concise_summary(mlir_summary)
        
        prompt = f"""You are an expert compiler optimization planner for IREE.

Given the following MLIR program summary:
{mlir_text}

Target Backend: {backend}
Hardware Profile: {json.dumps(hardware_profile, indent=2)}

Available Optimization Strategies:
"""
        
        for strategy in self.optimization_menu.get_all_strategies():
            prompt += f"\n- {strategy.name}: {strategy.description}"
            prompt += f"\n  Expected speedup: {strategy.expected_speedup}"
            prompt += f"\n  Complexity: {strategy.complexity}"
        
        prompt += """

Select the most appropriate optimization strategies for this program and hardware.
Consider:
1. Operation types and their characteristics
2. Hardware capabilities (tensor cores, cache sizes, etc.)
3. Expected performance gains
4. Compatibility between strategies

Return your selection as a JSON array of strategy names, ordered by priority:
{"strategies": ["strategy1", "strategy2", ...]}
"""
        
        return prompt
