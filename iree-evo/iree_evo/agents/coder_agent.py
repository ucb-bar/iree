#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Coder Agent: Generates compilation flags and Transform Dialect scripts."""

from typing import List, Dict, Any, Optional
from pathlib import Path

from iree_evo.agents.optimization_menu import OptimizationStrategy


class CoderAgent:
    """Generates concrete compilation flags from optimization strategies.
    
    This agent takes high-level strategies and converts them into specific
    iree-compile flags or Transform Dialect scripts. In the full implementation,
    this could use an LLM to generate more sophisticated transformations.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.3,  # Lower temperature for code generation
    ):
        self.model_name = model_name
        self.temperature = temperature
    
    def generate_flags(
        self,
        strategies: List[OptimizationStrategy],
        backend: str,
    ) -> List[str]:
        """Generate compilation flags from optimization strategies.
        
        Args:
            strategies: List of selected optimization strategies
            backend: Target backend
            
        Returns:
            List of iree-compile flags
        """
        flags = []
        
        # Add backend flag
        flags.append(f"--iree-hal-target-backends={backend}")
        
        # Collect flags from all strategies
        seen_flag_names = set()
        
        for strategy in strategies:
            for flag in strategy.flags:
                # Extract flag name (before '=')
                flag_name = flag.split('=')[0]
                
                # Avoid duplicate flags
                if flag_name not in seen_flag_names:
                    flags.append(flag)
                    seen_flag_names.add(flag_name)
        
        return flags
    
    def generate_transform_dialect_script(
        self,
        strategies: List[OptimizationStrategy],
        output_path: Path,
        tile_sizes: Optional[List[int]] = None,
    ) -> Optional[Path]:
        """Generate a Transform Dialect script for advanced optimizations.
        
        Args:
            strategies: List of strategies that require Transform Dialect
            output_path: Where to save the script
            tile_sizes: Optional tile sizes for tiling transformations
            
        Returns:
            Path to generated script, or None if no script needed
        """
        # Check if any strategy requires Transform Dialect
        needs_transform = any(
            strategy.transform_dialect_required
            for strategy in strategies
        )
        
        if not needs_transform:
            return None
        
        # PLACEHOLDER: Generate a basic Transform Dialect script
        # In a full implementation, this would be more sophisticated
        # and potentially LLM-generated
        
        if tile_sizes is None:
            tile_sizes = [64, 64, 8]  # Default tile sizes
        
        script_content = self._create_basic_tiling_script(tile_sizes)
        
        # Write script
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(script_content)
        
        return output_path
    
    def _create_basic_tiling_script(self, tile_sizes: List[int]) -> str:
        """Create a basic Transform Dialect script for tiling.
        
        PLACEHOLDER: This is a simplified example. Real Transform Dialect
        scripts are more complex and operation-specific.
        """
        script = f"""// Auto-generated Transform Dialect Script
// Copyright 2024 The IREE Authors
// Licensed under the Apache License v2.0 with LLVM Exceptions.

module {{
  transform.sequence failures(propagate) {{
  ^bb0(%arg0: !transform.any_op):
    // Tile linalg operations
    %0 = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops:3 = transform.structured.tile %0 [{tile_sizes[0]}, {tile_sizes[1]}, {tile_sizes[2]}]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
  }}
}}
"""
        return script
    
    def mutate_flags(
        self,
        current_flags: List[str],
        mutation_type: str = "random",
    ) -> List[str]:
        """Mutate existing flags for evolutionary optimization.
        
        This implements the "evolutionary coding" aspect where we mutate
        compilation configurations to explore the optimization space.
        
        Args:
            current_flags: Current compilation flags
            mutation_type: Type of mutation (random, incremental, swap)
            
        Returns:
            Mutated flags
        """
        import random
        import copy
        
        mutated = copy.copy(current_flags)
        
        if mutation_type == "random":
            # Randomly add, remove, or modify a flag
            action = random.choice(["add", "remove", "modify"])
            
            if action == "add" and len(mutated) < 20:
                # Add a random flag from known good flags
                candidate_flags = [
                    "--iree-flow-enable-aggressive-fusion",
                    "--iree-llvmcpu-enable-ukernels=all",
                    "--iree-flow-enable-fuse-dequantization-matmul",
                    "--iree-codegen-gpu-native-math-precision=true",
                ]
                new_flag = random.choice(candidate_flags)
                if new_flag not in mutated:
                    mutated.append(new_flag)
            
            elif action == "remove" and len(mutated) > 2:
                # Remove a random flag (but keep backend flag)
                removable = [f for f in mutated if not f.startswith("--iree-hal-target-backends")]
                if removable:
                    mutated.remove(random.choice(removable))
            
            elif action == "modify":
                # Modify a numeric parameter in a flag
                for i, flag in enumerate(mutated):
                    if "=" in flag and any(char.isdigit() for char in flag):
                        # Try to tweak numeric values
                        parts = flag.split("=")
                        if len(parts) == 2:
                            # This is a simplified mutation
                            # In practice, you'd parse and intelligently mutate
                            pass
        
        return mutated
    
    def _call_llm_api(self, prompt: str) -> str:
        """Call LLM API for code generation.
        
        PLACEHOLDER: This would integrate with an LLM API to generate
        more sophisticated Transform Dialect scripts or flag combinations.
        """
        # TODO: Implement LLM API integration for code generation
        raise NotImplementedError(
            "LLM API integration for code generation not yet implemented."
        )
