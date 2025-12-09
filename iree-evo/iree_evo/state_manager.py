#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""State Manager: Tracks evolutionary optimization state across generations."""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict, field
import time


@dataclass
class Variant:
    """Represents a compilation variant (genome in evolutionary terms)."""
    variant_id: str
    generation: int
    parent_id: Optional[str]
    compilation_flags: List[str]
    compilation_success: bool
    compilation_time: float = 0.0
    correctness_verified: bool = False
    benchmark_success: bool = False
    mean_latency_ms: float = float('inf')
    binary_size_bytes: int = 0
    fitness_score: float = 0.0
    vmfb_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class Generation:
    """Represents a generation in the evolutionary process."""
    generation_number: int
    variants: List[Variant]
    best_variant: Optional[Variant]
    created_at: float = field(default_factory=time.time)


class StateManager:
    """Manages state for the evolutionary optimization process."""
    
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        self.generations: List[Generation] = []
        self.all_variants: Dict[str, Variant] = {}
        self.best_overall: Optional[Variant] = None
        
        # State file
        self.state_file = work_dir / "state.json"
    
    def create_variant(
        self,
        generation: int,
        flags: List[str],
        parent_id: Optional[str] = None,
    ) -> Variant:
        """Create a new variant."""
        variant_id = f"gen{generation}_var{len(self.all_variants)}"
        
        variant = Variant(
            variant_id=variant_id,
            generation=generation,
            parent_id=parent_id,
            compilation_flags=flags,
            compilation_success=False,
        )
        
        self.all_variants[variant_id] = variant
        return variant
    
    def update_variant(
        self,
        variant_id: str,
        **updates,
    ):
        """Update a variant's attributes."""
        if variant_id in self.all_variants:
            variant = self.all_variants[variant_id]
            for key, value in updates.items():
                if hasattr(variant, key):
                    setattr(variant, key, value)
    
    def finish_generation(
        self,
        generation_number: int,
        variants: List[Variant],
    ) -> Generation:
        """Mark a generation as complete and identify the best variant."""
        # Find best variant in this generation
        successful_variants = [
            v for v in variants
            if v.compilation_success and v.correctness_verified
        ]
        
        if successful_variants:
            best = min(successful_variants, key=lambda v: v.mean_latency_ms)
        else:
            best = None
        
        generation = Generation(
            generation_number=generation_number,
            variants=variants,
            best_variant=best,
        )
        
        self.generations.append(generation)
        
        # Update best overall
        if best and (not self.best_overall or best.mean_latency_ms < self.best_overall.mean_latency_ms):
            self.best_overall = best
        
        return generation
    
    def get_top_variants(self, n: int = 3) -> List[Variant]:
        """Get the top N variants across all generations."""
        successful = [
            v for v in self.all_variants.values()
            if v.compilation_success and v.correctness_verified
        ]
        
        successful.sort(key=lambda v: v.mean_latency_ms)
        return successful[:n]
    
    def calculate_fitness(
        self,
        variant: Variant,
        latency_weight: float = 1.0,
        size_weight: float = 0.0,
        baseline_latency: float = 1.0,
    ) -> float:
        """Calculate fitness score for a variant.
        
        Fitness combines multiple objectives:
        - Latency (lower is better)
        - Binary size (smaller is better)
        
        Returns a score where higher is better.
        """
        if not variant.compilation_success or not variant.correctness_verified:
            return 0.0
        
        # Latency component (normalize by baseline, invert so higher is better)
        latency_score = baseline_latency / max(variant.mean_latency_ms, 0.001)
        
        # Size component (invert so smaller is better, but less important)
        size_mb = variant.binary_size_bytes / (1024 * 1024)
        size_score = 1.0 / max(size_mb, 0.1)
        
        # Combined fitness
        fitness = (latency_weight * latency_score + size_weight * size_score)
        
        return fitness
    
    def save_state(self):
        """Save current state to disk."""
        state_data = {
            'generations': [
                {
                    'generation_number': gen.generation_number,
                    'variants': [asdict(v) for v in gen.variants],
                    'best_variant': asdict(gen.best_variant) if gen.best_variant else None,
                    'created_at': gen.created_at,
                }
                for gen in self.generations
            ],
            'best_overall': asdict(self.best_overall) if self.best_overall else None,
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state_data, f, indent=2)
    
    def load_state(self) -> bool:
        """Load state from disk."""
        if not self.state_file.exists():
            return False
        
        try:
            with open(self.state_file, 'r') as f:
                state_data = json.load(f)
            
            # Reconstruct generations
            self.generations = []
            for gen_data in state_data['generations']:
                variants = [Variant(**v) for v in gen_data['variants']]
                best = Variant(**gen_data['best_variant']) if gen_data['best_variant'] else None
                
                generation = Generation(
                    generation_number=gen_data['generation_number'],
                    variants=variants,
                    best_variant=best,
                    created_at=gen_data['created_at'],
                )
                self.generations.append(generation)
                
                # Add to all_variants
                for variant in variants:
                    self.all_variants[variant.variant_id] = variant
            
            # Reconstruct best overall
            if state_data['best_overall']:
                self.best_overall = Variant(**state_data['best_overall'])
            
            return True
        
        except Exception as e:
            print(f"Error loading state: {e}")
            return False
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the optimization state."""
        lines = []
        lines.append("=" * 70)
        lines.append("IREE-Evo Optimization Summary")
        lines.append("=" * 70)
        
        lines.append(f"\nTotal Generations: {len(self.generations)}")
        lines.append(f"Total Variants Evaluated: {len(self.all_variants)}")
        
        if self.best_overall:
            lines.append(f"\n🏆 Best Variant: {self.best_overall.variant_id}")
            lines.append(f"   Mean Latency: {self.best_overall.mean_latency_ms:.3f} ms")
            lines.append(f"   Binary Size: {self.best_overall.binary_size_bytes / (1024*1024):.2f} MB")
            lines.append(f"   Fitness Score: {self.best_overall.fitness_score:.3f}")
            lines.append(f"   Generation: {self.best_overall.generation}")
        
        # Per-generation summary
        lines.append(f"\nPer-Generation Results:")
        for gen in self.generations:
            successful = sum(1 for v in gen.variants if v.compilation_success and v.correctness_verified)
            lines.append(f"\n  Generation {gen.generation_number}:")
            lines.append(f"    Variants: {len(gen.variants)}")
            lines.append(f"    Successful: {successful}")
            if gen.best_variant:
                lines.append(f"    Best Latency: {gen.best_variant.mean_latency_ms:.3f} ms")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)
