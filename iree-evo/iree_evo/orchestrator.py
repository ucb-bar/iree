#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Orchestrator: Main evolutionary optimization loop for IREE-Evo."""

import subprocess
import random
from typing import List, Optional
from pathlib import Path

from iree_evo.config import IREEEvoConfig, HARDWARE_PROFILES
from iree_evo.perception import MLIRSlicer, ErrorLogCleaner
from iree_evo.verification import TestGenerator, BaselineManager, BaselineResult
from iree_evo.execution import CompilerWrapper, BenchmarkRunner
from iree_evo.agents import OptimizationMenu, PlannerAgent, CoderAgent
from iree_evo.state_manager import StateManager, Variant


class Orchestrator:
    """Main orchestrator for the IREE-Evo evolutionary optimization system."""
    
    def __init__(self, config: IREEEvoConfig):
        self.config = config
        
        # Initialize components
        self.mlir_slicer = MLIRSlicer()
        self.error_cleaner = ErrorLogCleaner()
        self.test_generator = TestGenerator(config.target_device)
        self.baseline_manager = BaselineManager(
            config.iree_compile_path,
            config.iree_benchmark_path,
            config.target_backend,
            config.target_device,
        )
        self.compiler_wrapper = CompilerWrapper(config.iree_compile_path)
        self.benchmark_runner = BenchmarkRunner(
            config.iree_benchmark_path,
            config.target_device,
        )
        
        # Initialize agents
        self.optimization_menu = OptimizationMenu()
        self.planner_agent = PlannerAgent(
            config.planner_model,
            config.llm_temperature,
        )
        self.coder_agent = CoderAgent(
            config.coder_model,
            config.llm_temperature,
        )
        
        # State management
        self.state_manager = StateManager(config.work_dir)
        
        # Runtime state
        self.baseline: Optional[BaselineResult] = None
        self.mlir_file: Optional[Path] = None
    
    def optimize(self, mlir_file: Path) -> Variant:
        """Run the full evolutionary optimization loop.
        
        Args:
            mlir_file: Path to input MLIR file
            
        Returns:
            Best variant found
        """
        self.mlir_file = mlir_file
        
        if self.config.verbose:
            print("=" * 70)
            print("IREE-Evo: Autonomous Compiler Optimization")
            print("=" * 70)
            print(f"\nInput: {mlir_file}")
            print(f"Backend: {self.config.target_backend}")
            print(f"Device: {self.config.target_device}")
            print(f"Generations: {self.config.num_generations}")
            print(f"Population: {self.config.population_size}")
        
        # Step 1: Parse MLIR
        if self.config.verbose:
            print("\n📊 Step 1: Parsing MLIR...")
        mlir_summary = self.mlir_slicer.parse_file(mlir_file)
        
        if self.config.verbose:
            print(self.mlir_slicer.to_concise_summary(mlir_summary))
        
        # Step 2: Establish baseline
        if self.config.verbose:
            print("\n📏 Step 2: Establishing baseline...")
        
        baseline_dir = self.config.work_dir / "baseline"
        self.baseline = self.baseline_manager.establish_baseline(
            mlir_file,
            baseline_dir,
            timeout=self.config.compile_timeout,
            benchmark=True,
        )
        
        if not self.baseline.compilation_success:
            raise RuntimeError(
                f"Baseline compilation failed: {self.baseline.error_message}"
            )
        
        if self.config.verbose:
            print(f"  ✓ Baseline compiled: {self.baseline.vmfb_file}")
            if self.baseline.benchmark_success:
                print(f"  ✓ Baseline latency: {self.baseline.mean_latency_ms:.3f} ms")
                print(f"  ✓ Binary size: {self.baseline.binary_size_bytes / (1024*1024):.2f} MB")
        
        # Step 3: Run evolutionary loop
        if self.config.verbose:
            print(f"\n🧬 Step 3: Running evolutionary optimization...")
        
        for gen in range(self.config.num_generations):
            if self.config.verbose:
                print(f"\n{'='*70}")
                print(f"Generation {gen}")
                print(f"{'='*70}")
            
            self._run_generation(gen, mlir_summary, mlir_file)
        
        # Step 4: Report results
        if self.config.verbose:
            print("\n" + self.state_manager.get_summary())
        
        # Save final state
        self.state_manager.save_state()
        
        return self.state_manager.best_overall
    
    def _run_generation(
        self,
        generation: int,
        mlir_summary,
        mlir_file: Path,
    ):
        """Run a single generation of the evolutionary process."""
        variants = []
        
        # Generation 0: Create initial population
        if generation == 0:
            variants = self._create_initial_population(mlir_summary)
        else:
            # Subsequent generations: Mutate from top performers
            variants = self._create_mutated_population(generation)
        
        # Evaluate all variants
        for i, variant in enumerate(variants):
            if self.config.verbose:
                print(f"\n  Variant {i+1}/{len(variants)}: {variant.variant_id}")
            
            self._evaluate_variant(variant, mlir_file, mlir_summary)
        
        # Finish generation
        self.state_manager.finish_generation(generation, variants)
    
    def _create_initial_population(self, mlir_summary) -> List[Variant]:
        """Create the initial population (Generation 0)."""
        variants = []
        
        # Get hardware profile
        hardware_profile = HARDWARE_PROFILES.get(
            "cpu_avx512" if self.config.target_backend == "llvm-cpu" else "nvidia_a100",
            {}
        )
        
        # Use planner to get strategies
        strategies = self.planner_agent.plan(
            mlir_summary,
            hardware_profile,
            self.config.target_backend,
        )
        
        if self.config.verbose:
            print(f"\n  Planner selected {len(strategies)} strategies:")
            for s in strategies:
                print(f"    - {s.name}: {s.description}")
        
        # Create variants from strategies
        for i, strategy in enumerate(strategies[:self.config.population_size]):
            flags = self.coder_agent.generate_flags(
                [strategy],
                self.config.target_backend,
            )
            
            variant = self.state_manager.create_variant(
                generation=0,
                flags=flags,
                parent_id=None,
            )
            variants.append(variant)
        
        # Fill remaining population with random combinations
        while len(variants) < self.config.population_size:
            # Combine 2-3 random strategies
            num_strategies = random.randint(2, min(3, len(strategies)))
            selected = random.sample(strategies, num_strategies)
            
            flags = self.coder_agent.generate_flags(
                selected,
                self.config.target_backend,
            )
            
            variant = self.state_manager.create_variant(
                generation=0,
                flags=flags,
                parent_id=None,
            )
            variants.append(variant)
        
        return variants
    
    def _create_mutated_population(self, generation: int) -> List[Variant]:
        """Create a new population by mutating top performers."""
        variants = []
        
        # Get top performers from previous generations
        top_variants = self.state_manager.get_top_variants(
            n=self.config.selection_top_k
        )
        
        if not top_variants:
            # Fallback: create random variants
            return self._create_initial_population(
                self.mlir_slicer.parse_file(self.mlir_file)
            )
        
        # Create new variants by mutating top performers
        for i in range(self.config.population_size):
            # Select a parent
            parent = random.choice(top_variants)
            
            # Mutate flags
            mutated_flags = self.coder_agent.mutate_flags(
                parent.compilation_flags,
                mutation_type="random",
            )
            
            variant = self.state_manager.create_variant(
                generation=generation,
                flags=mutated_flags,
                parent_id=parent.variant_id,
            )
            variants.append(variant)
        
        return variants
    
    def _evaluate_variant(
        self,
        variant: Variant,
        mlir_file: Path,
        mlir_summary,
    ):
        """Evaluate a single variant through compilation, verification, and benchmarking."""
        # Create output directory for this variant
        variant_dir = self.config.work_dir / variant.variant_id
        variant_dir.mkdir(parents=True, exist_ok=True)
        
        vmfb_file = variant_dir / f"{mlir_file.stem}.vmfb"
        
        # Step 1: Compile
        if self.config.verbose:
            print(f"    Compiling with {len(variant.compilation_flags)} flags...")
        
        compile_result = self.compiler_wrapper.compile(
            mlir_file,
            vmfb_file,
            variant.compilation_flags,
            timeout=self.config.compile_timeout,
        )
        
        self.state_manager.update_variant(
            variant.variant_id,
            compilation_success=compile_result.success,
            compilation_time=compile_result.compilation_time,
            vmfb_path=str(vmfb_file) if compile_result.success else None,
        )
        
        if not compile_result.success:
            if self.config.verbose:
                print(f"    ✗ Compilation failed")
                # Clean error message
                error_info = self.error_cleaner.clean_stderr(compile_result.stderr)
                print(f"      Error: {error_info.error_message[:100]}...")
            
            self.state_manager.update_variant(
                variant.variant_id,
                error_message=compile_result.stderr[:500],
            )
            return
        
        if self.config.verbose:
            print(f"    ✓ Compilation succeeded ({compile_result.compilation_time:.2f}s)")
        
        # Step 2: Verify correctness
        if self.config.verbose:
            print(f"    Verifying correctness...")
        
        test_script = variant_dir / "verify_test.py"
        self.test_generator.generate_test_script(
            mlir_summary,
            Path(self.baseline.vmfb_file),
            vmfb_file,
            test_script,
            rtol=self.config.correctness_rtol,
            atol=self.config.correctness_atol,
        )
        
        # Run test
        try:
            test_result = subprocess.run(
                ["python3", str(test_script)],
                capture_output=True,
                text=True,
                timeout=self.config.test_timeout,
            )
            
            correctness_verified = (test_result.returncode == 0)
        except subprocess.TimeoutExpired:
            correctness_verified = False
        except Exception:
            correctness_verified = False
        
        self.state_manager.update_variant(
            variant.variant_id,
            correctness_verified=correctness_verified,
        )
        
        if not correctness_verified:
            if self.config.verbose:
                print(f"    ✗ Correctness verification failed")
            return
        
        if self.config.verbose:
            print(f"    ✓ Correctness verified")
        
        # Step 3: Benchmark
        if self.config.verbose:
            print(f"    Benchmarking...")
        
        bench_result = self.benchmark_runner.benchmark(
            vmfb_file,
            timeout=self.config.benchmark_timeout,
        )
        
        self.state_manager.update_variant(
            variant.variant_id,
            benchmark_success=bench_result.success,
            mean_latency_ms=bench_result.mean_latency_ms,
            binary_size_bytes=bench_result.binary_size_bytes,
        )
        
        if bench_result.success:
            # Calculate fitness
            fitness = self.state_manager.calculate_fitness(
                variant,
                latency_weight=self.config.latency_weight,
                size_weight=self.config.size_weight,
                baseline_latency=self.baseline.mean_latency_ms,
            )
            
            self.state_manager.update_variant(
                variant.variant_id,
                fitness_score=fitness,
            )
            
            if self.config.verbose:
                speedup = self.baseline.mean_latency_ms / bench_result.mean_latency_ms
                print(f"    ✓ Benchmark: {bench_result.mean_latency_ms:.3f} ms ({speedup:.2f}x speedup)")
                print(f"    ✓ Fitness: {fitness:.3f}")
        else:
            if self.config.verbose:
                print(f"    ✗ Benchmark failed")
