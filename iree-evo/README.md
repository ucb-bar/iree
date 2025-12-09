# IREE-Evo: Autonomous Agentic Compiler Optimization Framework

IREE-Evo is a deeply integrated, autonomous agentic flow for the IREE compiler that uses evolutionary strategies and AI planning to automatically optimize MLIR code for heterogeneous hardware targets.

## 🎯 Vision

Unlike surface-level flag tuners, IREE-Evo has deep semantic understanding of the compiler stack. It autonomously:
- Analyzes MLIR IR to understand computation patterns
- Plans optimization strategies based on hardware characteristics
- Generates compilation flags and Transform Dialect scripts
- Verifies correctness through automatic test generation
- Optimizes for end-to-end latency (and optionally binary size)
- Evolves solutions across multiple generations

## 🧬 Architecture

IREE-Evo implements a sophisticated evolutionary optimization loop:

```
Input MLIR → Parse → Establish Baseline → [Evolutionary Loop] → Optimized Binary
                                               ↓
                    ┌──────────────────────────┴────────────────────────┐
                    ↓                                                    ↓
              Generate Variants                                   Evaluate Variants
            (Plan → Code → Compile)                          (Compile → Verify → Benchmark)
                    ↑                                                    ↓
                    └────────────────────── Select Best ─────────────────┘
```

### Core Components

1. **Perception Module (MLIR Slicer)**
   - Parses MLIR to extract critical components
   - Identifies compute-intensive operations
   - Analyzes tensor shapes and dialects
   - Cleans compilation error logs

2. **Knowledge Base (Anti-Hallucination Layer)**
   - Catalogs valid optimization strategies
   - Maps strategies to IREE compilation flags
   - Validates flag combinations

3. **Planner Agent**
   - Selects high-level optimization strategies
   - Considers hardware characteristics
   - Uses heuristics (with LLM integration placeholder)

4. **Coder Agent**
   - Generates concrete compilation flags
   - Creates Transform Dialect scripts
   - Implements evolutionary mutations

5. **Verification Harness**
   - Generates correctness tests automatically
   - Compares optimized vs baseline outputs
   - Uses numpy.allclose for numerical comparison

6. **Execution Engine**
   - Safe subprocess-based compilation
   - Benchmarking with metric extraction
   - Timeout handling

7. **Orchestrator**
   - Manages the evolutionary loop
   - Coordinates all components
   - Tracks state across generations

## 🚀 Quick Start

### Installation

```bash
cd iree-evo
pip install -e .
```

### Prerequisites

IREE-Evo requires the IREE compiler and runtime tools:

```bash
# Install IREE (example using pip)
pip install iree-compiler iree-runtime
```

### Basic Usage

Optimize a matmul for CPU:

```bash
iree-evo --input examples/matmul.mlir --backend llvm-cpu
```

Optimize for NVIDIA GPU:

```bash
iree-evo --input examples/matmul.mlir --backend cuda --device cuda --generations 10
```

Quick test with small population:

```bash
iree-evo --input examples/matmul.mlir --population 5 --generations 3
```

### Command-Line Options

```
Required:
  --input PATH              Input MLIR file to optimize

Target Configuration:
  --backend TYPE            Target backend (llvm-cpu, cuda, rocm, vulkan, metal)
  --device NAME             Target device (default: local-task)

Evolutionary Parameters:
  --generations N           Number of generations (default: 5)
  --population N            Population size per generation (default: 10)
  --top-k N                 Top variants to keep for breeding (default: 3)

Optimization Objectives:
  --optimize-latency        Optimize for latency (default: true)
  --optimize-size           Also optimize for binary size
  --latency-weight FLOAT    Weight for latency in fitness (default: 1.0)
  --size-weight FLOAT       Weight for size in fitness (default: 0.0)

Paths:
  --work-dir PATH           Working directory (default: /tmp/iree_evo_work)
  --iree-compile PATH       Path to iree-compile
  --iree-benchmark PATH     Path to iree-benchmark-module

Output:
  --verbose                 Verbose output (default: true)
  --quiet                   Minimal output
```

## 📚 Research Foundation

IREE-Evo synthesizes state-of-the-art methodologies:

1. **AutoComp/COMPASS (Plan-then-Implement)**
   - Two-step process: strategy selection → implementation
   - Prevents blind flag guessing

2. **AlphaEvolve (Evolutionary Coding)**
   - Treats compilation config as a genome
   - Iterative mutation based on feedback

3. **MLGO (RL Loop)**
   - Closes the loop: State → Action → Reward → Update
   - State: MLIR, Action: Compile, Reward: Latency/Size

## 🔧 Optimization Strategies

IREE-Evo includes predefined strategies:

- **enable_ukernels**: CPU micro-kernels for matmul/conv
- **aggressive_fusion**: Operation fusion to reduce memory traffic
- **tensor_core**: NVIDIA Tensor Core optimizations
- **gpu_vectorize**: GPU vectorization
- **software_pipelining**: GPU software pipelining
- **amd_wmma**: AMD Wave Matrix operations
- **cpu_data_tiling**: Cache-aware tiling
- **fuse_dequantization**: Quantized model optimizations
- **pad_operations**: Tensor padding for alignment

## 📁 Project Structure

```
iree-evo/
├── iree_evo/
│   ├── __init__.py                 # Package initialization
│   ├── config.py                   # Configuration and constants
│   ├── orchestrator.py             # Main optimization loop
│   ├── state_manager.py            # Evolutionary state tracking
│   ├── cli.py                      # Command-line interface
│   ├── perception/                 # MLIR parsing and analysis
│   │   ├── mlir_slicer.py         # MLIR parser
│   │   └── error_log_cleaner.py   # Error extraction
│   ├── verification/               # Correctness testing
│   │   ├── test_generator.py      # Test script generation
│   │   └── baseline_manager.py    # Baseline management
│   ├── execution/                  # Compilation and benchmarking
│   │   ├── compiler_wrapper.py    # Safe compilation
│   │   └── benchmark_runner.py    # Benchmarking
│   └── agents/                     # AI agents
│       ├── optimization_menu.py   # Strategy catalog
│       ├── planner_agent.py       # Strategy selection
│       └── coder_agent.py         # Flag generation
├── examples/                       # Example MLIR files
├── tests/                         # Unit and integration tests
├── pyproject.toml                 # Package configuration
├── setup.py                       # Setup script
└── README.md                      # This file
```

## 🔬 Example Workflow

1. **Input**: User provides `matmul.mlir` targeting NVIDIA A100

2. **Parse**: MLIR Slicer extracts:
   - Operation: `linalg.matmul ins(1024x1024, 1024x1024)`
   - Entry point: `@main`
   - Input/output signatures

3. **Baseline**: Compile with minimal flags, benchmark → 12ms

4. **Plan**: Planner identifies:
   - Hardware has Tensor Cores
   - Strategy: "Tile for Tensor Cores, enable vectorization"

5. **Generate**: Coder produces flags:
   ```
   --iree-hal-target-backends=cuda
   --iree-codegen-llvmgpu-enable-transform-dialect-jit
   --iree-codegen-gpu-native-math-precision=true
   ```

6. **Verify**: Test generator creates Python script:
   - Generates random 1024x1024 matrices
   - Runs baseline and candidate
   - Compares with `np.allclose`

7. **Benchmark**: iree-benchmark-module → 5ms (2.4x speedup!)

8. **Evolve**: Mutate flags, try pipeline depth variations...

9. **Result**: Best variant achieves 4.2ms (2.9x speedup)

## 🧪 Testing

Run unit tests:

```bash
cd iree-evo
pytest tests/
```

Integration test with example:

```bash
iree-evo --input examples/matmul.mlir --population 3 --generations 2
```

## 🛠️ Development Status

Current implementation status:

- ✅ Core framework and architecture
- ✅ MLIR parsing and analysis
- ✅ Compilation and benchmarking infrastructure
- ✅ Test generation and verification
- ✅ Evolutionary loop with mutation/selection
- ✅ Predefined optimization strategies
- ✅ CLI interface
- 🚧 LLM API integration (placeholder implemented)
- 🚧 Transform Dialect script generation (basic template)
- 🚧 Advanced knowledge base scraping
- 📋 Comprehensive test suite
- 📋 Real-world benchmarks

## 🤝 Contributing

IREE-Evo is part of the IREE project. Contributions are welcome!

Areas for contribution:
- Additional optimization strategies
- LLM API integrations (Gemini, GPT-4, Claude)
- Transform Dialect script generation
- Hardware-specific profiles
- Test coverage
- Documentation

## 📄 License

Copyright 2024 The IREE Authors

Licensed under the Apache License v2.0 with LLVM Exceptions.
See https://llvm.org/LICENSE.txt for license information.

## 🙏 Acknowledgments

IREE-Evo is inspired by:
- AutoComp/COMPASS research on plan-then-implement compilation
- AlphaEvolve's evolutionary coding approaches
- MLGO's reinforcement learning for compiler optimization
- The IREE compiler team's excellent infrastructure

---

**Ready to start optimizing!** 🚀
