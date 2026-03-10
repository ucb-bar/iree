# Concurrent Scheduling Example - Implementation Summary

## Overview
This implementation provides a complete example for comparing vanilla IREE scheduling with custom "oracle" scheduling for concurrent neural network execution.

## Components

### 1. Neural Network Models (MLIR)
- **model_a_conv.mlir**: Convolutional feature extractor (1x28x28x3 → 1x14x14x16)
  - Uses conv2d, ReLU activation, max pooling
  - Simulates MobileNet-style feature extraction
  
- **model_b_dense.mlir**: Dense classifier (1x14x14x16 → 1x10)
  - Flattens input, applies 2 dense layers with ReLU
  - Simulates classifier head
  
- **model_c_residual.mlir**: Residual bottleneck (1x32x32x8 → 1x32x32x8)
  - 1x1 conv (bottleneck) → 3x3 conv → 1x1 conv (expand) + residual
  - Simulates ResNet-style processing

### 2. Pipeline Orchestrators

#### Vanilla Async Pipeline (MLIR)
- **pipeline_vanilla_async.mlir**: IREE automatic scheduling
  - Uses `--iree-execution-model=async-external`
  - Declares external functions with `iree.abi.model = "coarse-fences"`
  - Two functions:
    - `pipeline_vanilla`: Single iteration
    - `pipeline_vanilla_multi_freq`: Multiple iterations with frequency control

#### Oracle Scheduler (C)
- **concurrent_scheduling_oracle.c**: Manual hardware-aware scheduling
  - Explicit device placement (CPU cluster 0/1, NPU)
  - Manual synchronization between models
  - Per-model timing statistics
  - Configuration for SpaceMIT X60 hardware

#### Vanilla Scheduler (C)
- **concurrent_scheduling_vanilla.c**: Wrapper for MLIR pipeline
  - Lets IREE handle all scheduling
  - Minimal code, maximum automation
  - Comparison baseline for oracle scheduler

### 3. Build System
- **CMakeLists.txt**: CMake configuration
  - Compiles MLIR to VMFB modules
  - Builds C executables
  - Adds lit tests
  
- **BUILD.bazel**: Bazel configuration
  - Mirror of CMake functionality
  - Uses IREE build rules

### 4. Helper Scripts
- **compile_models.sh**: Compiles all MLIR models to VMFB
- **test_example.sh**: Validates setup and provides instructions
- **.gitignore**: Excludes build artifacts

### 5. Documentation
- **README.md**: Comprehensive user guide
  - Architecture comparison
  - Build instructions
  - Usage examples
  - Performance analysis guidance

## Design Decisions

### Model Selection
- **Different operators**: Conv2d, dense (matmul), residual connections
- **Different sizes**: Varied compute requirements
- **Well-known patterns**: Based on MobileNet and ResNet architectures
- **Realistic shapes**: Typical ML workload dimensions

### Dependency Structure
```
A (Conv) → B (Dense)    [Dependent pipeline, high frequency]
C (Residual)            [Independent, lower frequency]
```

### Hardware Mapping (Oracle)
- Model A (Conv): Cluster 1 with NPU (best for convolutions)
- Model B (Dense): Cluster 0 (CPU-intensive matmul)
- Model C (Residual): Cluster 1 (can run concurrently with A's completion)

### Frequency Control
- A+B: Every iteration (simulates real-time inference)
- C: Every 2 iterations (simulates periodic background task)

## Key Features

### Vanilla Approach
✅ Fully automatic - no manual intervention
✅ Portable across hardware
✅ Uses IREE's coarse-fences for async
✅ Minimal code complexity

### Oracle Approach
✅ Hardware-aware placement
✅ Explicit synchronization control
✅ Per-model performance tracking
✅ Configurable scheduling strategy
⚠️ Hardware-specific (less portable)
⚠️ More complex implementation

## Testing Strategy

### Validation
1. ✅ MLIR syntax validation (builtin.module, func.func)
2. ✅ C code API usage (matches existing samples)
3. ✅ Build system integration (CMakeLists.txt, BUILD.bazel)
4. ⏳ Compilation test (requires full IREE build)
5. ⏳ Runtime test (requires IREE runtime)

### Expected Outcomes
- Both approaches should produce valid results
- Oracle may show better throughput on target hardware
- Vanilla provides baseline for comparison
- Performance difference depends on hardware topology

## Integration with IREE

### File Locations
```
samples/
├── CMakeLists.txt              [UPDATED: added concurrent_scheduling]
└── concurrent_scheduling/
    ├── BUILD.bazel
    ├── CMakeLists.txt
    ├── README.md
    ├── .gitignore
    ├── test_example.sh
    ├── compile_models.sh
    ├── model_a_conv.mlir
    ├── model_b_dense.mlir
    ├── model_c_residual.mlir
    ├── pipeline_vanilla_async.mlir
    ├── concurrent_scheduling_oracle.c
    └── concurrent_scheduling_vanilla.c
```

### Build Integration
- Added to `samples/CMakeLists.txt` via `add_subdirectory(concurrent_scheduling)`
- Follows existing patterns from `samples/multiple_modules/` and `samples/custom_module/async/`

## Next Steps (User Actions)

1. Build IREE from source
2. Run `compile_models.sh` to generate VMFB files
3. Build C executables
4. Run both schedulers with different iteration counts
5. Compare performance metrics
6. Optionally: Adapt for actual SpaceMIT X60 hardware

## Extensibility

### Adding Models
1. Create new MLIR file
2. Add to CMakeLists.txt/BUILD.bazel
3. Update pipeline orchestrators
4. Update C runners

### Changing Frequencies
- MLIR: Modify `scf.for` loop conditions
- C: Change `if (iter % N == 0)` logic

### Hardware Tuning
- Update `target_cluster` in oracle config
- Modify compilation flags for specific CPU/NPU targets
- Add CPU affinity calls for cluster pinning

## References
- Based on: `samples/multiple_modules/pipeline_async.mlir`
- Runtime pattern: `samples/custom_module/async/main.c`
- Build pattern: Existing sample CMakeLists.txt files
