# IREE Scheduling Investigation Summary

**Date**: December 2, 2024
**Issue**: Understanding IREE's scheduling architecture and implementing custom Flexible Job Shop Scheduling

## Investigation Results

### What We Discovered

IREE implements a sophisticated **multi-layer scheduling system** that operates at three distinct levels:

#### 1. Compiler-Level Scheduling (Ahead-of-Time)
- **Location**: `compiler/src/iree/compiler/Dialect/Stream/Transforms/`
- **Key Passes**:
  - `ScheduleExecution.cpp` - Partitions operations into executable regions
  - `ScheduleAllocation.cpp` - Schedules memory allocations/deallocations
  - `ScheduleConcurrency.cpp` - Optimizes concurrent execution opportunities

#### 2. HAL Layer (Device Abstraction)
- **Location**: `runtime/src/iree/hal/`
- **Components**:
  - Command buffers record work
  - Device queues submit work
  - Semaphores/fences provide synchronization
  - Drivers implement device-specific scheduling

#### 3. Runtime Task System (Fine-Grained Execution)
- **Location**: `runtime/src/iree/task/`
- **Architecture**: Wavefront-style work-stealing scheduler
- **Components**:
  - Task executor coordinates scheduling
  - Workers (thread pool) execute tasks
  - Topology maps workers to CPU cores
  - Affinity sets control task-to-core assignment

### Current Capabilities

✅ **IREE Already Has**:
- Sophisticated task scheduling with work-stealing
- Topology-aware worker placement
- Affinity control for task-to-core mapping
- Concurrent execution of multiple models via timelines
- Pipelined execution with fine-grained dependencies
- Stream-ordered memory allocation

❌ **IREE Does NOT Have**:
- Built-in job shop scheduling algorithm
- Explicit resource reservation (e.g., for NPU)
- Priority-based preemptive scheduling
- Reactive scheduling that adapts to environment
- Cross-model orchestration with custom policies

### Your Specific Use Case

**Hardware Configuration**:
- Cluster 0: 4 general-purpose cores (cores 0-3)
- Cluster 1: 4 cores with NPU RISC-V extension (cores 4-7)
- Requirement: Schedule multiple concurrent MLIR models with job shop constraints
- Target: Real-time robotics with reactive scheduling

**Key Questions Answered**:

1. **Q: How does IREE schedule multiple MLIR files executing concurrently?**
   - **A**: Via timeline-based execution using HAL semaphores. Each model can execute on independent timelines, or use fork-join patterns to coordinate. The runtime task system handles fine-grained parallelism within each model.

2. **Q: How does IREE assign dispatches to CPU cores?**
   - **A**: Workers are mapped to CPU cores via `iree_task_topology_t`. Tasks specify affinity via `iree_task_affinity_set_t` (64-bit bitmask). The coordinator posts tasks to workers matching the affinity, and workers can steal work from others.

3. **Q: Can IREE handle heterogeneous clusters (one with NPU)?**
   - **A**: Yes, through custom topology and affinity control. However, explicit resource management (exclusive NPU access) requires custom implementation.

4. **Q: Is IREE already doing Flexible Job Shop Scheduling?**
   - **A**: No. IREE has work-stealing and topology awareness, but not job shop scheduling with precedence constraints, resource reservation, and deadline handling. You need to implement this.

## Implementation Recommendations

### Recommended Approach: Custom HAL Driver ⭐

**Why**: Clean separation, full control, maintainable, testable

**Steps**:
1. Fork `runtime/src/iree/hal/drivers/local_task/` → `job_shop/`
2. Add job shop scheduler to device structure
3. Implement scheduling algorithm in `queue_execute()`
4. Add NPU resource manager with exclusive access
5. Expose scheduling metrics for monitoring

**Estimated Effort**: 2-3 weeks for experienced C developer

### Key Components to Implement

#### 1. Job Shop Scheduler
```c
typedef struct iree_job_shop_scheduler_t {
  iree_slim_mutex_t queue_mutex;
  iree_scheduler_job_t* ready_queue;    // Priority-ordered
  iree_scheduler_job_t* pending_queue;  // Waiting on dependencies
  iree_cluster_state_t clusters[2];     // Track cluster state
  iree_npu_manager_t* npu_manager;      // Exclusive NPU access
  scheduling_policy_t policy;           // FIFO/Priority/Deadline/Shortest
} iree_job_shop_scheduler_t;
```

#### 2. NPU Resource Manager
```c
typedef struct iree_npu_manager_t {
  iree_task_affinity_set_t npu_core_mask;  // Cores 4-7
  iree_atomic_int32_t npu_in_use;          // Exclusive access
  iree_scheduler_job_t* npu_queue;         // Pending NPU jobs
} iree_npu_manager_t;
```

#### 3. Job Metadata
```c
typedef struct iree_job_metadata_t {
  uint32_t job_id;              // Job identifier
  uint32_t operation_id;        // Operation sequence
  uint32_t priority;            // 0-255
  uint64_t deadline_ns;         // Absolute deadline
  bool requires_npu;            // Needs NPU access
  uint32_t estimated_duration;  // For scheduling decisions
} iree_job_metadata_t;
```

#### 4. Reactive Scheduling
```c
void update_scheduler_telemetry(
    iree_job_shop_scheduler_t* scheduler,
    uint32_t cluster_id,
    uint32_t temperature,
    uint32_t memory_pressure) {
  
  // Thermal throttling
  if (temperature > 90) {
    scheduler->clusters[cluster_id].max_concurrent_jobs = 2;
  } else if (temperature < 70) {
    scheduler->clusters[cluster_id].max_concurrent_jobs = 4;
  }
  
  // Trigger rescheduling
  reschedule(scheduler);
}
```

### Extension Points in IREE

1. **Topology Definition**: Define custom topology matching your hardware
   ```c
   iree_task_topology_t topology;
   // Configure clusters 0 and 1 with appropriate masks
   ```

2. **Affinity Control**: Set task affinity at compile or runtime
   ```c
   task->affinity_set = 0b11110000;  // Cores 4-7 (NPU cluster)
   ```

3. **HAL Driver**: Implement `iree_hal_device_t` interface
   ```c
   static iree_status_t job_shop_device_queue_execute(...) {
     // Extract metadata
     // Run scheduling algorithm
     // Assign to appropriate cluster
     // Submit to task executor
   }
   ```

4. **Compiler Annotations**: Add custom attributes to IR (optional)
   ```mlir
   stream.cmd.dispatch @kernel[%x, %y, %z]
       attributes {
         iree.dispatch.requires_npu = true,
         iree.dispatch.priority = 200
       }
   ```

## Documentation Created

We created three comprehensive documents:

### 1. [iree-scheduling-deep-dive.md](./iree-scheduling-deep-dive.md)
**856 lines** - Complete analysis of IREE's scheduling architecture
- Multi-layer scheduling overview
- Compiler scheduling passes
- Runtime task system internals
- CPU core allocation mechanisms
- Concurrent execution patterns
- Current capabilities vs. what's missing
- Key files for implementation

### 2. [custom-scheduler-implementation-guide.md](./custom-scheduler-implementation-guide.md)
**900 lines** - Step-by-step implementation guide
- Complete code examples for custom HAL driver
- Job shop scheduler data structures
- NPU resource manager implementation
- Reactive scheduling for robotics
- Build configuration (CMake/Bazel)
- Testing strategies
- Usage examples

### 3. [SCHEDULING_README.md](./SCHEDULING_README.md)
**94 lines** - Index and quick start guide
- Document overview
- Quick start instructions
- Use cases and key concepts
- Links to related documentation

## Key Files Identified

### Runtime (Task System)
```
runtime/src/iree/task/
├── executor.h/c          - Main task executor (coordinator)
├── worker.h/c            - Worker thread implementation
├── task.h/c              - Task types and lifecycle
├── topology.h/c          - CPU topology detection
├── affinity_set.h        - Affinity bit manipulation
└── submission.h/c        - Task submission API
```

### HAL (Device Abstraction)
```
runtime/src/iree/hal/
├── device.h/c            - Device interface
└── drivers/
    ├── local_task/       - Task-based HAL device (YOUR STARTING POINT)
    └── local_sync/       - Synchronous device (reference)
```

### Compiler (Scheduling)
```
compiler/src/iree/compiler/Dialect/Stream/Transforms/
├── ScheduleExecution.cpp     - Main execution scheduling
├── ScheduleAllocation.cpp    - Memory scheduling
└── ScheduleConcurrency.cpp   - Concurrency optimization
```

## Next Steps for Implementation

1. **Phase 1: Prototype** (Week 1)
   - Define custom topology for your 2-cluster hardware
   - Test basic affinity control
   - Verify worker-to-core mapping

2. **Phase 2: Core Scheduler** (Week 2-3)
   - Fork `local_task` driver → `job_shop` driver
   - Implement job shop scheduler
   - Add NPU resource manager
   - Integrate with HAL device

3. **Phase 3: Reactive Features** (Week 4)
   - Add telemetry integration
   - Implement thermal throttling
   - Test deadline handling
   - Add priority preemption

4. **Phase 4: Validation** (Week 5)
   - Test with multiple concurrent models
   - Measure scheduling overhead
   - Validate NPU exclusivity
   - Tune for robotics workload

## Example Usage

```c
// 1. Create topology for your hardware
iree_task_topology_t topology;
configure_2_cluster_topology(&topology);  // Clusters 0-1

// 2. Create task executor
iree_task_executor_t* executor;
iree_task_executor_create(options, &topology, allocator, &executor);

// 3. Create custom HAL device with scheduler
iree_hal_device_t* device;
iree_hal_job_shop_device_create("job-shop", executor, allocator, &device);

// 4. Load and execute multiple models
iree_runtime_session_t* session;
load_models(session, {"perception.vmfb", "planning_npu.vmfb", "control.vmfb"});

// 5. Execute with pipelining
execute_with_scheduling(session, device, timeline);

// 6. Monitor and adapt
while (running) {
  update_telemetry(device, temperature, memory_pressure);
  // Scheduler adapts automatically
}
```

## Performance Considerations

### Scheduling Overhead
- **Goal**: < 10 microseconds per scheduling decision
- **Critical Path**: Lock contention on ready queue
- **Optimization**: Per-cluster queues, lock-free when possible

### NPU Contention
- **Challenge**: Multiple jobs may want NPU
- **Solution**: Priority queue for NPU jobs, preemption if needed
- **Metric**: NPU utilization > 90%

### Thermal Throttling
- **Trigger**: Core temperature > 85°C
- **Action**: Reduce concurrency, migrate work
- **Hysteresis**: Restore at < 70°C

### Deadline Satisfaction
- **Requirement**: 99% of high-priority jobs meet deadlines
- **Approach**: Earliest Deadline First (EDF) scheduling
- **Fallback**: Abort low-priority jobs if needed

## Testing Strategy

### Unit Tests
- Scheduler algorithm correctness
- NPU exclusivity
- Priority ordering
- Thermal throttling

### Integration Tests
- Multiple concurrent models
- Core assignment verification
- Timeline synchronization

### Performance Tests
- Scheduling overhead measurement
- Throughput comparison vs. default
- Stress test with many jobs

### Real-time Tests
- Deadline satisfaction rate
- Latency distribution
- Reactive scheduling behavior

## Conclusion

IREE provides an excellent foundation for custom scheduling:
- ✅ Clean architecture with clear extension points
- ✅ Powerful task system with topology awareness
- ✅ Timeline-based execution for pipelining
- ✅ Well-documented code

You need to add:
- ❌ Job shop scheduling algorithm
- ❌ Resource reservation (NPU)
- ❌ Priority/deadline enforcement
- ❌ Reactive adaptation

**The Good News**: IREE was designed for exactly this kind of extension! The abstractions are clean, the extension points are well-defined, and you can implement custom scheduling without modifying core IREE code.

**Estimated Total Effort**: 4-6 weeks for complete implementation and testing

## Resources

- **Documentation**: `docs/website/docs/developers/design-docs/`
- **IREE Discord**: https://discord.gg/wEWh6Z9nMU
- **GitHub Issues**: https://github.com/iree-org/iree/issues
- **Mailing List**: iree-technical-discussion@lists.lfaidata.foundation

## Contact

For questions about this investigation or implementation:
- Open a GitHub issue with tag `question`
- Ask in IREE Discord #performance channel
- Email the technical discussion mailing list

---

**Investigation completed by**: GitHub Copilot Coding Agent
**Date**: December 2, 2024
