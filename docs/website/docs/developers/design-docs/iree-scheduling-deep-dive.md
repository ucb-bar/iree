# IREE Scheduling Deep Dive: Understanding and Extending IREE's Task System

!!! note - "Authored December, 2024"

This document provides a comprehensive analysis of how IREE performs scheduling across all layers of the system, from compiler-level scheduling passes to runtime task execution and CPU core allocation. This guide is designed for developers who want to understand the exact scheduling mechanisms and implement custom scheduling policies, such as Flexible Job Shop Scheduling for heterogeneous computing environments.

## Table of Contents

1. [Overview of IREE's Multi-Layer Scheduling](#overview)
2. [Compiler-Level Scheduling](#compiler-scheduling)
3. [Runtime Task System](#runtime-task-system)
4. [CPU Core Allocation and Affinity](#cpu-allocation)
5. [Concurrent Execution and Pipelining](#concurrent-execution)
6. [Implementing Custom Scheduling Policies](#custom-scheduling)
7. [Key Files for Scheduling Implementation](#key-files)
8. [Recommendations for Flexible Job Shop Scheduling](#recommendations)

## Overview of IREE's Multi-Layer Scheduling {#overview}

IREE's scheduling operates across three main layers:

### 1. Compiler Layer (Ahead-of-Time)
- **Stream Dialect Scheduling**: Partitions operations into executable regions
- **Schedule Execution Pass**: Creates execution timelines and dependencies
- **Schedule Allocation Pass**: Manages memory allocation scheduling
- **Schedule Concurrency Pass**: Optimizes concurrent execution opportunities

### 2. HAL (Hardware Abstraction Layer)
- **Command Buffers**: Records work to be executed
- **Device Queues**: Submit work to execution resources
- **Semaphores/Fences**: Synchronize across timelines

### 3. Runtime Task System
- **Task Executor**: Central scheduling coordinator
- **Workers**: Thread pool mapped to CPU topology
- **Task Graphs**: DAG-based execution with fine-grained dependencies
- **Affinity Sets**: Control which workers execute which tasks

## Compiler-Level Scheduling {#compiler-scheduling}

### Stream Dialect Scheduling Passes

IREE's compiler performs scheduling at the IR level through several key passes:

#### ScheduleExecution Pass
**Location**: `compiler/src/iree/compiler/Dialect/Stream/Transforms/ScheduleExecution.cpp`

This pass:
- Partitions operations into executable regions (dispatches)
- Builds execution timelines with dependencies
- Creates `stream.cmd.execute` operations that wrap partitioned work
- Determines which operations can execute concurrently

Key concepts:
```mlir
// Example: Partitioned execution with dependencies
stream.cmd.execute
  with(%buffer as %arg0: !stream.resource<*>)
  on(#hal.affinity.queue<0>)
  {
    // Dispatch operations
    stream.cmd.dispatch @workload[%x, %y, %z](%arg0)
  }
```

#### ScheduleAllocation Pass
**Location**: `compiler/src/iree/compiler/Dialect/Stream/Transforms/ScheduleAllocation.cpp`

This pass:
- Schedules buffer allocations and deallocations
- Implements stream-ordered memory allocation
- Minimizes peak memory usage
- Enables memory reuse across dispatches

#### ScheduleConcurrency Pass
**Location**: `compiler/src/iree/compiler/Dialect/Stream/Transforms/ScheduleConcurrency.cpp`

This pass:
- Analyzes opportunities for concurrent execution
- Inserts explicit concurrency primitives
- Optimizes pipeline depth

### Compilation to Task Graphs

The compiler generates:
1. **Dispatch regions**: Kernel invocations with 3D grid dimensions
2. **Dependencies**: Task dependency edges (happens-before relationships)
3. **Affinity hints**: Which workers/cores should execute tasks

## Runtime Task System {#runtime-task-system}

### Architecture Overview

The runtime task system is a sophisticated work-stealing scheduler inspired by GPU execution models.

**Location**: `runtime/src/iree/task/`

Key components:
- **Executor** (`executor.h/c`): Central coordinator
- **Workers** (`worker.h/c`): Thread pool workers
- **Tasks** (`task.h/c`): Work units with dependencies
- **Topology** (`topology.h/c`): CPU topology and cache hierarchy
- **Affinity Sets** (`affinity_set.h`): Worker selection bitmasks

### Task Executor Design

The executor implements a wavefront-style scheduling algorithm:

```
┌─────────────────────────────────────────────────────┐
│              iree_task_executor_t                   │
│                                                     │
│  ┌─────────────────┐         ┌──────────────────┐ │
│  │ Incoming Ready  │────────►│   Coordinator    │ │
│  │     Queue       │         │   (Scheduler)    │ │
│  └─────────────────┘         └──────────────────┘ │
│                                      │             │
│                    ┌─────────────────┼────────┐    │
│                    │                 │        │    │
│                    ▼                 ▼        ▼    │
│            ┌────────────┐    ┌────────────┐ ...   │
│            │  Worker 0  │    │  Worker 1  │       │
│            │            │    │            │       │
│            │ ┌────────┐ │    │ ┌────────┐ │       │
│            │ │Mailbox │ │    │ │Mailbox │ │       │
│            │ └────────┘ │    │ └────────┘ │       │
│            │ ┌────────┐ │    │ ┌────────┐ │       │
│            │ │  FIFO  │ │    │ │  FIFO  │ │       │
│            │ └────────┘ │    │ └────────┘ │       │
│            └────────────┘    └────────────┘       │
└─────────────────────────────────────────────────────┘
```

### Task Types

From `runtime/src/iree/task/task.h`:

```c
enum iree_task_type_bits_t {
  IREE_TASK_TYPE_NOP = 0u,          // No-op for flexibility
  IREE_TASK_TYPE_CALL = 1u,         // Function call
  IREE_TASK_TYPE_BARRIER = 2u,      // Join/fork point
  IREE_TASK_TYPE_FENCE = 3u,        // Timeline synchronization
  IREE_TASK_TYPE_WAIT = 4u,         // Wait on external handle
  IREE_TASK_TYPE_DISPATCH = 5u,     // 3D grid dispatch
  IREE_TASK_TYPE_DISPATCH_SHARD = 6u, // Dispatch shard (worker portion)
};
```

### Task Lifecycle

1. **Task Creation**: Users create task DAGs with dependencies
   ```c
   iree_task_t* task = ...; // Allocated from pool
   iree_task_initialize(IREE_TASK_TYPE_DISPATCH, scope, task);
   task->affinity_set = worker_affinity; // Select workers
   task->completion_task = dependent_task; // Set dependency
   ```

2. **Submission**: Tasks submitted to executor's incoming queue
   ```c
   iree_task_submission_t submission;
   iree_task_submission_initialize(&submission);
   iree_task_submission_enqueue(&submission, task);
   iree_task_executor_submit(executor, &submission);
   ```

3. **Coordination**: Coordinator schedules ready tasks
   - Flushes incoming queue to coordinator-local FIFO
   - Schedules tasks to workers based on affinity
   - Posts tasks to worker mailboxes

4. **Execution**: Workers process tasks
   - Flush mailbox to local FIFO
   - Execute tasks in order
   - May steal work from other workers
   - Retire tasks and ready dependents

### Worker Execution Flow

From `runtime/src/iree/task/worker.c`:

```
┌──────────────────────────────────────────┐
│          Worker Thread Loop              │
│                                          │
│  1. Check mailbox for new tasks         │
│     ↓                                    │
│  2. Flush mailbox → local FIFO           │
│     ↓                                    │
│  3. Execute tasks from FIFO              │
│     ↓                                    │
│  4. Tasks complete → ready dependents    │
│     ↓                                    │
│  5. No more work?                        │
│     ├─ Try work stealing                │
│     ├─ Try coordinator role (schedule)   │
│     └─ Wait for notification             │
└──────────────────────────────────────────┘
```

## CPU Core Allocation and Affinity {#cpu-allocation}

### Topology Detection

IREE automatically detects the CPU topology including:
- Number of NUMA nodes
- Processor cores and logical processors (SMT/hyperthreading)
- Cache hierarchy (L1, L2, L3)
- Constructive cache sharing relationships

**Implementation**: `runtime/src/iree/task/topology.h` and platform-specific:
- `topology_sysfs.c` (Linux)
- `topology_darwin.c` (macOS)
- `topology_win32.c` (Windows)
- `topology_cpuinfo.c` (generic fallback)

### Topology Structure

```c
typedef struct iree_task_topology_group_t {
  uint8_t group_index;                    // Group ID
  char name[31];                          // "worker-0", etc.
  uint32_t processor_index;               // Logical processor
  iree_task_topology_caches_t caches;     // L1/L2/L3 sizes
  iree_thread_affinity_t ideal_thread_affinity;  // OS thread affinity
  iree_task_topology_group_mask_t constructive_sharing_mask;  // Cache sharing
} iree_task_topology_group_t;

typedef struct iree_task_topology_t {
  iree_host_size_t group_count;
  iree_task_topology_group_t groups[IREE_TASK_EXECUTOR_MAX_WORKER_COUNT];
} iree_task_topology_t;
```

### Affinity Sets

Tasks specify which workers can execute them using affinity sets:

```c
typedef uint64_t iree_task_affinity_set_t;

// Task affinity examples:
task->affinity_set = iree_task_affinity_for_worker(3);        // Only worker 3
task->affinity_set = iree_task_affinity_for_worker_range(0,4);// Workers 0-3
task->affinity_set = iree_task_affinity_for_any_worker();     // Any worker
task->affinity_set = 0b00001111;  // Workers 0,1,2,3 (bitset)
```

### How Core Assignment Works

1. **Topology Initialization**: Query system for CPU structure
   ```c
   iree_task_topology_t topology;
   iree_task_topology_initialize_from_physical_cores(
       /*max_core_count=*/8, &topology);
   ```

2. **Executor Creation**: Create workers mapped to topology
   ```c
   iree_task_executor_options_t options;
   iree_task_executor_options_initialize(&options);
   iree_task_executor_create(options, &topology, allocator, &executor);
   ```

3. **Worker Thread Affinity**: Each worker thread is pinned to its assigned core
   ```c
   // From worker.c initialization:
   worker->ideal_thread_affinity = group->ideal_thread_affinity;
   iree_thread_create_native(worker_main, worker, 
                              worker->ideal_thread_affinity,
                              &worker->thread);
   ```

4. **Task Scheduling**: Coordinator respects affinity when posting tasks
   ```c
   // From executor.c scheduling:
   iree_task_affinity_set_t allowed_workers = task->affinity_set;
   for (each worker in allowed_workers) {
     post_task_to_worker_mailbox(worker, task);
   }
   ```

## Concurrent Execution and Pipelining {#concurrent-execution}

### Multiple VMFB/MLIR Models

IREE supports several execution patterns for concurrent models:

#### 1. Multiple Contexts (Isolated Execution)
```c
// Each context has independent state
iree_vm_context_t* context1;
iree_vm_context_t* context2;

// Can execute concurrently on different timelines
iree_hal_semaphore_t* timeline1;
iree_hal_semaphore_t* timeline2;

// Submit work independently
iree_vm_invoke_async(context1, function1, /*wait*/timeline1, /*signal*/timeline1);
iree_vm_invoke_async(context2, function2, /*wait*/timeline2, /*signal*/timeline2);
```

#### 2. Shared Context (Sequential with Internal Pipelining)
```c
// Single context, multiple invocations
iree_hal_semaphore_t* timeline;

// These execute in submission order, but internally pipeline
iree_vm_invoke_async(context, function1, 
                     /*wait*/timeline_at(0), /*signal*/timeline_at(1));
iree_vm_invoke_async(context, function2, 
                     /*wait*/timeline_at(1), /*signal*/timeline_at(2));
iree_vm_invoke_async(context, function3, 
                     /*wait*/timeline_at(2), /*signal*/timeline_at(3));
```

#### 3. Fork-Join Patterns
```c
// Execute multiple models in parallel, then join
iree_hal_semaphore_t* timeline;

// Fork: both wait on same fence, signal different fences
iree_vm_invoke_async(context1, model1, 
                     /*wait*/timeline_at(0), /*signal*/timeline_at(1));
iree_vm_invoke_async(context2, model2, 
                     /*wait*/timeline_at(0), /*signal*/timeline_at(2));

// Join: wait on both completion fences
iree_hal_fence_t* join_fence = create_fence({timeline_at(1), timeline_at(2)});
iree_hal_fence_wait(join_fence, /*timeout*/INFINITE);
```

### Task-Level Concurrency

Within a single invocation, the task system provides fine-grained concurrency:

```
Dispatch (NxMxK grid)
    │
    ├─► Shard 0 (Worker 0) ─┐
    ├─► Shard 1 (Worker 1) ─┤
    ├─► Shard 2 (Worker 2) ─┼─► All complete
    └─► Shard 3 (Worker 3) ─┘      │
                                     ▼
                              Dependent Tasks
```

### Timeline-Based Pipelining

IREE's timeline model (see `invocation-execution-model.md`) enables:
- Out-of-order execution
- Producer-consumer pipelining
- Multi-model orchestration
- Stream-ordered memory allocation

Example of pipelined execution:
```
Timeline:  t0    t1    t2    t3    t4
           │     │     │     │     │
Model A:   │═════│═════│     │     │
           │     └─────┼─────│     │
Model B:   │     │═════│═════│     │
           │     │     └─────┼─────│
Model C:   │     │     │═════│═════│
```

## Implementing Custom Scheduling Policies {#custom-scheduling}

### Current Extension Points

IREE provides several mechanisms to influence scheduling:

#### 1. Task Affinity Control
You can control which workers execute tasks by setting affinity:
```c
// Custom affinity for NPU-capable cores
#define NPU_CLUSTER_MASK 0b11110000  // Cores 4-7 have NPU

iree_task_dispatch_t* dispatch = ...;
dispatch->header.affinity_set = NPU_CLUSTER_MASK;
```

#### 2. Worker Topology Customization
Define custom topology to match your hardware:
```c
iree_task_topology_t topology;
iree_task_topology_initialize(&topology);

// Cluster 0: Cores 0-3 (general purpose)
for (int i = 0; i < 4; i++) {
  iree_task_topology_group_t* group = &topology.groups[i];
  iree_task_topology_group_initialize(i, group);
  group->processor_index = i;
  group->constructive_sharing_mask = 0b00001111;  // Share L3
}

// Cluster 1: Cores 4-7 (with NPU extension)
for (int i = 4; i < 8; i++) {
  iree_task_topology_group_t* group = &topology.groups[i];
  iree_task_topology_group_initialize(i, group);
  group->processor_index = i;
  group->constructive_sharing_mask = 0b11110000;  // Share L3 + NPU
  // Could add custom metadata for NPU capability
}

topology.group_count = 8;
```

#### 3. Custom Task Pools
Implement custom task allocation for scheduling metadata:
```c
// Extend task structure with scheduling metadata
typedef struct my_custom_dispatch_t {
  iree_task_dispatch_t base;
  int priority;           // Job priority
  int job_id;            // Job shop job ID
  int operation_id;      // Operation within job
  uint64_t deadline_ns;  // Deadline constraint
  uint32_t resource_requirements;  // Required capabilities
} my_custom_dispatch_t;
```

#### 4. HAL Driver Customization
Create a custom HAL driver that implements scheduling at the HAL level:

**Location**: `runtime/src/iree/hal/drivers/`

Key files to reference:
- `local_task/task_device.c` - Task-based device implementation
- `local_sync/sync_device.c` - Synchronous device implementation

### Implementing Flexible Job Shop Scheduling

For your specific use case (4 cores per cluster, 2 clusters, 1 with NPU), here's a recommended approach:

#### Approach 1: Custom HAL Device (Recommended for Complex Scheduling)

1. **Create Custom HAL Driver**
   - Extend `local_task` driver with custom scheduling logic
   - Implement job shop scheduling algorithm in device
   - Location: `runtime/src/iree/hal/drivers/custom_scheduler/`

2. **Implement Scheduling Algorithm**
   ```c
   typedef struct {
     iree_hal_device_t base;
     iree_task_executor_t* executor;
     
     // Job shop state
     struct {
       iree_slim_mutex_t mutex;
       job_queue_t pending_jobs;
       schedule_t active_schedule;
       resource_state_t cluster_states[2];
       npu_manager_t npu_manager;
     } scheduler;
   } custom_scheduler_device_t;
   
   // Custom dispatch submission with scheduling
   iree_status_t custom_device_queue_execute(
       iree_hal_device_t* base_device,
       iree_hal_command_buffer_t* command_buffer,
       iree_hal_semaphore_list_t wait_semaphores,
       iree_hal_semaphore_list_t signal_semaphores) {
     
     custom_scheduler_device_t* device = 
         (custom_scheduler_device_t*)base_device;
     
     // Extract dispatches and metadata
     job_t job = analyze_command_buffer(command_buffer);
     
     // Run scheduling algorithm
     schedule_decision_t decision = 
         schedule_job(&device->scheduler, job);
     
     // Assign to appropriate cluster/cores
     task->affinity_set = decision.assigned_cores;
     
     // Submit with proper dependencies
     return submit_scheduled_task(device->executor, task);
   }
   ```

3. **NPU Resource Management**
   ```c
   typedef struct {
     bool npu_available;
     iree_task_affinity_set_t npu_cores;  // Cores 4-7
     iree_atomic_int32_t npu_in_use;
     queue_t npu_queue;
   } npu_manager_t;
   
   bool try_acquire_npu(npu_manager_t* mgr, task_t* task) {
     if (task->requires_npu) {
       if (iree_atomic_compare_exchange_strong(
             &mgr->npu_in_use, 0, 1, 
             iree_memory_order_acquire,
             iree_memory_order_relaxed)) {
         task->affinity_set = mgr->npu_cores;
         return true;
       }
       return false;  // Queue for later
     }
     return true;  // No NPU needed
   }
   ```

#### Approach 2: Custom Executor (More Control, More Work)

1. **Extend Task Executor**
   - Fork `runtime/src/iree/task/executor.c`
   - Add job shop scheduling logic to coordinator
   - Location: `runtime/src/iree/task/custom_executor.c`

2. **Implement Custom Coordinator**
   ```c
   static iree_status_t custom_coordinator_schedule(
       iree_task_executor_t* executor,
       iree_task_list_t* incoming_tasks) {
     
     job_shop_scheduler_t* scheduler = &executor->custom_scheduler;
     
     // Analyze tasks
     for (task in incoming_tasks) {
       job_metadata_t* metadata = extract_metadata(task);
       
       // Classify by requirements
       if (metadata->needs_npu) {
         enqueue(&scheduler->npu_queue, task);
       } else if (metadata->priority == HIGH) {
         enqueue(&scheduler->high_priority_queue, task);
       } else {
         enqueue(&scheduler->normal_queue, task);
       }
     }
     
     // Schedule based on job shop algorithm
     schedule_result_t result = compute_schedule(scheduler);
     
     // Post to workers according to schedule
     return post_scheduled_tasks(executor, result);
   }
   ```

#### Approach 3: Compiler-Level Annotations (Least Invasive)

1. **Add Custom Attributes to IR**
   ```mlir
   // In your MLIR
   stream.cmd.dispatch @matmul_npu[%x, %y, %z]
       attributes {
         iree.dispatch.priority = 10,
         iree.dispatch.requires_npu = true,
         iree.dispatch.job_id = 5,
         iree.dispatch.operation_id = 2
       }
   ```

2. **Custom Compilation Pass**
   - Add pass to propagate annotations to task affinity
   - Location: `compiler/src/iree/compiler/Dialect/Stream/Transforms/CustomScheduling.cpp`

3. **Runtime Reads Annotations**
   - Modify task submission to read and apply annotations
   - Simpler but less dynamic

### Reactive Scheduling for Robotics

For reactive scheduling that adapts to environment:

1. **Telemetry and Feedback**
   ```c
   typedef struct {
     uint64_t last_execution_time_ns;
     uint32_t cpu_temperature;
     uint32_t memory_pressure;
     uint32_t battery_level;
     bool npu_thermal_throttling;
   } system_state_t;
   
   void update_schedule_based_on_state(
       scheduler_t* scheduler,
       system_state_t* state) {
     
     if (state->npu_thermal_throttling) {
       // Migrate work from NPU cluster to general cluster
       migrate_tasks(scheduler, NPU_CLUSTER, GENERAL_CLUSTER);
     }
     
     if (state->battery_level < 20) {
       // Reduce parallelism, favor efficiency
       scheduler->max_active_workers = 4;
     }
     
     if (state->memory_pressure > THRESHOLD) {
       // Serialize some concurrent jobs
       scheduler->max_concurrent_jobs = 2;
     }
   }
   ```

2. **Priority Inversion Handling**
   ```c
   void check_and_resolve_priority_inversion(scheduler_t* s) {
     for (each blocked high-priority task) {
       task_t* blocker = find_blocker(task);
       if (blocker && blocker->priority < task->priority) {
         // Temporarily boost blocker priority
         boost_priority(blocker, task->priority);
       }
     }
   }
   ```

## Key Files for Scheduling Implementation {#key-files}

### Compiler (MLIR/Scheduling)
```
compiler/src/iree/compiler/Dialect/Stream/Transforms/
├── ScheduleExecution.cpp          - Main execution scheduling
├── ScheduleAllocation.cpp         - Memory allocation scheduling  
├── ScheduleConcurrency.cpp        - Concurrency optimization
└── Passes.h                       - Pass definitions

compiler/src/iree/compiler/Dialect/Stream/Analysis/
└── Partitioning.h                 - Operation partitioning
```

### Runtime (Task System)
```
runtime/src/iree/task/
├── executor.h/c                   - Main task executor
├── executor_impl.h                - Internal executor structures
├── worker.h/c                     - Worker thread implementation
├── task.h/c                       - Task definitions and lifecycle
├── topology.h/c                   - CPU topology detection
├── affinity_set.h                 - Affinity bit manipulation
├── queue.h/c                      - Task queues
├── pool.h/c                       - Task pools
└── submission.h/c                 - Task submission

runtime/src/iree/base/internal/
├── threading.h/c                  - Platform threading
├── atomics.h                      - Atomic operations
└── synchronization.h              - Synchronization primitives
```

### HAL (Device Abstraction)
```
runtime/src/iree/hal/
├── device.h/c                     - Device interface
├── command_buffer.h/c             - Command recording
└── semaphore.h/c                  - Synchronization

runtime/src/iree/hal/drivers/local_task/
├── task_device.h/c                - Task-based HAL device
├── task_queue.h/c                 - Task queue implementation
└── task_command_buffer.h/c       - Task command buffer

runtime/src/iree/hal/drivers/local_sync/
└── sync_device.h/c                - Synchronous HAL device (reference)
```

### Build System
```
runtime/src/iree/task/
├── CMakeLists.txt                 - CMake build rules
└── BUILD.bazel                    - Bazel build rules
```

## Recommendations for Flexible Job Shop Scheduling {#recommendations}

Based on your requirements (4 cores/cluster × 2 clusters, 1 with NPU, robotics):

### Recommendation 1: Custom HAL Driver with Job Shop Scheduler ⭐ (Best Option)

**Why**: 
- Clean separation of concerns
- Can schedule across multiple models/contexts
- Full control over resource allocation
- Easy to maintain and test

**Implementation Steps**:
1. Fork `runtime/src/iree/hal/drivers/local_task/` → `custom_scheduler/`
2. Add job shop scheduling state to device structure
3. Implement scheduling algorithm in `queue_execute()`
4. Add NPU resource manager
5. Expose scheduling metrics for monitoring
6. Build with CMake/Bazel

**Estimated Effort**: 2-3 weeks for experienced C developer

### Recommendation 2: Extended Task Executor

**Why**:
- More fundamental control
- Can optimize task-level scheduling
- Better for fine-grained parallelism

**Drawbacks**:
- More invasive changes
- Harder to maintain across IREE updates
- Need to understand executor internals deeply

**Estimated Effort**: 3-4 weeks

### Recommendation 3: Hybrid Approach (Recommended for Gradual Migration)

1. **Phase 1**: Start with topology + affinity
   - Define custom topology matching your hardware
   - Use affinity sets to partition work
   - **Effort**: 1 week

2. **Phase 2**: Add HAL-level scheduling hooks
   - Intercept command buffer submission
   - Implement basic job prioritization
   - **Effort**: 2 weeks

3. **Phase 3**: Full job shop scheduler
   - Implement complete scheduling algorithm
   - Add reactive capabilities
   - **Effort**: 2-3 weeks

### Key Design Principles for Your Implementation

1. **Hardware Awareness**
   ```
   Cluster 0 (General):     Cluster 1 (NPU):
   ┌─────────────────┐     ┌─────────────────┐
   │ Core 0  Core 1  │     │ Core 4  Core 5  │
   │ Core 2  Core 3  │     │ Core 6  Core 7  │
   │                 │     │      NPU        │
   │    L2 Cache     │     │   L2 + NPU      │
   └─────────────────┘     └─────────────────┘
          │                        │
          └────────┬───────────────┘
                   │
              L3 Cache (Shared)
   ```

2. **Job Metadata**
   - Job ID and operation sequence
   - Resource requirements (compute, memory, NPU)
   - Deadline constraints
   - Priority levels
   - Precedence relationships

3. **Scheduling Algorithm Components**
   - **Ready Queue**: Jobs ready to execute
   - **Resource State**: Track cluster/NPU availability
   - **Precedence Graph**: Job dependencies
   - **Cost Model**: Estimate execution time
   - **Assignment Policy**: Map jobs to resources

4. **Reactive Features**
   - Monitor system state (temp, power, latency)
   - Adjust scheduling parameters dynamically
   - Preempt/migrate long-running tasks if needed
   - Handle priority inversion

5. **Integration Points**
   - Hook into HAL command submission
   - Read dispatch metadata (grid size, kernel type)
   - Apply NPU-specific optimizations
   - Emit scheduling traces for analysis

### Testing Strategy

1. **Unit Tests**
   - Test scheduling algorithm in isolation
   - Test resource allocation logic
   - Test priority handling

2. **Integration Tests**
   - Run multiple concurrent models
   - Verify correct core assignment
   - Check NPU exclusivity

3. **Performance Tests**
   - Measure scheduling overhead
   - Compare throughput vs. default scheduler
   - Stress test with many concurrent jobs

4. **Real-time Tests**
   - Verify deadline satisfaction
   - Test reactive scheduling behavior
   - Measure latency distribution

## Summary: Does IREE Already Do This?

**Current State**:
- ✅ IREE has sophisticated task scheduling and work-stealing
- ✅ IREE supports topology-aware worker placement
- ✅ IREE provides affinity control for task-to-core mapping
- ✅ IREE enables concurrent execution of multiple models via timelines
- ✅ IREE supports pipelined execution with fine-grained dependencies
- ❌ IREE does NOT have built-in job shop scheduling
- ❌ IREE does NOT have explicit resource reservation (like NPU)
- ❌ IREE does NOT have priority-based preemptive scheduling
- ❌ IREE does NOT have reactive scheduling that adapts to environment

**What You Need to Add**:
1. Job shop scheduling algorithm (decide which job/operation runs when/where)
2. Resource manager for NPU (exclusive access, queueing)
3. Priority and deadline enforcement
4. Reactive scheduling based on telemetry
5. Cross-model orchestration with custom policies

**The Good News**:
IREE's architecture is designed for exactly this kind of extension! The abstractions are clean, the code is well-documented, and there are clear extension points. Your custom scheduler can live alongside IREE's default scheduler and you can choose which to use per-device or per-application.

## Next Steps

1. **Familiarize Yourself with the Codebase**
   - Read the referenced files in this document
   - Run existing tests: `cd runtime && ctest -R task`
   - Trace execution with IREE's Tracy integration

2. **Build a Prototype**
   - Start with custom topology definition
   - Add affinity-based partitioning
   - Verify correct core assignment

3. **Implement Core Scheduling**
   - Design your job shop scheduler algorithm
   - Implement in custom HAL driver
   - Add NPU resource manager

4. **Add Reactive Features**
   - Integrate telemetry
   - Implement adaptation logic
   - Test under various conditions

5. **Optimize and Validate**
   - Profile scheduling overhead
   - Measure end-to-end latency
   - Validate against robotics requirements

## Additional Resources

- **IREE Discord**: https://discord.gg/wEWh6Z9nMU (Ask in #general or #performance)
- **IREE Developer Docs**: https://iree.dev/developers/
- **Task System Design**: `runtime/src/iree/task/executor.h` (see header comments)
- **Invocation Model**: `docs/website/docs/developers/design-docs/invocation-execution-model.md`
- **HAL Design**: `docs/website/docs/developers/design-docs/` (various HAL driver docs)

## Questions & Feedback

This document was created to help you understand IREE's scheduling and implement custom job shop scheduling. If you have questions or need clarification on any section, please reach out through:
- GitHub Issues: https://github.com/iree-org/iree/issues
- IREE Discord: https://discord.gg/wEWh6Z9nMU
- Email: iree-technical-discussion mailing list

Good luck with your implementation! IREE's task system is powerful and flexible - you should be able to implement sophisticated scheduling policies while maintaining compatibility with the rest of the stack.
