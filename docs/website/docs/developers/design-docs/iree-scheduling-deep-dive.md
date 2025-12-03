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
7. [Ahead-of-Time (AOT) Scheduling with Flexible Job Shop](#aot-scheduling)
8. [Key Files for Scheduling Implementation](#key-files)
9. [Recommendations for Flexible Job Shop Scheduling](#recommendations)

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

### Dynamic Dispatch: Fine-Grained Per-Kernel Scheduling

Instead of statically pinning workloads to fixed core sets, you can implement **dynamic dispatch** where each dispatch/kernel is assigned to cores at runtime based on current system state. This provides maximum flexibility and avoids reserving cores exclusively.

#### Key Concept: Dynamic Affinity Assignment

Rather than setting `task->affinity_set = 0b11110000` (fixed to cores 4-7), the scheduler dynamically computes affinity per-dispatch:

```c
// Dynamic affinity computation at dispatch time
iree_task_affinity_set_t compute_dynamic_affinity(
    scheduler_t* scheduler,
    dispatch_t* dispatch) {
  
  // Analyze dispatch characteristics
  bool needs_npu = dispatch_requires_npu(dispatch);
  uint32_t compute_intensity = estimate_compute_intensity(dispatch);
  
  // Check current cluster utilization
  float cluster0_util = get_cluster_utilization(scheduler, 0);
  float cluster1_util = get_cluster_utilization(scheduler, 1);
  
  if (needs_npu) {
    // NPU required - can only use cluster 1
    if (is_npu_available(scheduler)) {
      return scheduler->clusters[1].core_mask;  // 0b11110000
    } else {
      // NPU busy - queue for later
      return 0;  // Will be rescheduled
    }
  }
  
  // General compute - choose dynamically
  if (cluster0_util < cluster1_util * 0.7) {
    // Cluster 0 less loaded - use it
    return scheduler->clusters[0].core_mask;  // 0b00001111
  } else if (cluster1_util < 0.5 && !npu_in_use(scheduler)) {
    // Cluster 1 available and NPU not in use
    return scheduler->clusters[1].core_mask;  // 0b11110000
  } else {
    // Balance across both clusters
    return select_least_loaded_cores(scheduler, 4);
  }
}
```

#### Per-Dispatch Granularity

To achieve high granularity (different cores per dispatch), intercept at the dispatch level:

```c
iree_status_t custom_device_queue_execute(
    iree_hal_device_t* base_device,
    iree_hal_command_buffer_t* command_buffer,
    ...) {
  
  custom_device_t* device = (custom_device_t*)base_device;
  
  // Iterate through all dispatches in command buffer
  for (each dispatch in command_buffer) {
    
    // Analyze this specific dispatch
    dispatch_characteristics_t chars = analyze_dispatch(dispatch);
    
    // Compute affinity dynamically based on:
    // - Current cluster load
    // - Dispatch compute characteristics
    // - Resource availability (NPU, memory bandwidth)
    // - Thermal state
    iree_task_affinity_set_t affinity = 
        compute_dynamic_affinity(&device->scheduler, &chars);
    
    // If no cores available, queue for later
    if (affinity == 0) {
      enqueue_for_rescheduling(&device->scheduler, dispatch);
      continue;
    }
    
    // Set affinity on the task
    iree_task_dispatch_t* task = create_dispatch_task(dispatch);
    task->header.affinity_set = affinity;  // Dynamic assignment!
    
    // Submit to executor
    submit_task(&device->scheduler, task);
  }
  
  return iree_ok_status();
}
```

#### Fine-Grained Core Selection

For even finer granularity, select individual cores within a cluster:

```c
// Select N least-loaded cores from available set
iree_task_affinity_set_t select_least_loaded_cores(
    scheduler_t* scheduler,
    uint32_t num_cores) {
  
  // Get per-core load
  uint32_t load[8];  // 8 cores total
  for (int i = 0; i < 8; i++) {
    load[i] = get_core_load(scheduler, i);
  }
  
  // Sort cores by load (ascending)
  uint32_t sorted_cores[8];
  sort_by_load(sorted_cores, load, 8);
  
  // Build affinity mask with N least-loaded cores
  iree_task_affinity_set_t affinity = 0;
  for (int i = 0; i < num_cores && i < 8; i++) {
    affinity |= (1ULL << sorted_cores[i]);
  }
  
  return affinity;
}
```

#### Dispatch Sharding with Dynamic Distribution

IREE's dispatch sharding mechanism can be leveraged for dynamic distribution:

```c
// Custom dispatch shard distribution
void distribute_dispatch_shards_dynamically(
    iree_task_executor_t* executor,
    iree_task_dispatch_t* dispatch,
    scheduler_state_t* state) {
  
  // Determine how many shards (tiles) this dispatch has
  uint32_t total_tiles = dispatch->workgroup_count.x * 
                         dispatch->workgroup_count.y * 
                         dispatch->workgroup_count.z;
  
  // Get currently available cores (not fixed!)
  iree_task_affinity_set_t available_cores = 
      get_available_cores_now(state);
  
  uint32_t num_workers = count_bits(available_cores);
  
  // Create shards - each shard gets a subset of tiles
  for (uint32_t i = 0; i < num_workers; i++) {
    iree_task_dispatch_shard_t* shard = create_shard(dispatch);
    
    // Assign shard to specific available worker
    uint32_t worker_id = get_nth_set_bit(available_cores, i);
    shard->header.affinity_set = (1ULL << worker_id);
    
    // Shard will steal tiles from dispatch's tile pool
    submit_shard(executor, shard);
  }
}
```

#### Work Stealing as Dynamic Rebalancing

IREE's work-stealing already provides dynamic rebalancing. Enable it effectively:

```c
// Configure topology for work stealing
void configure_dynamic_topology(iree_task_topology_t* topology) {
  for (int i = 0; i < 8; i++) {
    iree_task_topology_group_t* group = &topology->groups[i];
    
    // Allow stealing from nearby cores
    if (i < 4) {
      // Cluster 0: Can steal from cluster 0 (prioritize) and cluster 1
      group->constructive_sharing_mask = 0b11111111;  // All cores
    } else {
      // Cluster 1: Can steal from cluster 1 (prioritize) and cluster 0
      group->constructive_sharing_mask = 0b11111111;  // All cores
    }
  }
  
  // Workers will naturally rebalance via work stealing
  // No fixed pinning - dynamic load balancing!
}
```

#### Monitoring and Adaptation

Track dispatch execution to refine dynamic decisions:

```c
typedef struct {
  uint64_t dispatch_id;
  uint64_t start_time_ns;
  uint64_t duration_ns;
  iree_task_affinity_set_t cores_used;
  uint32_t cluster_id;
  float achieved_utilization;
} dispatch_history_t;

void learn_from_dispatch_execution(
    scheduler_t* scheduler,
    dispatch_history_t* history) {
  
  // Update dispatch characteristics database
  if (history->achieved_utilization > 0.9) {
    // Good assignment - record for similar dispatches
    record_successful_assignment(
        scheduler->dispatch_db,
        history->dispatch_id,
        history->cluster_id);
  } else if (history->achieved_utilization < 0.5) {
    // Poor assignment - avoid in future
    record_poor_assignment(
        scheduler->dispatch_db,
        history->dispatch_id,
        history->cluster_id);
  }
}
```

#### Benefits of Dynamic Dispatch

1. **No Fixed Reservations**: Cores aren't locked to specific workloads
2. **Load Balancing**: Work naturally flows to available cores
3. **Adaptability**: Responds to thermal throttling, priority changes
4. **Efficiency**: Maximizes utilization across all cores
5. **Flexibility**: Same hardware can serve diverse workload mixes

#### Implementation Strategy

**Phase 1: Basic Dynamic Assignment**
- Compute affinity per command buffer submission
- Choose cluster based on current load
- No fixed core masks

**Phase 2: Per-Dispatch Granularity**
- Iterate dispatches within command buffer
- Assign affinity per dispatch
- Queue if cores unavailable

**Phase 3: Shard-Level Distribution**
- Use dispatch sharding mechanism
- Distribute shards to available workers
- Leverage work stealing

**Phase 4: Adaptive Learning**
- Track dispatch execution history
- Refine assignment decisions
- Predict optimal core assignments

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

## Ahead-of-Time (AOT) Scheduling with Flexible Job Shop {#aot-scheduling}

### Overview

Instead of relying entirely on runtime scheduling, you can implement **heavy ahead-of-time scheduling** where the compiler pre-computes an optimal schedule based on known architecture, software characteristics, and workload patterns. This is particularly valuable when you have:

- **Known hardware topology**: Fixed CPU clusters, accelerator configurations
- **Known workloads**: Multiple models with predictable characteristics
- **Deterministic requirements**: Real-time robotics with hard deadlines
- **Optimization opportunities**: Cross-model scheduling that runtime can't see

### AOT Scheduling Architecture

```
┌────────────────────────────────────────────────────────┐
│         Compiler (Ahead-of-Time)                       │
│                                                        │
│  1. Analyze multiple models/workloads                 │
│  2. Build combined scheduling graph                   │
│  3. Run Flexible Job Shop Scheduling algorithm        │
│  4. Compute optimal schedule with windows             │
│  5. Emit schedule as IR annotations/metadata          │
└────────────────────┬───────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│         Stream Dialect (Schedule Encoding)             │
│                                                        │
│  - Annotate dispatches with timing/affinity           │
│  - Insert explicit barriers/dependencies              │
│  - Encode resource reservations                       │
│  - Generate async execution with fences               │
└────────────────────┬───────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│         Runtime (Schedule Execution)                   │
│                                                        │
│  - Read schedule annotations                          │
│  - Enforce timing/affinity constraints                │
│  - Handle deviations (adaptive fallback)              │
│  - Monitor execution vs. predicted schedule           │
└────────────────────────────────────────────────────────┘
```

### Where to Implement AOT Scheduling

#### Option 1: Custom Compiler Pass (Recommended)

**Location**: `compiler/src/iree/compiler/Dialect/Stream/Transforms/FlexibleJobShopScheduling.cpp`

**Approach**: Add a new pass that runs after `ScheduleExecution` but before lowering to HAL:

```cpp
// FlexibleJobShopScheduling.cpp

class FlexibleJobShopSchedulingPass 
    : public PassWrapper<FlexibleJobShopSchedulingPass, 
                         OperationPass<ModuleOp>> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // 1. Collect all dispatches across all functions
    SmallVector<DispatchInfo> dispatches;
    collectDispatches(module, dispatches);
    
    // 2. Build scheduling graph with dependencies
    SchedulingGraph graph = buildSchedulingGraph(dispatches);
    
    // 3. Run Flexible Job Shop Scheduling algorithm
    Schedule optimalSchedule = 
        computeFlexibleJobShopSchedule(graph, hardwareTopology_);
    
    // 4. Apply schedule by annotating IR
    applyScheduleToIR(module, optimalSchedule);
  }
  
private:
  void applyScheduleToIR(ModuleOp module, const Schedule& schedule) {
    for (const ScheduleDecision& decision : schedule.decisions) {
      // Find dispatch operation
      auto dispatchOp = findDispatch(module, decision.dispatchId);
      
      // Annotate with scheduling decisions
      dispatchOp->setAttr("iree.scheduling.start_time_ns",
                          builder.getI64IntegerAttr(decision.startTime));
      dispatchOp->setAttr("iree.scheduling.affinity",
                          builder.getI64IntegerAttr(decision.affinity));
      dispatchOp->setAttr("iree.scheduling.priority",
                          builder.getI32IntegerAttr(decision.priority));
      
      // Encode resource reservations
      if (decision.requiresNPU) {
        dispatchOp->setAttr("iree.scheduling.resource",
                            builder.getStringAttr("npu"));
      }
    }
  }
};
```

**Integration Point**: Add to pass pipeline in `Passes.cpp`:

```cpp
void buildIREEVMTransformPassPipeline(OpPassManager &pm) {
  // ... existing passes ...
  pm.addPass(createScheduleExecutionPass());
  pm.addPass(createScheduleAllocationPass());
  
  // ADD YOUR AOT SCHEDULING PASS HERE
  pm.addPass(createFlexibleJobShopSchedulingPass());
  
  pm.addPass(createScheduleConcurrencyPass());
  // ... rest of pipeline ...
}
```

#### Option 2: External Tool + IR Annotation

**Approach**: Separate optimization tool that analyzes compiled modules and produces scheduling annotations:

```python
# aot_scheduler.py - External scheduling tool

import iree.compiler as compiler
import networkx as nx
from ortools.sat.python import cp_model

def schedule_multiple_modules(module_paths, hardware_config):
    # 1. Load and analyze modules
    modules = [load_iree_module(path) for path in module_paths]
    
    # 2. Extract dispatch information
    dispatches = []
    for module in modules:
        dispatches.extend(extract_dispatches(module))
    
    # 3. Build job shop scheduling model
    model = cp_model.CpModel()
    
    # Variables: start time, assigned core for each dispatch
    starts = {}
    cores = {}
    
    for d in dispatches:
        starts[d.id] = model.NewIntVar(0, MAX_TIME, f'start_{d.id}')
        cores[d.id] = model.NewIntVar(0, NUM_CORES-1, f'core_{d.id}')
    
    # 4. Add constraints
    # - Precedence constraints (dependencies)
    for dep in get_dependencies(dispatches):
        model.Add(starts[dep.successor] >= 
                  starts[dep.predecessor] + dep.predecessor.duration)
    
    # - Resource constraints (NPU exclusivity)
    for d1, d2 in get_npu_dispatches(dispatches):
        if d1 != d2:
            model.AddNoOverlap([
                model.NewIntervalVar(starts[d1.id], d1.duration, ...),
                model.NewIntervalVar(starts[d2.id], d2.duration, ...)
            ])
    
    # - Core capacity constraints
    for core_id in range(NUM_CORES):
        core_dispatches = [d for d in dispatches 
                           if can_run_on_core(d, core_id)]
        # Add disjunctive constraint for core
        model.AddNoOverlap([...])
    
    # 5. Optimize (minimize makespan)
    makespan = model.NewIntVar(0, MAX_TIME, 'makespan')
    for d in dispatches:
        model.Add(makespan >= starts[d.id] + d.duration)
    
    model.Minimize(makespan)
    
    # 6. Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL:
        schedule = {
            d.id: {
                'start_time': solver.Value(starts[d.id]),
                'core': solver.Value(cores[d.id])
            }
            for d in dispatches
        }
        return schedule
    
    return None

# 7. Emit annotated IR
def emit_scheduled_mlir(modules, schedule, output_path):
    for module in modules:
        for dispatch in module.dispatches:
            sched = schedule[dispatch.id]
            
            # Add scheduling attributes
            dispatch.add_attr('iree.scheduling.start_time_ns', sched['start_time'])
            dispatch.add_attr('iree.scheduling.affinity', 1 << sched['core'])
    
    # Serialize back to MLIR with annotations
    compiler.write_mlir(modules, output_path)
```

**Usage**:
```bash
# 1. Compile modules to MLIR (before final lowering)
iree-compile model_a.mlir -o=model_a.scheduled.mlir --emit-mlir
iree-compile model_b.mlir -o=model_b.scheduled.mlir --emit-mlir

# 2. Run AOT scheduler
python aot_scheduler.py \
    --inputs model_a.scheduled.mlir model_b.scheduled.mlir \
    --hardware-config cluster_config.json \
    --output scheduled_combined.mlir

# 3. Continue compilation with schedule
iree-compile scheduled_combined.mlir -o=final.vmfb
```

### How to Encode Schedule in IR

#### Stream Dialect Annotations

Add custom attributes to stream operations:

```mlir
// Before AOT scheduling
stream.cmd.execute with(%buffer) {
  stream.cmd.dispatch @workload[%x, %y, %z](%buffer)
}

// After AOT scheduling
stream.cmd.execute with(%buffer) {
  stream.cmd.dispatch @workload[%x, %y, %z](%buffer)
    {
      iree.scheduling.start_time_ns = 1500000 : i64,
      iree.scheduling.affinity = 240 : i64,  // 0b11110000 = cores 4-7
      iree.scheduling.priority = 100 : i32,
      iree.scheduling.resource = "npu"
    }
}
```

#### Explicit Timeline Construction

Use async-external execution model with explicit fences:

```mlir
// model_a.mlir
module @model_a {
  func.func @predict_async(%input: tensor<1x224x224x3xf32>) 
      -> !hal.fence attributes {
    iree.abi.model = "async-external"
  } {
    %fence = hal.fence.create : !hal.fence
    
    // Dispatch annotated with schedule
    stream.cmd.execute 
        on(#hal.affinity.queue<[0]>)  // Cluster 0
        wait(%input_ready_fence)
        signal(%fence) {
      stream.cmd.dispatch @conv2d[%x, %y, %z](%input)
        { iree.scheduling.start_time_ns = 0 : i64 }
    }
    
    return %fence : !hal.fence
  }
}

// model_b.mlir  
module @model_b {
  func.func @plan_async(%perception: tensor<...>) 
      -> !hal.fence attributes {
    iree.abi.model = "async-external"
  } {
    %fence = hal.fence.create : !hal.fence
    
    // Runs after model_a, on cluster 1 with NPU
    stream.cmd.execute
        on(#hal.affinity.queue<[1]>)  // Cluster 1
        wait(%perception_fence)
        signal(%fence) {
      stream.cmd.dispatch @npu_planning[%x, %y, %z](%perception)
        { 
          iree.scheduling.start_time_ns = 5000000 : i64,
          iree.scheduling.resource = "npu"
        }
    }
    
    return %fence : !hal.fence
  }
}
```

### Runtime Integration

The runtime needs to respect the AOT schedule while adapting to reality:

#### Option 1: Strict Enforcement (Deterministic)

```c
// Custom HAL device that enforces AOT schedule

static iree_status_t aot_scheduled_device_queue_execute(
    iree_hal_device_t* base_device,
    iree_hal_command_buffer_t* command_buffer,
    ...) {
  
  aot_scheduled_device_t* device = (aot_scheduled_device_t*)base_device;
  
  // Extract schedule from command buffer metadata
  int64_t scheduled_start_time_ns = 
      get_dispatch_attr(command_buffer, "iree.scheduling.start_time_ns");
  int64_t scheduled_affinity =
      get_dispatch_attr(command_buffer, "iree.scheduling.affinity");
  
  // Wait until scheduled start time
  int64_t current_time_ns = iree_time_now();
  if (current_time_ns < scheduled_start_time_ns) {
    iree_time_t wait_duration = scheduled_start_time_ns - current_time_ns;
    iree_wait_until(device->timer, wait_duration);
  }
  
  // Enforce affinity
  iree_task_t* task = create_task_for_command_buffer(command_buffer);
  task->affinity_set = (iree_task_affinity_set_t)scheduled_affinity;
  
  // Submit with schedule constraints
  return submit_task_at_time(device->executor, task, scheduled_start_time_ns);
}
```

#### Option 2: Adaptive Execution (Practical)

```c
// Respects schedule but adapts to deviations

static iree_status_t adaptive_aot_device_queue_execute(
    iree_hal_device_t* base_device,
    iree_hal_command_buffer_t* command_buffer,
    ...) {
  
  adaptive_aot_device_t* device = (adaptive_aot_device_t*)base_device;
  
  // Get scheduled parameters
  int64_t scheduled_start_ns = get_dispatch_attr(
      command_buffer, "iree.scheduling.start_time_ns");
  int64_t scheduled_affinity = get_dispatch_attr(
      command_buffer, "iree.scheduling.affinity");
  int32_t priority = get_dispatch_attr(
      command_buffer, "iree.scheduling.priority");
  
  int64_t current_time = iree_time_now();
  int64_t schedule_slack = current_time - scheduled_start_ns;
  
  // If we're behind schedule, adjust
  if (schedule_slack > THRESHOLD_NS) {
    // We're late - boost priority
    priority = min(priority + 50, 255);
    
    // Maybe use different cores if scheduled ones are busy
    scheduled_affinity = compute_adaptive_affinity(
        device, scheduled_affinity, priority);
  }
  
  // Create task with schedule hints
  iree_task_t* task = create_task_for_command_buffer(command_buffer);
  task->affinity_set = (iree_task_affinity_set_t)scheduled_affinity;
  task->priority = priority;
  task->scheduled_start_time_ns = scheduled_start_ns;
  
  // Submit - runtime will try to respect schedule but can adapt
  return submit_task_with_schedule(device->executor, task);
}
```

### Windowed Flexible Job Shop Scheduling

For long-running robotics applications, use **sliding window** approach:

```c
// Compiler: Compute schedule for next N time windows

Schedule compute_windowed_schedule(
    WorkloadSet workloads,
    HardwareConfig hardware,
    uint32_t window_size_ms,
    uint32_t num_windows) {
  
  Schedule full_schedule;
  
  for (uint32_t window = 0; window < num_windows; window++) {
    uint64_t window_start = window * window_size_ms * 1000000;
    uint64_t window_end = window_start + window_size_ms * 1000000;
    
    // Get jobs active in this window
    JobSet window_jobs = get_jobs_in_window(
        workloads, window_start, window_end);
    
    // Solve job shop scheduling for this window
    WindowSchedule window_schedule = solve_job_shop(
        window_jobs, hardware, window_start, window_end);
    
    // Add to full schedule
    full_schedule.merge(window_schedule);
  }
  
  return full_schedule;
}

// Runtime: Execute current window, prepare next

void execute_windowed_schedule(runtime_t* runtime) {
  uint32_t current_window = 0;
  
  while (runtime->running) {
    // Execute current window schedule
    execute_window(runtime, current_window);
    
    // Prepare next window (can overlap with execution)
    prepare_window(runtime, current_window + 1);
    
    // Advance window
    current_window++;
    
    // Optional: Recompute schedule if environment changed significantly
    if (deviation_too_large(runtime)) {
      recompute_schedule(runtime, current_window);
    }
  }
}
```

### Information Needed for AOT Scheduling

To design an effective AOT schedule, collect:

#### 1. Hardware Topology
```json
{
  "clusters": [
    {
      "id": 0,
      "cores": [0, 1, 2, 3],
      "l2_cache_kb": 2048,
      "max_frequency_mhz": 2400
    },
    {
      "id": 1,
      "cores": [4, 5, 6, 7],
      "l2_cache_kb": 2048,
      "npu": {
        "type": "risc-v-extension",
        "ops_per_sec": 1000000000000
      }
    }
  ],
  "shared_l3_cache_kb": 8192,
  "memory_bandwidth_gbps": 50
}
```

#### 2. Workload Characteristics
```python
# Profile each model's dispatches
workload_profiles = {
    'model_a': {
        'dispatches': [
            {
                'name': 'conv2d_1',
                'estimated_cycles': 1500000,
                'memory_bytes': 4096000,
                'requires_npu': False,
                'frequency': 30  # Hz - runs 30 times/sec
            },
            # ... more dispatches
        ]
    },
    'model_b': {
        'dispatches': [
            {
                'name': 'npu_matmul',
                'estimated_cycles': 2000000,
                'requires_npu': True,
                'frequency': 10  # Hz
            }
        ]
    }
}
```

#### 3. Dependencies Between Models
```python
dependencies = {
    'model_b.input': 'model_a.output',  # Model B depends on Model A
    'model_c.planning': 'model_b.features',
}
```

#### 4. Real-time Constraints
```python
deadlines = {
    'model_a.predict': 33_000_000,  # 33ms (30 FPS)
    'model_b.plan': 100_000_000,    # 100ms
    'model_c.control': 10_000_000,  # 10ms (critical!)
}
```

### Does This Replace Runtime Scheduling?

**No, it complements it:**

- **Compiler (AOT)**: Computes optimal schedule assuming perfect conditions
- **Runtime**: Executes schedule but adapts to reality (cache misses, interrupts, thermal throttling)

**Hybrid approach works best:**

```
Compiler:
  ├─ Compute global optimal schedule
  ├─ Encode as IR annotations
  └─ Generate async execution structure

Runtime:
  ├─ Read schedule annotations
  ├─ Try to follow schedule (start times, affinity)
  ├─ Adapt when deviations occur
  │  ├─ Adjust priorities
  │  ├─ Reallocate cores
  │  └─ Skip/defer low-priority work
  └─ Monitor and report deviations
```

### Practical Implementation Steps

1. **Phase 1: Profile Workloads**
   - Run each model independently
   - Collect dispatch timing, memory usage
   - Identify NPU requirements
   
2. **Phase 2: Build Scheduling Tool**
   - Implement FJSP algorithm (CP-SAT, genetic algorithm)
   - Input: workload profiles + hardware config
   - Output: schedule with start times + affinity
   
3. **Phase 3: Integrate with Compiler**
   - Add custom pass to annotate IR
   - Use async-external execution model
   - Emit explicit fences/barriers
   
4. **Phase 4: Runtime Support**
   - Custom HAL device reads annotations
   - Enforces schedule constraints
   - Monitors execution vs. plan
   
5. **Phase 5: Adaptive Layer**
   - Handle schedule deviations
   - Recompute on large deviations
   - Continuous improvement via profiling

### Example: Combining Networks with AOT Schedule

```bash
# Compile models with async-external and custom scheduling pass

iree-compile \
    --iree-execution-model=async-external \
    --iree-hal-target-device=local \
    --iree-scheduling-enable-aot=true \
    --iree-scheduling-workload-profile=model_a_profile.json \
    model_a.mlir -o=model_a.vmfb

iree-compile \
    --iree-execution-model=async-external \
    --iree-hal-target-device=local \
    --iree-scheduling-enable-aot=true \
    --iree-scheduling-workload-profile=model_b_profile.json \
    model_b.mlir -o=model_b.vmfb

# Run combined scheduling analysis
iree-aot-schedule \
    --modules model_a.vmfb model_b.vmfb \
    --hardware-config cluster_config.json \
    --dependencies model_deps.json \
    --deadlines deadlines.json \
    --output combined_schedule.json

# Execute with schedule-aware runtime
iree-run-module \
    --device=aot-scheduled \
    --schedule=combined_schedule.json \
    --module=model_a.vmfb \
    --module=model_b.vmfb \
    --function=execute_pipeline
```

### Benefits of AOT Scheduling

1. **Optimal Resource Allocation**: Global view enables better decisions
2. **Deadline Guarantees**: Pre-computed schedule meets timing requirements
3. **Predictable Performance**: Less runtime variability
4. **Lower Runtime Overhead**: No runtime scheduling decisions
5. **Cross-Model Optimization**: Can optimize across model boundaries

### Trade-offs

1. **Reduced Adaptability**: Less responsive to unexpected conditions
2. **Compilation Complexity**: Requires accurate profiles and models
3. **Profile Accuracy**: Schedule only as good as profiles
4. **Hardware Variations**: May not handle heterogeneity well
5. **Development Effort**: Significant compiler/runtime changes needed

### Recommendation for Your Use Case

Given your requirements (robotics, known hardware/software, hard deadlines):

**Use Hybrid AOT + Adaptive Runtime:**

1. **Compiler**: Compute baseline schedule with FJSP for typical scenarios
2. **Runtime**: Execute schedule with adaptive fallback
3. **Monitoring**: Track deviations and trigger replanning
4. **Windowed**: Use sliding windows for long-running execution

This gives you predictability where possible and flexibility where needed.

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
