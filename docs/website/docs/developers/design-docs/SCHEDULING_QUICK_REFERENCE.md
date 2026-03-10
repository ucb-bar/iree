# IREE Scheduling Quick Reference Card

## File Locations

### Documentation
- **Main Guide**: `docs/website/docs/developers/design-docs/iree-scheduling-deep-dive.md`
- **Implementation Guide**: `docs/website/docs/developers/design-docs/custom-scheduler-implementation-guide.md`
- **Index**: `docs/website/docs/developers/design-docs/SCHEDULING_README.md`
- **Summary**: `IREE_SCHEDULING_INVESTIGATION_SUMMARY.md`

### Source Code - Runtime Task System
```
runtime/src/iree/task/
├── executor.h/c          ← Main scheduler coordinator
├── worker.h/c            ← Worker thread implementation  
├── task.h/c              ← Task types (dispatch, barrier, etc.)
├── topology.h/c          ← CPU topology detection
├── affinity_set.h        ← Worker affinity bitmasks
└── submission.h/c        ← Task submission API
```

### Source Code - HAL Drivers
```
runtime/src/iree/hal/
├── device.h/c                     ← Device interface
├── drivers/local_task/            ← ⭐ START HERE for custom driver
│   ├── task_device.h/c           ← Device implementation
│   ├── task_queue.h/c            ← Queue implementation
│   └── task_command_buffer.h/c   ← Command recording
└── drivers/local_sync/            ← Simpler reference implementation
```

### Source Code - Compiler Scheduling
```
compiler/src/iree/compiler/Dialect/Stream/Transforms/
├── ScheduleExecution.cpp     ← Partition into executable regions
├── ScheduleAllocation.cpp    ← Schedule memory allocations
└── ScheduleConcurrency.cpp   ← Optimize concurrency
```

## Key Data Structures

### Task
```c
typedef struct iree_task_t {
  iree_task_t* next_task;                    // Intrusive list
  iree_task_scope_t* scope;                  // Error propagation
  iree_task_t* completion_task;              // Dependency
  iree_task_affinity_set_t affinity_set;     // Which workers (64-bit mask)
  iree_atomic_int32_t pending_dependency_count;
  iree_task_type_t type;                     // NOP/CALL/BARRIER/DISPATCH/etc.
} iree_task_t;
```

### Worker
```c
typedef struct iree_task_worker_t {
  iree_atomic_task_slist_t mailbox_slist;    // Posted tasks from coordinator
  iree_task_executor_t* executor;            // Parent executor
  iree_host_size_t worker_index;             // Global worker ID
  iree_task_affinity_set_t worker_bit;       // This worker's bit in sets
  iree_thread_affinity_t ideal_thread_affinity;  // OS-level affinity
  iree_task_affinity_set_t constructive_sharing_mask;  // Cache sharing
} iree_task_worker_t;
```

### Topology
```c
typedef struct iree_task_topology_group_t {
  uint8_t group_index;                       // Group ID (0-63)
  uint32_t processor_index;                  // Logical processor
  iree_task_topology_caches_t caches;        // L1/L2/L3 sizes
  iree_thread_affinity_t ideal_thread_affinity;  // OS thread affinity
  iree_task_topology_group_mask_t constructive_sharing_mask;
} iree_task_topology_group_t;
```

## Common Operations

### Set Task Affinity
```c
// Run on any worker
task->affinity_set = iree_task_affinity_for_any_worker();

// Run on specific worker
task->affinity_set = iree_task_affinity_for_worker(3);

// Run on worker range
task->affinity_set = iree_task_affinity_for_worker_range(0, 4);  // Workers 0-3

// Run on custom set (bitwise)
task->affinity_set = 0b11110000;  // Workers 4-7
```

### Create Custom Topology
```c
iree_task_topology_t topology;
iree_task_topology_initialize(&topology);

// Add groups for each core
for (int i = 0; i < 8; i++) {
  iree_task_topology_group_t* group = &topology.groups[i];
  iree_task_topology_group_initialize(i, group);
  group->processor_index = i;
  
  // Cluster 0: cores 0-3
  // Cluster 1: cores 4-7
  group->constructive_sharing_mask = (i < 4) ? 0b00001111 : 0b11110000;
}

topology.group_count = 8;
```

### Create Executor with Topology
```c
iree_task_executor_options_t options;
iree_task_executor_options_initialize(&options);

iree_task_executor_t* executor;
iree_task_executor_create(options, &topology, allocator, &executor);
```

### Submit Tasks
```c
iree_task_submission_t submission;
iree_task_submission_initialize(&submission);
iree_task_submission_enqueue(&submission, task);
iree_task_executor_submit(executor, &submission);
```

## HAL Device Implementation

### Minimal Device
```c
typedef struct my_custom_device_t {
  iree_hal_resource_t resource;
  iree_task_executor_t* executor;
  // Add your scheduler here
} my_custom_device_t;
```

### Key Function to Override
```c
static iree_status_t my_device_queue_execute(
    iree_hal_device_t* base_device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers,
    iree_hal_buffer_binding_table_t const* binding_tables) {
  
  // 1. Extract metadata from command buffer
  // 2. Run your scheduling algorithm
  // 3. Assign affinity
  // 4. Submit to task executor
  
  return iree_ok_status();
}
```

## Timeline-Based Execution

### Sequential
```c
iree_hal_semaphore_t* timeline;
iree_hal_semaphore_create(device, 0, allocator, &timeline);

// t=0 → t=1
async_invoke(func1, timeline_at(0), timeline_at(1));

// t=1 → t=2  
async_invoke(func2, timeline_at(1), timeline_at(2));

// Wait for completion
iree_hal_semaphore_wait(timeline, 2, iree_infinite_timeout());
```

### Fork-Join
```c
// Fork: both wait on t=0
async_invoke(func1, timeline_at(0), timeline_at(1));
async_invoke(func2, timeline_at(0), timeline_at(2));

// Join: wait on both
iree_hal_fence_t* join = create_fence({timeline_at(1), timeline_at(2)});
iree_hal_fence_wait(join, iree_infinite_timeout());
```

### Pipelined (Multiple Models)
```c
// Model A: t0→t1
async_invoke(modelA, timeline_at(0), timeline_at(1));

// Model B: t1→t2 (waits for A)
async_invoke(modelB, timeline_at(1), timeline_at(2));

// Model C: t2→t3 (waits for B)
async_invoke(modelC, timeline_at(2), timeline_at(3));
```

## Scheduling Patterns for Your Use Case

### NPU Exclusive Access
```c
typedef struct {
  iree_task_affinity_set_t npu_cores;  // 0b11110000 (cores 4-7)
  iree_atomic_int32_t npu_in_use;      // 0=free, 1=busy
  queue_t npu_waiting_jobs;
} npu_manager_t;

bool try_acquire_npu(job_t* job) {
  if (job->needs_npu) {
    if (atomic_compare_exchange(&npu_in_use, 0, 1)) {
      job->affinity_set = npu_cores;
      return true;
    }
    enqueue(&npu_waiting_jobs, job);
    return false;
  }
  return true;  // Doesn't need NPU
}
```

### Dynamic Dispatch (No Fixed Pinning)
```c
// Compute affinity dynamically at dispatch time
iree_task_affinity_set_t compute_dynamic_affinity(
    scheduler_t* scheduler, dispatch_t* dispatch) {
  
  // Get current utilization
  float util0 = get_cluster_utilization(scheduler, 0);
  float util1 = get_cluster_utilization(scheduler, 1);
  
  if (dispatch->needs_npu) {
    return is_npu_available(scheduler) ? 
        scheduler->cluster1_mask : 0;  // Queue if busy
  }
  
  // Dynamic selection: prefer less loaded
  if (util0 < util1 * 0.7f) {
    return scheduler->cluster0_mask;  // Cores 0-3
  } else if (util1 < 0.8f && !npu_in_use(scheduler)) {
    return scheduler->cluster1_mask;  // Cores 4-7
  }
  
  return (util0 <= util1) ? 
      scheduler->cluster0_mask : scheduler->cluster1_mask;
}

// Apply dynamically in scheduling loop
while (job = get_next_ready_job(scheduler)) {
  iree_task_affinity_set_t affinity = 
      compute_dynamic_affinity(scheduler, job);
  
  if (affinity != 0) {
    job->task->affinity_set = affinity;  // DYNAMIC!
    submit_task(scheduler, job);
  } else {
    requeue_for_later(scheduler, job);
  }
}
```

### Fine-Grained Per-Core Selection
```c
// Select N least-loaded cores (not fixed clusters)
iree_task_affinity_set_t select_least_loaded_cores(
    scheduler_t* scheduler, uint32_t num_cores) {
  
  uint32_t load[8];  // Load per core
  for (int i = 0; i < 8; i++) {
    load[i] = get_core_load(scheduler, i);
  }
  
  // Sort by load
  uint32_t sorted[8];
  sort_cores_by_load(sorted, load);
  
  // Build mask with N least-loaded
  iree_task_affinity_set_t mask = 0;
  for (int i = 0; i < num_cores; i++) {
    mask |= (1ULL << sorted[i]);
  }
  return mask;  // e.g., 0b10010101 - not aligned to clusters!
}
```

### Load Balancing Across Clusters
```c
uint32_t choose_cluster(job_t* job) {
  int32_t load0 = atomic_load(&cluster[0].active_jobs);
  int32_t load1 = atomic_load(&cluster[1].active_jobs);
  
  // Thermal throttling
  if (cluster[1].temperature > 85) return 0;
  
  // Choose less loaded
  return (load0 <= load1) ? 0 : 1;
}
```

### Priority Scheduling
```c
void insert_job_by_priority(job_t** queue_head, job_t* job) {
  job_t** pos = queue_head;
  while (*pos && (*pos)->priority >= job->priority) {
    pos = &(*pos)->next;
  }
  job->next = *pos;
  *pos = job;
}
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Scheduling overhead | < 10 μs per decision |
| NPU utilization | > 90% |
| Deadline satisfaction | > 99% for high-priority |
| Lock contention | < 5% of time |
| Work stealing latency | < 100 μs |

## Common Pitfalls

❌ **Don't**: Modify core IREE code
✅ **Do**: Create custom HAL driver

❌ **Don't**: Use global locks on hot path
✅ **Do**: Use per-cluster locks, lock-free when possible

❌ **Don't**: Ignore thermal throttling
✅ **Do**: Monitor temperature and adapt concurrency

❌ **Don't**: Block in scheduler
✅ **Do**: Make scheduling decisions quickly, execute async

❌ **Don't**: Forget to release resources (NPU, memory)
✅ **Do**: Track resource lifetime carefully

## Debugging

### Enable Tracing
```c
#include "iree/base/internal/tracing.h"

IREE_TRACE_ZONE_BEGIN(z0);
IREE_TRACE_ZONE_APPEND_TEXT(z0, "my_scheduler");
// ... scheduling logic ...
IREE_TRACE_PLOT_VALUE_I64("scheduler.ready_jobs", count);
IREE_TRACE_ZONE_END(z0);
```

### Build with Tracy
```bash
cmake -DIREE_ENABLE_RUNTIME_TRACING=ON ...
```

### Inspect Task Flow
```bash
# Enable verbose logging
export IREE_TASK_EXECUTOR_TRACE=1

# Run with Tracy profiler
./your_app
```

## Testing

### Unit Test Template
```c
TEST(CustomSchedulerTest, BasicScheduling) {
  iree_task_executor_t* executor = create_test_executor();
  my_scheduler_t* scheduler = create_test_scheduler(executor);
  
  // Submit test job
  submit_job(scheduler, test_job);
  
  // Verify scheduled correctly
  EXPECT_EQ(get_assigned_cluster(test_job), expected_cluster);
  
  cleanup();
}
```

## Resources

- **IREE Discord**: https://discord.gg/wEWh6Z9nMU (#performance, #general)
- **GitHub**: https://github.com/iree-org/iree
- **Docs**: https://iree.dev/developers/
- **Mailing List**: iree-technical-discussion@lists.lfaidata.foundation

## See Also

- `iree-scheduling-deep-dive.md` - Complete architecture
- `custom-scheduler-implementation-guide.md` - Step-by-step implementation
- `invocation-execution-model.md` - Timeline-based execution
- `runtime/src/iree/task/executor.h` - Detailed comments on task system

---

**Quick Start**: Start with `local_task` driver, add scheduler to device, override `queue_execute()`
