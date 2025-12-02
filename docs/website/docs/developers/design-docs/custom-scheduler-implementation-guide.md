# Custom Scheduler Implementation Guide for IREE

This guide provides practical, step-by-step instructions for implementing a custom Flexible Job Shop Scheduler in IREE for heterogeneous CPU clusters with specialized accelerators (e.g., NPU extensions).

## Scenario

You have:
- **Cluster 0**: 4 general-purpose cores (cores 0-3)
- **Cluster 1**: 4 cores with NPU extension (cores 4-7)
- **Requirement**: Schedule multiple concurrent MLIR models/VMFBs with job shop constraints
- **Goal**: Reactive scheduling for robotics applications

## Implementation Overview

We'll implement a custom HAL driver that extends IREE's `local_task` driver with job shop scheduling capabilities.

```
┌─────────────────────────────────────────────────────────┐
│           Your Application                              │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│         IREE Runtime (VM + HAL)                         │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│   Custom Job Shop Scheduler HAL Driver (YOUR CODE)     │
│                                                         │
│  ┌───────────────────────────────────────────────┐    │
│  │  Job Shop Scheduler                           │    │
│  │  - Priority queues                            │    │
│  │  - Resource allocation                        │    │
│  │  - Deadline tracking                          │    │
│  │  - NPU manager                                │    │
│  └───────────────────────────────────────────────┘    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│         IREE Task Executor (Workers + Tasks)            │
└─────────────────────────────────────────────────────────┘
```

## Step 1: Project Setup

### 1.1 Create Custom Driver Directory

```bash
cd /path/to/iree
mkdir -p runtime/src/iree/hal/drivers/job_shop
cd runtime/src/iree/hal/drivers/job_shop
```

### 1.2 Create Initial Files

Create the following file structure:
```
runtime/src/iree/hal/drivers/job_shop/
├── CMakeLists.txt
├── BUILD.bazel
├── job_shop_driver.h
├── job_shop_driver.c
├── job_shop_device.h
├── job_shop_device.c
├── scheduler.h
├── scheduler.c
├── npu_manager.h
└── npu_manager.c
```

## Step 2: Define the Scheduler Data Structures

### scheduler.h

```c
// runtime/src/iree/hal/drivers/job_shop/scheduler.h

#ifndef IREE_HAL_DRIVERS_JOB_SHOP_SCHEDULER_H_
#define IREE_HAL_DRIVERS_JOB_SHOP_SCHEDULER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/task/affinity_set.h"

#ifdef __cplusplus
extern "C" {
#endif

// Job metadata extracted from command buffers
typedef struct iree_job_metadata_t {
  uint32_t job_id;              // Unique job identifier
  uint32_t operation_id;        // Operation within job sequence
  uint32_t priority;            // 0 (lowest) - 255 (highest)
  uint64_t deadline_ns;         // Absolute deadline (0 = no deadline)
  uint32_t estimated_duration_ns; // Estimated execution time
  
  // Resource requirements
  bool requires_npu;            // Needs NPU access
  uint32_t memory_mb;           // Estimated memory usage
  uint32_t compute_intensity;   // 0-100 scale
} iree_job_metadata_t;

// Job state in the scheduler
typedef enum iree_job_state_e {
  IREE_JOB_STATE_PENDING,       // Waiting for dependencies
  IREE_JOB_STATE_READY,         // Ready to execute
  IREE_JOB_STATE_SCHEDULED,     // Scheduled to a cluster
  IREE_JOB_STATE_RUNNING,       // Currently executing
  IREE_JOB_STATE_COMPLETED,     // Finished successfully
  IREE_JOB_STATE_FAILED,        // Failed with error
} iree_job_state_t;

// A job in the scheduler
typedef struct iree_scheduler_job_t {
  iree_job_metadata_t metadata;
  iree_job_state_t state;
  
  // Command buffer to execute
  iree_hal_command_buffer_t* command_buffer;
  
  // Synchronization
  iree_hal_semaphore_list_t wait_semaphores;
  iree_hal_semaphore_list_t signal_semaphores;
  
  // Scheduling info
  uint64_t submit_time_ns;
  uint64_t start_time_ns;
  uint64_t complete_time_ns;
  iree_task_affinity_set_t assigned_cores;
  
  // Intrusive list node
  struct iree_scheduler_job_t* next;
} iree_scheduler_job_t;

// Cluster resource state
typedef struct iree_cluster_state_t {
  uint32_t cluster_id;
  iree_task_affinity_set_t core_mask;     // Which cores in cluster
  iree_atomic_int32_t active_jobs;        // Number of running jobs
  uint32_t max_concurrent_jobs;           // Concurrency limit
  
  // Statistics
  uint64_t total_jobs_executed;
  uint64_t total_execution_time_ns;
  uint32_t current_temperature;           // For thermal throttling
} iree_cluster_state_t;

// Job shop scheduler
typedef struct iree_job_shop_scheduler_t {
  iree_allocator_t allocator;
  
  // Job queues (priority-ordered)
  iree_slim_mutex_t queue_mutex;
  iree_scheduler_job_t* ready_queue_head;
  iree_scheduler_job_t* pending_queue_head;
  
  // Resource state
  iree_cluster_state_t clusters[2];       // 2 clusters
  
  // NPU management (owned)
  struct iree_npu_manager_t* npu_manager;
  
  // Task executor
  iree_task_executor_t* task_executor;
  
  // Scheduling policy
  enum {
    IREE_SCHED_POLICY_FIFO,
    IREE_SCHED_POLICY_PRIORITY,
    IREE_SCHED_POLICY_DEADLINE,
    IREE_SCHED_POLICY_SHORTEST_JOB,
  } scheduling_policy;
  
  // Performance telemetry
  struct {
    uint64_t jobs_scheduled;
    uint64_t jobs_completed;
    uint64_t deadline_misses;
    uint64_t average_latency_ns;
  } stats;
} iree_job_shop_scheduler_t;

// Initialize the scheduler
iree_status_t iree_job_shop_scheduler_initialize(
    iree_task_executor_t* task_executor,
    iree_allocator_t allocator,
    iree_job_shop_scheduler_t* out_scheduler);

// Deinitialize the scheduler
void iree_job_shop_scheduler_deinitialize(
    iree_job_shop_scheduler_t* scheduler);

// Submit a job to the scheduler
iree_status_t iree_job_shop_scheduler_submit(
    iree_job_shop_scheduler_t* scheduler,
    iree_job_metadata_t metadata,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores);

// Run the scheduling algorithm and dispatch ready jobs
iree_status_t iree_job_shop_scheduler_schedule(
    iree_job_shop_scheduler_t* scheduler);

// Update scheduler state based on system telemetry
void iree_job_shop_scheduler_update_telemetry(
    iree_job_shop_scheduler_t* scheduler,
    uint32_t cluster_id,
    uint32_t temperature,
    uint32_t memory_pressure_mb);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_JOB_SHOP_SCHEDULER_H_
```

### npu_manager.h

```c
// runtime/src/iree/hal/drivers/job_shop/npu_manager.h

#ifndef IREE_HAL_DRIVERS_JOB_SHOP_NPU_MANAGER_H_
#define IREE_HAL_DRIVERS_JOB_SHOP_NPU_MANAGER_H_

#include "iree/base/api.h"
#include "iree/task/affinity_set.h"

#ifdef __cplusplus
extern "C" {
#endif

// NPU resource manager
typedef struct iree_npu_manager_t {
  // Which cores have NPU access
  iree_task_affinity_set_t npu_core_mask;  // e.g., 0b11110000 for cores 4-7
  
  // Exclusive access control
  iree_atomic_int32_t npu_in_use;          // 0 = free, 1 = in use
  
  // Pending NPU jobs
  iree_slim_mutex_t queue_mutex;
  struct iree_scheduler_job_t* npu_queue_head;
  
  // Statistics
  uint64_t npu_acquisitions;
  uint64_t npu_wait_time_ns;
  uint64_t total_npu_execution_time_ns;
} iree_npu_manager_t;

// Initialize NPU manager
iree_status_t iree_npu_manager_initialize(
    iree_task_affinity_set_t npu_core_mask,
    iree_allocator_t allocator,
    iree_npu_manager_t** out_manager);

// Deinitialize NPU manager
void iree_npu_manager_deinitialize(iree_npu_manager_t* manager);

// Try to acquire NPU for a job
// Returns true if acquired, false if NPU is busy
bool iree_npu_manager_try_acquire(
    iree_npu_manager_t* manager,
    struct iree_scheduler_job_t* job);

// Release NPU after job completion
void iree_npu_manager_release(
    iree_npu_manager_t* manager,
    struct iree_scheduler_job_t* job);

// Queue a job that needs NPU
iree_status_t iree_npu_manager_enqueue(
    iree_npu_manager_t* manager,
    struct iree_scheduler_job_t* job);

// Try to schedule a waiting NPU job
struct iree_scheduler_job_t* iree_npu_manager_dequeue_ready(
    iree_npu_manager_t* manager);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_JOB_SHOP_NPU_MANAGER_H_
```

## Step 3: Implement the Core Scheduler

### scheduler.c (Key Functions)

```c
// runtime/src/iree/hal/drivers/job_shop/scheduler.c

#include "iree/hal/drivers/job_shop/scheduler.h"
#include "iree/hal/drivers/job_shop/npu_manager.h"
#include "iree/base/internal/time.h"

// Initialize scheduler
iree_status_t iree_job_shop_scheduler_initialize(
    iree_task_executor_t* task_executor,
    iree_allocator_t allocator,
    iree_job_shop_scheduler_t* out_scheduler) {
  
  memset(out_scheduler, 0, sizeof(*out_scheduler));
  out_scheduler->allocator = allocator;
  out_scheduler->task_executor = task_executor;
  out_scheduler->scheduling_policy = IREE_SCHED_POLICY_PRIORITY;
  
  iree_slim_mutex_initialize(&out_scheduler->queue_mutex);
  
  // Initialize cluster 0 (general purpose)
  out_scheduler->clusters[0].cluster_id = 0;
  out_scheduler->clusters[0].core_mask = 0b00001111;  // Cores 0-3
  out_scheduler->clusters[0].max_concurrent_jobs = 4;
  iree_atomic_store_int32(&out_scheduler->clusters[0].active_jobs, 0,
                          iree_memory_order_relaxed);
  
  // Initialize cluster 1 (with NPU)
  out_scheduler->clusters[1].cluster_id = 1;
  out_scheduler->clusters[1].core_mask = 0b11110000;  // Cores 4-7
  out_scheduler->clusters[1].max_concurrent_jobs = 4;
  iree_atomic_store_int32(&out_scheduler->clusters[1].active_jobs, 0,
                          iree_memory_order_relaxed);
  
  // Initialize NPU manager
  return iree_npu_manager_initialize(
      0b11110000,  // Cores 4-7 have NPU
      allocator,
      &out_scheduler->npu_manager);
}

// Submit a job
iree_status_t iree_job_shop_scheduler_submit(
    iree_job_shop_scheduler_t* scheduler,
    iree_job_metadata_t metadata,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores) {
  
  // Allocate job
  iree_scheduler_job_t* job = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      scheduler->allocator, sizeof(*job), (void**)&job));
  
  memset(job, 0, sizeof(*job));
  job->metadata = metadata;
  job->state = IREE_JOB_STATE_PENDING;
  job->command_buffer = command_buffer;
  job->wait_semaphores = wait_semaphores;
  job->signal_semaphores = signal_semaphores;
  job->submit_time_ns = iree_time_now();
  
  iree_hal_command_buffer_retain(command_buffer);
  
  // Add to appropriate queue
  iree_slim_mutex_lock(&scheduler->queue_mutex);
  
  // Check if job is immediately ready or needs to wait
  bool is_ready = true;
  for (iree_host_size_t i = 0; i < wait_semaphores.count; ++i) {
    uint64_t value;
    iree_hal_semaphore_query(wait_semaphores.semaphores[i], &value);
    if (value < wait_semaphores.payload_values[i]) {
      is_ready = false;
      break;
    }
  }
  
  if (is_ready) {
    // Insert into ready queue (priority-sorted)
    job->state = IREE_JOB_STATE_READY;
    insert_job_sorted(&scheduler->ready_queue_head, job);
  } else {
    // Add to pending queue
    insert_job_sorted(&scheduler->pending_queue_head, job);
  }
  
  scheduler->stats.jobs_scheduled++;
  
  iree_slim_mutex_unlock(&scheduler->queue_mutex);
  
  // Trigger scheduling
  return iree_job_shop_scheduler_schedule(scheduler);
}

// Main scheduling algorithm
iree_status_t iree_job_shop_scheduler_schedule(
    iree_job_shop_scheduler_t* scheduler) {
  
  iree_slim_mutex_lock(&scheduler->queue_mutex);
  
  // Move pending jobs to ready queue if dependencies met
  update_pending_jobs(scheduler);
  
  // Schedule ready jobs based on policy
  iree_scheduler_job_t* job = scheduler->ready_queue_head;
  while (job != NULL) {
    iree_scheduler_job_t* next = job->next;
    
    // Determine target cluster
    uint32_t target_cluster;
    iree_task_affinity_set_t assigned_cores;
    
    if (job->metadata.requires_npu) {
      // Try to acquire NPU
      if (iree_npu_manager_try_acquire(scheduler->npu_manager, job)) {
        target_cluster = 1;
        assigned_cores = scheduler->clusters[1].core_mask;
      } else {
        // NPU busy, queue the job
        iree_npu_manager_enqueue(scheduler->npu_manager, job);
        remove_from_queue(&scheduler->ready_queue_head, job);
        job = next;
        continue;
      }
    } else {
      // Choose cluster based on load balancing
      target_cluster = choose_cluster_for_job(scheduler, job);
      assigned_cores = scheduler->clusters[target_cluster].core_mask;
    }
    
    // Check if cluster has capacity
    int32_t active = iree_atomic_load_int32(
        &scheduler->clusters[target_cluster].active_jobs,
        iree_memory_order_relaxed);
    
    if (active >= scheduler->clusters[target_cluster].max_concurrent_jobs) {
      // Cluster at capacity, try next job
      job = next;
      continue;
    }
    
    // Dispatch job to task executor
    iree_status_t status = dispatch_job_to_executor(
        scheduler, job, assigned_cores);
    
    if (iree_status_is_ok(status)) {
      // Mark as scheduled
      job->state = IREE_JOB_STATE_SCHEDULED;
      job->assigned_cores = assigned_cores;
      job->start_time_ns = iree_time_now();
      
      // Increment active jobs
      iree_atomic_fetch_add_int32(
          &scheduler->clusters[target_cluster].active_jobs, 1,
          iree_memory_order_relaxed);
      
      // Remove from ready queue
      remove_from_queue(&scheduler->ready_queue_head, job);
    }
    
    job = next;
  }
  
  // Check for NPU jobs that can now run
  schedule_npu_jobs(scheduler);
  
  iree_slim_mutex_unlock(&scheduler->queue_mutex);
  
  return iree_ok_status();
}

// Choose cluster for non-NPU job (load balancing)
static uint32_t choose_cluster_for_job(
    iree_job_shop_scheduler_t* scheduler,
    iree_scheduler_job_t* job) {
  
  // Simple load balancing: choose cluster with fewer active jobs
  int32_t cluster0_load = iree_atomic_load_int32(
      &scheduler->clusters[0].active_jobs, iree_memory_order_relaxed);
  int32_t cluster1_load = iree_atomic_load_int32(
      &scheduler->clusters[1].active_jobs, iree_memory_order_relaxed);
  
  // Consider temperature (thermal throttling)
  if (scheduler->clusters[1].current_temperature > 85) {
    // Cluster 1 too hot, prefer cluster 0
    return 0;
  }
  
  // Choose less loaded cluster
  return (cluster0_load <= cluster1_load) ? 0 : 1;
}

// Reactive scheduling: update based on telemetry
void iree_job_shop_scheduler_update_telemetry(
    iree_job_shop_scheduler_t* scheduler,
    uint32_t cluster_id,
    uint32_t temperature,
    uint32_t memory_pressure_mb) {
  
  if (cluster_id >= 2) return;
  
  scheduler->clusters[cluster_id].current_temperature = temperature;
  
  // Thermal throttling
  if (temperature > 90) {
    // Reduce concurrency on hot cluster
    scheduler->clusters[cluster_id].max_concurrent_jobs = 2;
  } else if (temperature < 70) {
    // Restore normal concurrency
    scheduler->clusters[cluster_id].max_concurrent_jobs = 4;
  }
  
  // Trigger rescheduling if needed
  iree_job_shop_scheduler_schedule(scheduler);
}
```

## Step 4: Implement the HAL Device

### job_shop_device.h

```c
// runtime/src/iree/hal/drivers/job_shop/job_shop_device.h

#ifndef IREE_HAL_DRIVERS_JOB_SHOP_DEVICE_H_
#define IREE_HAL_DRIVERS_JOB_SHOP_DEVICE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/job_shop/scheduler.h"
#include "iree/task/executor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_job_shop_device_t {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;
  iree_allocator_t host_allocator;
  
  // Task executor (from local_task)
  iree_task_executor_t* executor;
  
  // Custom scheduler
  iree_job_shop_scheduler_t scheduler;
  
  // HAL resources
  iree_hal_allocator_t* device_allocator;
} iree_hal_job_shop_device_t;

iree_status_t iree_hal_job_shop_device_create(
    iree_string_view_t identifier,
    iree_task_executor_t* executor,
    iree_allocator_t host_allocator,
    iree_hal_device_t** out_device);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_JOB_SHOP_DEVICE_H_
```

### job_shop_device.c (Key Function)

```c
// runtime/src/iree/hal/drivers/job_shop/job_shop_device.c

#include "iree/hal/drivers/job_shop/job_shop_device.h"

// Implement queue_execute - this is where scheduling happens
static iree_status_t iree_hal_job_shop_device_queue_execute(
    iree_hal_device_t* base_device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers,
    iree_hal_buffer_binding_table_t const* binding_tables) {
  
  iree_hal_job_shop_device_t* device = 
      (iree_hal_job_shop_device_t*)base_device;
  
  // Extract job metadata from command buffer
  // In production, this might come from custom attributes in the IR
  iree_job_metadata_t metadata = {
    .job_id = extract_job_id(command_buffers[0]),
    .operation_id = extract_operation_id(command_buffers[0]),
    .priority = extract_priority(command_buffers[0]),
    .deadline_ns = extract_deadline(command_buffers[0]),
    .requires_npu = detect_npu_requirement(command_buffers[0]),
    .estimated_duration_ns = estimate_duration(command_buffers[0]),
  };
  
  // Submit to scheduler
  return iree_job_shop_scheduler_submit(
      &device->scheduler,
      metadata,
      command_buffers[0],
      wait_semaphore_list,
      signal_semaphore_list);
}
```

## Step 5: Build Configuration

### CMakeLists.txt

```cmake
# runtime/src/iree/hal/drivers/job_shop/CMakeLists.txt

iree_cc_library(
  NAME
    job_shop_driver
  HDRS
    "job_shop_device.h"
    "job_shop_driver.h"
    "npu_manager.h"
    "scheduler.h"
  SRCS
    "job_shop_device.c"
    "job_shop_driver.c"
    "npu_manager.c"
    "scheduler.c"
  DEPS
    ::npu_manager
    ::scheduler
    iree::base
    iree::base::internal
    iree::hal
    iree::hal::local::executable_loader
    iree::task
  PUBLIC
)

iree_cc_library(
  NAME
    scheduler
  HDRS
    "scheduler.h"
  SRCS
    "scheduler.c"
  DEPS
    ::npu_manager
    iree::base
    iree::base::internal::time
    iree::hal
    iree::task
  PUBLIC
)

iree_cc_library(
  NAME
    npu_manager
  HDRS
    "npu_manager.h"
  SRCS
    "npu_manager.c"
  DEPS
    iree::base
    iree::base::internal::synchronization
    iree::task
  PUBLIC
)
```

## Step 6: Usage Example

### Application Code

```c
// your_robot_application.c

#include "iree/hal/drivers/job_shop/job_shop_driver.h"
#include "iree/runtime/api.h"

int main(int argc, char** argv) {
  iree_allocator_t allocator = iree_allocator_system();
  
  // 1. Create custom topology for your hardware
  iree_task_topology_t topology;
  iree_task_topology_initialize(&topology);
  
  // Cluster 0: Cores 0-3 (general)
  for (int i = 0; i < 4; i++) {
    iree_task_topology_group_t* group = &topology.groups[i];
    iree_task_topology_group_initialize(i, group);
    group->processor_index = i;
    group->constructive_sharing_mask = 0b00001111;
  }
  
  // Cluster 1: Cores 4-7 (with NPU)
  for (int i = 4; i < 8; i++) {
    iree_task_topology_group_t* group = &topology.groups[i];
    iree_task_topology_group_initialize(i, group);
    group->processor_index = i;
    group->constructive_sharing_mask = 0b11110000;
  }
  
  topology.group_count = 8;
  
  // 2. Create task executor
  iree_task_executor_t* executor = NULL;
  iree_task_executor_options_t executor_options;
  iree_task_executor_options_initialize(&executor_options);
  iree_task_executor_create(executor_options, &topology, 
                           allocator, &executor);
  
  // 3. Create custom HAL device
  iree_hal_device_t* device = NULL;
  iree_hal_job_shop_device_create(
      iree_make_cstring_view("job-shop"),
      executor,
      allocator,
      &device);
  
  // 4. Create IREE runtime instance
  iree_runtime_instance_t* instance = NULL;
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_create(
      &instance_options, allocator, &instance);
  
  // 5. Load multiple models
  iree_runtime_session_t* session = NULL;
  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  iree_runtime_session_create_with_device(
      instance, &session_options, device,
      allocator, &session);
  
  // Load models
  iree_runtime_session_append_bytecode_module_from_file(
      session, "model1.vmfb");
  iree_runtime_session_append_bytecode_module_from_file(
      session, "model2.vmfb");
  iree_runtime_session_append_bytecode_module_from_file(
      session, "model3_npu.vmfb");  // This one needs NPU
  
  // 6. Execute with pipelining
  iree_hal_semaphore_t* timeline;
  iree_hal_semaphore_create(device, 0, allocator, &timeline);
  
  // Submit jobs to timeline
  iree_hal_fence_t* fence0 = create_fence(timeline, 0);
  iree_hal_fence_t* fence1 = create_fence(timeline, 1);
  iree_hal_fence_t* fence2 = create_fence(timeline, 2);
  
  // Job 1: High priority perception (no NPU needed)
  iree_runtime_call_t call1;
  setup_call(&call1, session, "model1", "perceive");
  set_call_metadata(&call1, 
                    /*job_id=*/1, 
                    /*priority=*/200,
                    /*deadline=*/10000000);  // 10ms
  iree_runtime_call_invoke_async(&call1, fence0, fence1);
  
  // Job 2: NPU-accelerated planning
  iree_runtime_call_t call2;
  setup_call(&call2, session, "model3_npu", "plan");
  set_call_metadata(&call2,
                    /*job_id=*/2,
                    /*priority=*/150,
                    /*requires_npu=*/true);
  iree_runtime_call_invoke_async(&call2, fence1, fence2);
  
  // 7. Monitor and adapt
  while (robot_is_running()) {
    // Get telemetry
    uint32_t temp0 = read_cpu_temperature(0);
    uint32_t temp1 = read_cpu_temperature(1);
    
    // Update scheduler
    iree_job_shop_scheduler_update_telemetry(
        &device->scheduler, 0, temp0, 0);
    iree_job_shop_scheduler_update_telemetry(
        &device->scheduler, 1, temp1, 0);
    
    // Wait for completion or timeout
    iree_hal_fence_wait(fence2, iree_make_deadline(1000000));
    
    // Process results...
  }
  
  // Cleanup
  iree_task_executor_release(executor);
  iree_hal_device_release(device);
  iree_runtime_session_release(session);
  iree_runtime_instance_release(instance);
  
  return 0;
}
```

## Step 7: Testing

### Unit Test

```c
// runtime/src/iree/hal/drivers/job_shop/scheduler_test.cc

#include "iree/hal/drivers/job_shop/scheduler.h"
#include "iree/testing/gtest.h"

TEST(SchedulerTest, BasicScheduling) {
  iree_allocator_t allocator = iree_allocator_system();
  
  // Create executor
  iree_task_topology_t topology;
  create_test_topology(&topology);
  iree_task_executor_t* executor;
  create_test_executor(&topology, &executor);
  
  // Create scheduler
  iree_job_shop_scheduler_t scheduler;
  IREE_ASSERT_OK(iree_job_shop_scheduler_initialize(
      executor, allocator, &scheduler));
  
  // Submit test job
  iree_job_metadata_t metadata = {
    .job_id = 1,
    .priority = 100,
    .requires_npu = false,
  };
  
  // Submit and schedule
  IREE_ASSERT_OK(iree_job_shop_scheduler_submit(
      &scheduler, metadata, test_command_buffer,
      empty_semaphore_list, empty_semaphore_list));
  
  // Verify job was scheduled
  EXPECT_EQ(scheduler.stats.jobs_scheduled, 1);
  
  // Cleanup
  iree_job_shop_scheduler_deinitialize(&scheduler);
  iree_task_executor_release(executor);
}

TEST(SchedulerTest, NPUExclusivity) {
  // Test that only one NPU job runs at a time
  // ...
}

TEST(SchedulerTest, PriorityOrdering) {
  // Test that higher priority jobs run first
  // ...
}

TEST(SchedulerTest, ThermalThrottling) {
  // Test that high temperature reduces concurrency
  // ...
}
```

## Step 8: Performance Monitoring

### Add Tracy Integration

```c
// In scheduler.c, add tracing:

#include "iree/base/internal/tracing.h"

iree_status_t iree_job_shop_scheduler_schedule(
    iree_job_shop_scheduler_t* scheduler) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, "job_shop_schedule");
  
  // ... scheduling logic ...
  
  // Emit metrics
  IREE_TRACE_PLOT_VALUE_I64("scheduler.ready_jobs", 
                            count_ready_jobs(scheduler));
  IREE_TRACE_PLOT_VALUE_I64("scheduler.cluster0_load",
                            scheduler->clusters[0].active_jobs);
  IREE_TRACE_PLOT_VALUE_I64("scheduler.cluster1_load",
                            scheduler->clusters[1].active_jobs);
  
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
```

## Summary

This implementation provides:

✅ **Custom scheduler** with job shop algorithm
✅ **NPU resource management** with exclusive access
✅ **Priority-based scheduling** with deadline tracking
✅ **Reactive scheduling** based on temperature/telemetry
✅ **Multi-cluster support** with load balancing
✅ **Concurrent model execution** via IREE timelines

## Next Steps

1. **Complete the implementation** by filling in the helper functions
2. **Add metadata extraction** from command buffers (or compile-time annotations)
3. **Implement advanced scheduling algorithms** (e.g., genetic algorithms, constraint programming)
4. **Add profiling and visualization** to tune performance
5. **Test on real hardware** with your robotics workload

## Resources

- IREE source: https://github.com/iree-org/iree
- Task system: `runtime/src/iree/task/`
- HAL drivers: `runtime/src/iree/hal/drivers/`
- Local task driver: `runtime/src/iree/hal/drivers/local_task/`
