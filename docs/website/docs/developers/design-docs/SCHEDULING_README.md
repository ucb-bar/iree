# IREE Scheduling Documentation

This directory contains comprehensive documentation about IREE's scheduling architecture and how to implement custom scheduling policies.

## Documents

### [IREE Scheduling Deep Dive](./iree-scheduling-deep-dive.md)
**Comprehensive analysis of IREE's multi-layer scheduling system**

This document provides a complete understanding of how IREE performs scheduling from compiler to runtime:
- Compiler-level scheduling (Stream dialect passes)
- Runtime task system architecture
- CPU core allocation and affinity control
- Concurrent execution and pipelining
- Analysis of existing capabilities
- Key files for scheduling implementation

**Target Audience**: Developers who want to understand IREE's scheduling internals

### [Custom Scheduler Implementation Guide](./custom-scheduler-implementation-guide.md)
**Step-by-step guide for implementing custom scheduling policies**

Practical implementation guide for building a custom Flexible Job Shop Scheduler in IREE:
- Complete code examples and data structures
- NPU resource management
- Priority-based scheduling with deadline tracking
- Reactive scheduling for robotics
- Build configuration and testing
- Integration with IREE's HAL and task system

**Target Audience**: Developers implementing custom schedulers for specialized hardware

## Quick Start

1. **Understanding IREE Scheduling**: Start with [iree-scheduling-deep-dive.md](./iree-scheduling-deep-dive.md) to understand:
   - How IREE schedules work across multiple models
   - How tasks are assigned to CPU cores
   - What extension points exist

2. **Implementing Custom Scheduling**: Follow [custom-scheduler-implementation-guide.md](./custom-scheduler-implementation-guide.md) for:
   - Creating a custom HAL driver
   - Implementing job shop scheduling algorithms
   - Managing specialized resources (e.g., NPU)
   - Building and testing your scheduler

## Use Cases

These documents address scenarios such as:
- **Heterogeneous Computing**: Systems with multiple CPU clusters with different capabilities
- **Specialized Accelerators**: Cores with ISA extensions (e.g., NPU, vector extensions)
- **Real-time Robotics**: Deadline-aware scheduling with priority handling
- **Resource Constrained**: Thermal-aware and power-aware scheduling
- **Multi-Model Orchestration**: Scheduling multiple concurrent ML models

## Key Concepts

### Compiler Scheduling
IREE's compiler performs ahead-of-time scheduling:
- `ScheduleExecution`: Partitions operations into executable regions
- `ScheduleAllocation`: Manages memory allocation timing
- `ScheduleConcurrency`: Optimizes concurrent execution

### Runtime Scheduling
IREE's runtime performs fine-grained task scheduling:
- **Task Executor**: Wavefront-style work-stealing scheduler
- **Workers**: Thread pool mapped to CPU topology
- **Affinity Sets**: Control which cores execute which tasks
- **Topology**: Hardware-aware core assignment

### Concurrent Execution
Multiple execution patterns supported:
- Multiple independent contexts (isolated execution)
- Shared context with internal pipelining
- Fork-join patterns across models
- Timeline-based out-of-order execution

## Contributing

Found an issue or have improvements? Please contribute:
1. File an issue: https://github.com/iree-org/iree/issues
2. Join Discord: https://discord.gg/wEWh6Z9nMU
3. Submit a PR with documentation improvements

## Related Documentation

- [Invocation Execution Model](./invocation-execution-model.md) - Timeline-based execution model
- [HAL Design Docs](.) - Hardware Abstraction Layer documentation
- [IREE Website](https://iree.dev/) - Official documentation

## Questions?

- **IREE Discord**: https://discord.gg/wEWh6Z9nMU (channels: #general, #performance)
- **Mailing List**: iree-technical-discussion@lists.lfaidata.foundation
- **GitHub Issues**: https://github.com/iree-org/iree/issues
