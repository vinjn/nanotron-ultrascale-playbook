# 【DeepEP】使用Cursor+Mermaid阅读代码（二）函数调用关系

**Author:** shifang

**Date:** 2025-06-28

**Link:** https://zhuanlan.zhihu.com/p/1915098043679769014

## 整体介绍

DeepSeek 在2025年初发布了DeepEP（Deep Expert Parallelism）

[deepseek-ai/DeepEP: DeepEP: an efficient expert-parallel communication library](https://link.zhihu.com/?target=https%3A//github.com/deepseek-ai/DeepEP/tree/main)

DeepEP 是一个专注于分布式机器学习的库，主要用于优化[Mixture of Experts](https://zhida.zhihu.com/search?content_id=258724382&content_type=Article&match_order=1&q=Mixture+of+Experts&zhida_source=entity) (MoE)模型的通信。

该库包含以下主要组件：

### Buffer管理

核心类处理GPU内存管理、进程间通信和设备同步

### 通信类型

-   节点内通信（Intranode）
-   节点间通信（Internode）
-   低延迟通信（Low-latency）

### [CUDA内核](https://zhida.zhihu.com/search?content_id=258724382&content_type=Article&match_order=1&q=CUDA%E5%86%85%E6%A0%B8&zhida_source=entity)

专门设计的并行计算内核

## 主要类和组件

### [Buffer类](https://zhida.zhihu.com/search?content_id=258724382&content_type=Article&match_order=1&q=Buffer%E7%B1%BB&zhida_source=entity)

核心类，管理GPU内存和通信。主要职责：

-   分配和管理GPU内存
-   处理[NVLink](https://zhida.zhihu.com/search?content_id=258724382&content_type=Article&match_order=1&q=NVLink&zhida_source=entity)通信（节点内）
-   处理NVSHMEM通信（节点间）
-   协调任务调度

### Config类

配置系统资源和参数：

-   指定SM（Streaming Multiprocessor）数量
-   设置最大数据传输大小
-   计算缓冲区大小提示

### [EventHandle类](https://zhida.zhihu.com/search?content_id=258724382&content_type=Article&match_order=1&q=EventHandle%E7%B1%BB&zhida_source=entity)

管理CUDA事件和流同步：

-   记录事件
-   等待事件完成
-   流之间的同步

## 函数调用关系

### 主要命名空间

（1）deep\_ep::intranode

节点内通信，处理同一节点上GPU之间的数据交换：

-   barrier: 同步屏障
-   notify\_dispatch: 通知分发操作
-   dispatch: 分发数据
-   cached\_notify\_combine: 通知组合操作
-   combine: 组合数据

（2）deep\_ep::internode

节点间通信，处理不同节点间的数据交换：

-   get\_unique\_id: 获取唯一标识符
-   init: 初始化通信
-   alloc: 分配内存
-   free: 释放内存
-   barrier: 同步屏障
-   finalize: 终止通信
-   get\_source\_meta\_bytes: 获取元数据大小
-   get\_dispatch\_layout: 获取分发布局
-   notify\_dispatch: 通知分发操作
-   dispatch: 分发数据
-   cached\_notify: 通知缓存操作
-   combine: 组合数据

（3）deep\_ep::internode\_ll

低延迟节点间通信：

-   clean\_low\_latency\_buffer: 清理低延迟缓冲区
-   dispatch: 低延迟分发
-   combine: 低延迟组合

### Buffer类方法调用关系

（1）初始化和销毁:

-   Buffer::Buffer → 分配内存 → 设置IPC句柄
-   Buffer::~Buffer → 同步 → 释放资源 → internode::finalize

（2）同步操作:

-   Buffer::sync → 处理IPC句柄 → internode::init

（3）分发操作:

-   Buffer::get\_dispatch\_layout → 准备分发布局
-   Buffer::intranode\_dispatch → intranode::notify\_dispatch → intranode::dispatch
-   Buffer::internode\_dispatch → internode::notify\_dispatch → internode::dispatch
-   Buffer::low\_latency\_dispatch → internode\_ll::dispatch

（4）组合操作:

-   Buffer::intranode\_combine → intranode::cached\_notify\_combine → intranode::combine
-   Buffer::internode\_combine → internode::cached\_notify → internode::combine
-   Buffer::low\_latency\_combine → internode\_ll::combine

（5）低延迟操作:

-   Buffer::clean\_low\_latency\_buffer → internode\_ll::clean\_low\_latency\_buffer
-   Buffer::get\_next\_low\_latency\_combine\_buffer → 为下一次组合准备缓冲区

## 关键数据流程

（1）token分发流程

-   准备分发布局（get\_dispatch\_layout）
-   根据通信类型选择分发方法（节点内/节点间/低延迟）
-   进行数据分发和任务调度

（3）token组合流程

-   发送方通知接收方准备接收
-   进行数据传输
-   接收方组合数据

（3）低延迟模式

-   使用特殊缓冲区布局（LowLatencyLayout）
-   通过零拷贝和异步操作减少延迟

## 内存管理

（1）NVLink缓冲区:

-   用于节点内GPU通信
-   通过CUDA IPC机制共享

（2）NVSHMEM缓冲区:

-   用于节点间通信
-   使用NVSHMEM库进行RDMA通信

（3）工作空间:

-   提供临时存储
-   用于计算和通信重叠

## 总结

DeepEP库的csrc目录实现了高效的分布式MoE通信机制，通过以下技术：

1.  精细的内存管理和缓冲区设计
2.  多层次的通信策略（节点内、节点间、低延迟）
3.  高度优化的CUDA内核
4.  事件同步和流管理

该库的核心是Buffer类，它协调各种通信类型和内存资源，确保高效的数据传输和模型并行计算。