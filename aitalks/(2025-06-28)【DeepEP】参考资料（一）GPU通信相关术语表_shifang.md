# 【DeepEP】参考资料（一）GPU通信相关术语表

**Author:** shifang

**Date:** 2025-06-28

**Link:** https://zhuanlan.zhihu.com/p/1919317452581475821

## **[DeepSeek开源Day2：DeepEP 原理，NVSHMEM实现All2All通信！](https://link.zhihu.com/?target=https%3A//www.youtube.com/watch%3Fv%3D5CZ07C5p5MU)**

![](https://pic3.zhimg.com/v2-75aa3d3828ce60a246a9ddcf5a73c0f2_1440w.jpg)

## 硬件与架构

-   **[GPU](https://zhida.zhihu.com/search?content_id=259315297&content_type=Article&match_order=1&q=GPU&zhida_source=entity)（图形处理单元）**  
    一种专为并行计算设计的专用处理器，广泛应用于深度学习和科学计算领域。
-   **SM（流式多处理器）/ [CU](https://zhida.zhihu.com/search?content_id=259315297&content_type=Article&match_order=1&q=CU&zhida_source=entity)（计算单元）**  
    GPU内部的并行处理单元，负责执行线程块（block）或工作组（work group）。
-   **CUDA Core / Processing Element（CUDA核心/处理单元）**  
    GPU中最小的硬件执行单元，负责执行单个线程。
-   **Global Memory（全局内存/GPU内存）**  
    所有SM/CU都可访问的大容量、相对较慢的内存。
-   **[Shared Memory](https://zhida.zhihu.com/search?content_id=259315297&content_type=Article&match_order=1&q=Shared+Memory&zhida_source=entity)（共享内存/本地内存）**  
    每个SM/CU内部的高速内存，仅同一线程块或工作组内的线程可访问。
-   **[PCIe](https://zhida.zhihu.com/search?content_id=259315297&content_type=Article&match_order=1&q=PCIe&zhida_source=entity)（外围组件互连高速总线）**  
    CPU与GPU、设备与设备间主流的高速数据传输总线。
-   **[NVLink](https://zhida.zhihu.com/search?content_id=259315297&content_type=Article&match_order=1&q=NVLink&zhida_source=entity)**  
    NVIDIA的高速GPU互联技术，比PCIe拥有更高带宽，用于GPU间通信。
-   **[NIC](https://zhida.zhihu.com/search?content_id=259315297&content_type=Article&match_order=1&q=NIC&zhida_source=entity)（网络接口卡）**  
    提供网络连接的硬件组件，常用于跨节点GPU间通信。

## 通信协议与库

-   **NCCL（NVIDIA集体通信库）**  
    NVIDIA优化的多GPU集体通信库，支持all-reduce、broadcast、all-gather等操作。
-   **[MPI](https://zhida.zhihu.com/search?content_id=259315297&content_type=Article&match_order=1&q=MPI&zhida_source=entity)（消息传递接口）**  
    分布式并行通信的标准协议，支持点对点和集体操作，部分实现支持GPU直接通信。
-   **Gloo**  
    Facebook开源的集体通信库，支持多种后端和硬件，包括GPU通信。
-   **[GPUDirect](https://zhida.zhihu.com/search?content_id=259315297&content_type=Article&match_order=1&q=GPUDirect&zhida_source=entity)**  
    NVIDIA技术，使GPU、NIC或存储设备间可直接通信，绕过CPU和主机内存。
-   **GPUDirect Peer-to-Peer (P2P)**  
    允许同一主机内多个GPU间直接交换数据，无需经过主机内存。
-   **GPUDirect RDMA**  
    使GPU可通过网络直接与远程设备通信，绕过主机内存和CPU。

## 通信机制与操作

-   **点对点通信（Point-to-Point, P2P）**  
    两个处理单元（如GPU）间的直接数据传输。
-   **集体通信（Collective Communication）**  
    多个处理单元参与的数据交换操作，如all-reduce、broadcast、all-gather、reduce-scatter等。
-   **All-Reduce（全归约）**  
    每个GPU先本地计算，然后全局归约（如求和）并同步结果至所有GPU。
-   **Broadcast（广播）**  
    一个GPU的数据广播到所有其他GPU。
-   **All-Gather（全收集）**  
    所有GPU互相收集数据，最终每个GPU都拥有完整数据集。
-   **Reduce-Scatter（归约-分散）**  
    先归约（如求和），再分散，每个GPU获得结果的一部分。
-   **Kernel Initiated Communication (KI)（内核发起通信）**  
    由GPU内核直接发起的通信操作，无需CPU参与。
-   **Stream Triggered Communication (ST)（流触发通信）**  
    通过GPU流异步调度通信，实现通信与计算重叠。

## 内存与寻址

-   **Pinned Memory（锁页内存）**  
    被锁定、不会被操作系统换出的主机内存，可加快CPU与GPU间数据传输。
-   **Unified Virtual Addressing ([UVA](https://zhida.zhihu.com/search?content_id=259315297&content_type=Article&match_order=1&q=UVA&zhida_source=entity))（统一虚拟寻址）**  
    为主机和GPU内存提供统一虚拟地址空间，简化内存管理。
-   **Unified Virtual Memory ([UVM](https://zhida.zhihu.com/search?content_id=259315297&content_type=Article&match_order=1&q=UVM&zhida_source=entity))（统一虚拟内存）**  
    允许CPU和GPU共享虚拟内存，自动迁移数据。

## 典型通信场景

-   **Intra-Node Communication（节点内通信）**  
    同一主机内多个GPU间的数据交换。
-   **Inter-Node Communication（节点间通信）**  
    跨主机（分布式系统）GPU间的数据交换，通常通过[InfiniBand](https://zhida.zhihu.com/search?content_id=259315297&content_type=Article&match_order=1&q=InfiniBand&zhida_source=entity)等高速网络完成。
-   **Zero-Copy Communication（零拷贝通信）**  
    设备间直接数据传输，无需中间缓冲区，降低延迟。
-   **Message Passing（消息传递）**  
    通过消息交换数据的机制，是GPU通信的基础（如MPI）。
-   **GPUNetIO**  
    由GPU直接进行网络通信的机制。

## 高级技术与术语

-   **[IBGDA](https://zhida.zhihu.com/search?content_id=259315297&content_type=Article&match_order=1&q=IBGDA&zhida_source=entity)（InfiniBand GPUDirect Async）**  
    允许GPU通过InfiniBand NIC直接与远程节点通信，将通信控制结构放入GPU内存，实现更低延迟和更高并发。
-   **[IBRC](https://zhida.zhihu.com/search?content_id=259315297&content_type=Article&match_order=1&q=IBRC&zhida_source=entity)（InfiniBand远程通信）**  
    指通过InfiniBand进行的远程通信机制，通常涉及RDMA和GPUDirect技术，实现高效数据交换。
-   **NVSHMEM**  
    NVIDIA基于OpenSHMEM的并行编程接口，为多GPU提供全局地址空间，实现高效、可扩展的数据通信。
-   **NVL72**  
    指NVIDIA DGX GB200 NVL72平台，通过NVLink Switch连接多达72块GPU，构建超高带宽、全互联GPU集群。
-   **MNNVL Domain**  
    不是标准术语，通常指多节点NVLink域，即通过NVLink互联的一组GPU，形成高带宽通信域。
-   **IB SHARP（InfiniBand可扩展分层聚合与归约协议）**  
    一种优化InfiniBand网络集体通信操作的协议，通过分层聚合和归约提升大规模GPU集群的通信效率。支持将集体通信卸载到网络硬件，降低CPU开销和延迟，常与GPUDirect RDMA等技术结合使用，加速多GPU和多节点通信。
-   **Multicast Communication（组播通信）**  
    指在GPU通信中，将一个源（如GPU或CPU）的数据同时高效地传输给多个目标设备（如多块GPU或节点）。组播由网络或互连（如InfiniBand/Ethernet）负责数据复制和分发，减少带宽占用和延迟，相比多次单播更高效。