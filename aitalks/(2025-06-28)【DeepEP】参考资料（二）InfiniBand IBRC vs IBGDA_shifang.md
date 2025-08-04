# 【DeepEP】参考资料（二）InfiniBand IBRC vs IBGDA

**Author:** shifang

**Date:** 2025-06-28

**Link:** https://zhuanlan.zhihu.com/p/1919711162783757414

### [InfiniBand](https://zhida.zhihu.com/search?content_id=259365690&content_type=Article&match_order=1&q=InfiniBand&zhida_source=entity) GPUDirect Async（IBGDA）in NVSHMEM

[Improving Network Performance of HPC Systems Using NVIDIA Magnum IO NVSHMEM and GPUDirect Async | NVIDIA Technical Blog](https://link.zhihu.com/?target=https%3A//developer.nvidia.com/blog/improving-network-performance-of-hpc-systems-using-nvidia-magnum-io-nvshmem-and-gpudirect-async/)

在NVIDIA系统中，节点内的GPU通过[NVLink](https://zhida.zhihu.com/search?content_id=259365690&content_type=Article&match_order=1&q=NVLink&zhida_source=entity)进行规模扩展互联，节点间则通过像InfiniBand这样的规模外网络连接。GPU用于通信、共享任务和高效并行操作的软件库统称为NVIDIA Magnum IO，这是一个面向并行、异步和智能数据中心输入输出的架构。NVIDIA Magnum IO 中的 NVSHMEM 是一个基于 [OpenSHMEM](https://zhida.zhihu.com/search?content_id=259365690&content_type=Article&match_order=1&q=OpenSHMEM&zhida_source=entity) 规范的通信库，提供了一个分区全局地址空间（PGAS）数据访问模型，用于访问高性能计算系统中所有 GPU 的内存。

由于与 GPU 架构的紧密集成，对于节点内的通讯，通过 NVLink 实现了对细粒度数据访问的高效支持。然而，对于节点间（internode）数据访问，由于需要主机 CPU 来管理通信操作，高效性仍然是一个挑战。

为了提高节点间的通讯效率，NVSHMEM 给出了一种新的通信方法——基于 GPUDirect Async 技术家族之上的 InfiniBand GPUDirect Async（IBGDA）。IBGDA 于 NVSHMEM 2.6.0 中首次引入，并在 2.7.0 和 2.8.0 版本中得到了显著改进。它使 GPU 在发起节点间 NVSHMEM 通信时能够绕过 CPU，无需对现有应用进行任何修改。正如我们展示的，这带来了使用 NVSHMEM 的应用在吞吐量和扩展性上的显著提升。

### IBRC 和 IBGDA 的区别

**IBRC（InfiniBand Reliable Connection）** 和 **IBGDA（InfiniBand GPUDirect Async）** 是两种基于 InfiniBand 的 GPU 通信方式，主要区别在于通信的发起者和路径：

| 特性 | IBRC | IBGDA |
| --- | --- | --- |
| 通信发起者 | CPU 代理（proxy）负责发起通信 | GPU SM（Streaming Multiprocessor）直接发起通信 |
| 控制路径 | GPU 通过 CPU 代理协调与 NIC 的通信 | GPU 与 NIC 直接交互，CPU 不参与通信控制路径 |
| 缓冲区位置 | 通信缓冲区通常在主机内存或由 CPU 管理 | 工作队列（WQ）和门铃（DBR）缓冲区放在 GPU 内存 |
| 延迟和性能 | 延迟较高，CPU 线程是瓶颈 | 延迟低，性能更好，尤其是稀疏通信场景 |
| 适用场景 | 传统的 GPU 远程通信 | 高性能、低延迟的 GPU 直接异步通信 |
| 典型性能表现 | 稠密通信时性能接近 IBGDA | 稀疏通信时延迟降低约 3.6 倍（如 Dispatch+Combine 测试） |

### 具体性能对比（参考[^1](https://link.zhihu.com/?target=https%3A//www.perplexity.ai/hub/blog/efficient-and-portable-mixture-of-experts-communication)）

-   在稠密 NVSHMEM all-to-all 操作中，IBRC 和 IBGDA 性能相近（6378 µs vs 6180 µs）。
-   在稀疏内核（Dispatch 和 Combine）中，IBGDA 显著更快（902 µs vs 3223 µs），延迟降低约 3.6 倍。
-   IBGDA 通过让 GPU 直接触发网络传输，消除了 CPU 代理，显著减少端到端延迟。

* * *

### 展示 IBRC 和 IBGDA 区别的测试用例

### 1\. NVSHMEM 性能测试套件中的 `shmem_put_bw`

-   `shmem_put_bw` 是 NVSHMEM 的带宽测试程序，支持测试 IBRC 和 IBGDA 两种传输模式。
-   通过设置环境变量控制启用 IBGDA：

```bash
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_IBGDA_NIC_HANDLER=gpu
```

-   关闭 IBGDA 则使用 IBRC。
-   运行示例：

```bash
mpirun -np 2 ./shmem_put_bw
```

-   该测试可直观比较两种传输方式的带宽和延迟差异。

### 2\. DeepEP 的 `test_internode.py`

-   DeepEP 框架的 `test_internode.py` 脚本可用于测试多节点 GPU 通信。
-   默认使用 IBRC，启用 IBGDA 需要额外配置。
-   通过运行该测试可以观察两种传输模式下的行为差异和性能表现（参考[^2](https://link.zhihu.com/?target=https%3A//github.com/deepseek-ai/DeepEP/issues/74)）。

* * *

### 参考总结

| 测试用例 | 说明 | 适用场景 |
| --- | --- | --- |
| shmem_put_bw | NVSHMEM 性能测试，支持 IBRC 和 IBGDA 比较 | 测试带宽和延迟，性能基准测试 |
| test_internode.py | DeepEP 多节点通信测试脚本，支持 IBRC 和 IBGDA | 验证多节点 GPU 通信功能和稳定性 |

* * *

### 结论

-   **IBRC** 是传统的 CPU 代理发起的 InfiniBand GPU 通信方式，适合一般场景。
-   **IBGDA** 是更先进的 GPU 直接异步发起通信，显著降低延迟，提升稀疏通信性能。
-   通过 NVSHMEM 的 `shmem_put_bw` 性能测试和 DeepEP 的 `test_internode.py` 脚本，可以直观地测试和对比这两种通信方式的差异和性能优势。
-   参考[^14](https://link.zhihu.com/?target=https%3A//github.com/Infrawaves/DeepEP_ibrc_dual-ports_multiQP) 对DeepEP的IBRC模式进行了优化。

* * *

以上内容基于最新的 DeepEP 和 NVSHMEM 相关资料整理，结合实际测试用例说明两者区别及验证方法。

* * *

### 参考资料

  
  
\[^1\]: [https://www.perplexity.ai/hub/blog/efficient-and-portable-mixture-of-experts-communication](https://link.zhihu.com/?target=https%3A//www.perplexity.ai/hub/blog/efficient-and-portable-mixture-of-experts-communication)  
  
\[^2\]: [https://github.com/deepseek-ai/DeepEP/issues/36](https://link.zhihu.com/?target=https%3A//github.com/deepseek-ai/DeepEP/issues/36)  
  
\[^3\]: [https://github.com/deepseek-ai/DeepEP/issues/74](https://link.zhihu.com/?target=https%3A//github.com/deepseek-ai/DeepEP/issues/74)  
  
\[^4\]: [https://docs.redhat.com/en/documentation/red\_hat\_enterprise\_linux/7/html/networking\_guide/sec-testing\_early\_infiniband\_rdma\_operation](https://link.zhihu.com/?target=https%3A//docs.redhat.com/en/documentation/red_hat_enterprise_linux/7/html/networking_guide/sec-testing_early_infiniband_rdma_operation)  
  
\[^5\]: [https://developer.nvidia.com/blog/improving-network-performance-of-hpc-systems-using-nvidia-magnum-io-nvshmem-and-gpudirect-async/](https://link.zhihu.com/?target=https%3A//developer.nvidia.com/blog/improving-network-performance-of-hpc-systems-using-nvidia-magnum-io-nvshmem-and-gpudirect-async/)  
  
\[^6\]: [https://hps.vi4io.org/\_media/events/2023/iodc23-newburn\_gpu\_io.pdf](https://link.zhihu.com/?target=https%3A//hps.vi4io.org/_media/events/2023/iodc23-newburn_gpu_io.pdf)  
  
\[^7\]: [https://docs.nvidia.com/nvshmem/release-notes-install-guide/best-practice-guide/device-apis.html](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/nvshmem/release-notes-install-guide/best-practice-guide/device-apis.html)  
  
\[^8\]: [https://www.fs.com/blog/exploring-infiniband-network-hdr-and-significance-of-ib-applications-in-supercomputing-8728.html](https://link.zhihu.com/?target=https%3A//www.fs.com/blog/exploring-infiniband-network-hdr-and-significance-of-ib-applications-in-supercomputing-8728.html)  
  
\[^9\]: [https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/abstract.html](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/abstract.html)  
  
\[^10\]: [https://github.com/deepseek-ai/DeepEP/issues/76](https://link.zhihu.com/?target=https%3A//github.com/deepseek-ai/DeepEP/issues/76)  
  
\[^11\]: [https://docs.nvidia.com/nvshmem/release-notes-install-guide/prior-releases/release-280.html](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/nvshmem/release-notes-install-guide/prior-releases/release-280.html)  
  
\[^12\]: [https://static.rainfocus.com/nvidia/gtcs24/sess/1693876934119001DqDe/FinalPresPDF/S61368\_1710778532525001A5Z4.pdf](https://link.zhihu.com/?target=https%3A//static.rainfocus.com/nvidia/gtcs24/sess/1693876934119001DqDe/FinalPresPDF/S61368_1710778532525001A5Z4.pdf)  
  
\[^13\]: [https://www.mecs-press.org/ijcnis/ijcnis-v8-n10/IJCNIS-V8-N10-2.pdf](https://link.zhihu.com/?target=https%3A//www.mecs-press.org/ijcnis/ijcnis-v8-n10/IJCNIS-V8-N10-2.pdf)  
  
\[^14\]: [https://github.com/Infrawaves/DeepEP\_ibrc\_dual-ports\_multiQP](https://link.zhihu.com/?target=https%3A//github.com/Infrawaves/DeepEP_ibrc_dual-ports_multiQP)