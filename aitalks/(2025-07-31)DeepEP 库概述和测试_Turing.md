# DeepEP 库概述和测试

**Author:** Turing

**Date:** 2025-07-31

**Link:** https://zhuanlan.zhihu.com/p/1933904714875008315

* * *

cssclasses: - wide-page

* * *

[DeepEP](https://zhida.zhihu.com/search?content_id=261016277&content_type=Article&match_order=1&q=DeepEP&zhida_source=entity) 是一个专为混合专家（Mixture-of-Experts, MoE）模型和专家并行（[Expert Parallelism](https://zhida.zhihu.com/search?content_id=261016277&content_type=Article&match_order=1&q=Expert+Parallelism&zhida_source=entity), EP）设计的通信库。它提供了高吞吐量、低延迟的全对全（all-to-all）GPU 内核，也就是 MoE 中的分发（dispatch）和合并（combine）操作，同时支持低精度运算，如 FP8。

## 相关前置知识

首先要理解 [MoE 模型](https://zhida.zhihu.com/search?content_id=261016277&content_type=Article&match_order=1&q=MoE+%E6%A8%A1%E5%9E%8B&zhida_source=entity)理论\[\[DeepSeek MoE\]\]，配合 DeepEP 通信的机制\[\[DeepEP 单节点通信\]\]理解代码，以及需要一些通用知识。

1.  分布式计算  
    
2.  **分布式训练基础**：理解数据并行、模型并行和专家并行的概念，以及它们在深度学习训练中的应用。DeepEP 主要聚焦于专家并行，需明白如何在多个设备或节点间分配和同步专家相关的计算与通信任务。  
    
3.  **MPI（Message Passing Interface）**：这是一种常用的分布式通信标准，DeepEP 的部分测试用例使用 MPI 进行多进程通信，因此需要了解 MPI 的基本概念，如进程管理、消息传递（发送、接收）和集体通信操作（广播、归约等）。  
    
4.  GPU 编程  
    
5.  **CUDA 编程**：DeepEP 大量使用 CUDA 进行 GPU 加速计算，需掌握 CUDA 的基本概念，如线程块、网格、共享内存、核函数等，以及如何在 Python 中通过 [PyTorch](https://zhida.zhihu.com/search?content_id=261016277&content_type=Article&match_order=1&q=PyTorch&zhida_source=entity) 调用 CUDA 相关功能。  
    
6.  **GPU 体系结构**：了解 GPU 的硬件架构，如流多处理器（SM）、内存层次结构（全局内存、共享内存、寄存器等），有助于理解 DeepEP 中对 GPU 资源的管理和优化策略。
7.  **NVSHMEM**：DeepEP 依赖修改后的 NVSHMEM，需要了解 NVSHMEM 的基本概念和使用方法。  
    
8.  网络通信  
    
9.  **[InfiniBand](https://zhida.zhihu.com/search?content_id=261016277&content_type=Article&match_order=1&q=InfiniBand&zhida_source=entity) 和 [RoCE](https://zhida.zhihu.com/search?content_id=261016277&content_type=Article&match_order=1&q=RoCE&zhida_source=entity)**：DeepEP 主要在 InfiniBand 网络上进行测试，理论上也支持 [RDMA](https://zhida.zhihu.com/search?content_id=261016277&content_type=Article&match_order=1&q=RDMA&zhida_source=entity) over Converged Ethernet (RoCE)。需要了解这两种网络技术的基本原理、特点以及如何在分布式系统中配置和使用它们。  
    
10.  **RDMA（Remote Direct Memory Access）**：DeepEP 的低延迟内核依赖 RDMA 技术，需掌握 RDMA 的基本概念和工作原理，以及它如何实现高效的远程内存访问。  
    
11.  深度学习框架  
    
12.  **PyTorch**：DeepEP 提供 Python 接口，基于 PyTorch 实现。需要熟悉 PyTorch 的张量操作、自动求导、分布式训练 API 等，以及如何在 PyTorch 项目中集成和使用 DeepEP。  
    

* * *

## File Hierarchy

```text
DeepEP/
├─csrc/                        # 包含 C++ 和 CUDA 源代码，用于实现核心功能
│  │  CMakeLists.txt           # CMake 配置文件，用于调试项目，正式设置使用 Torch 扩展
│  │  config.hpp               # 定义配置相关的结构体和类，如 LowLatencyLayout 结构体，用于计算低延迟布局所需的内存大小
│  │  deep_ep.cpp              # 实现 Buffer 类，负责初始化通信缓冲区，包括内存分配、IPC 句柄设置、计数器初始化等操作，提供缓冲区同步、销毁等功能
│  │  deep_ep.hpp              # 声明 Buffer 类及其成员函数，定义缓冲区的各种属性和操作接口，是 C++ 代码的头文件
│  │  event.hpp                # 可能包含事件处理相关的类和函数，用于管理 CUDA 事件，实现通信和计算的重叠
│  └─ kernels/                 # 包含 CUDA 内核代码
# 配置、API、异常处理
│          api.cuh             # 定义 CUDA 内核的 API 接口，供其他部分调用
│          buffer.cuh          # 与缓冲区操作相关的 CUDA 内核代码，可能包括缓冲区的分配、释放、读写等操作
│          CMakeLists.txt      # 用于构建 CUDA 内核库的 CMake 配置文件
│          configs.cuh         # 包含 CUDA 内核的配置信息，如线程块大小、网格大小等，计算 NVLink 和 RDMA 缓冲区的大小，`LowLatencyLayout` 结构体用于计算低延迟布局所需的内存大小。
│          exception.cuh       # 定义异常处理相关的类和函数，用于处理 CUDA 内核中的异常情况
# 通信
│          ibgda_device.cuh    # 与 InfiniBand 设备相关的 CUDA 内核代码，可能用于实现 RDMA 通信
│          internode.cu        # 实现节点间通信的 CUDA 内核代码，用于在不同节点之间传输数据
│          internode_ll.cu     # 实现低延迟节点间通信的 CUDA 内核代码，采用纯 RDMA 技术，减少延迟
│          intranode.cu        # 实现节点内通信的 CUDA 内核代码，用于在同一节点内的不同设备之间传输数据
# 启动和相关计算
│          launch.cuh          # 包含 CUDA 内核的启动函数，负责启动和管理 CUDA 内核的执行
│          layout.cu           # 实现布局计算的 CUDA 内核代码，用于计算数据在缓冲区中的布局
│          runtime.cu          # 包含运行时相关的 CUDA 内核代码，如内存管理、事件处理等
│          utils.cuh           # PTX 汇编实现的一些实用的 CUDA 内核函数，如数据转换、内存填充等
# Python 接口
├─deep_ep/                     # Python 代码的主要目录，提供 Python 接口
│      buffer.py               # 实现 dispatch 函数，用于将令牌分发到不同 rank，支持节点内和节点间设置
│      utils.py                # 定义 EventOverlap 类，用于管理 CUDA 事件，方便实现通信和计算的重叠；还提供了检查 NVLink 连接的函数
│      __init__.py             # Python 包的初始化文件，导入必要的模块和类
├─tests/                       # 包含测试相关的代码
│      test_internode.py       # 测试节点间通信功能的脚本，验证节点间数据传输的正确性
│      test_intranode.py       # 测试节点内通信功能的脚本，验证节点内数据传输的正确性
│      test_low_latency.py     # 测试低延迟通信功能的脚本，验证低延迟内核的性能和正确性
│      utils.py                # 定义 bench 函数，用于对函数进行基准测试，包括预热、测试和计算平均、最小、最大时间等操作
```

项目主要分为三个部分：Python 接口、C++ 封装层和 CUDA 内核层。 - Python 接口提供给用户使用，通过 Python 调用可用于构建专家并行（EP）通信缓冲区`Buffer`，以支持混合专家（MoE）模型的通信操作； - C++ 封装层`deep_ep.cpp`通过`buffer`负责处理 Python 和 CUDA 之间的交互，`deep_ep.hpp`提供了接口定义，； - CUDA 内核层实现具体的计算逻辑，承担起管理通信缓冲区、执行内核操作以及同步不同进程间信息的任务。

`csrc/kernels` 目录下包含了各种 CUDA 内核代码，如 `internode.cu`、`intranode.cu` 等，这些内核代码实现了具体的计算逻辑，如节点间和节点内的数据传输、布局计算等。

* * *

## DeepEP 通信组件（软硬件协同）

### **1\. [CUDA IPC](https://zhida.zhihu.com/search?content_id=261016277&content_type=Article&match_order=1&q=CUDA+IPC&zhida_source=entity) 在 DeepEP 中的作用**

### **技术背景**

-   **CUDA IPC（Inter-Process Communication）** 是 NVIDIA 提供的进程间通信机制，允许不同进程共享 GPU 显存。通过共享显存句柄（handle），进程可以直接访问远程 GPU 的内存，无需通过 CPU 中转。
-   **应用场景**：节点内多 GPU 之间的数据共享（如 NVLink 连接的 GPU）。

### **在 DeepEP 中的作用**

1.  **节点内 GPU 通信优化**：
2.  在 MoE 模型的专家并行（EP）中，不同 GPU 上的专家需要频繁交换数据（如分发和合并操作）。CUDA IPC 可直接共享显存，避免显存拷贝到 CPU 或跨节点传输的开销。
3.  **示例**：在 DeepEP 的 `all-to-all` 通信中，专家间的数据分发（dispatch）和合并（combine）可利用 CUDA IPC 实现节点内 GPU 的高速通信。  
    
4.  **低延迟分发与合并**：  
    
5.  DeepEP 的 `FP8 分发` 和 `BF16 合并` 操作需要低延迟的显存访问。CUDA IPC 通过共享句柄直接传递数据，减少 CPU 干预，降低通信延迟。  
    
6.  **资源利用率提升**：  
    
7.  通过 CUDA IPC 避免 CPU 中转，GPU 的计算资源（SM）可以专注于模型计算，而非等待数据传输完成。

### **2\. nvshmem 在 DeepEP 中的作用**

### **技术背景**

-   **nvshmem** 是 NVIDIA 基于 RDMA 技术开发的共享内存通信库，支持跨节点的高效数据传输。其核心特性包括：
-   **RDMA over InfiniBand**：绕过操作系统内核，直接访问远程节点内存，实现低延迟、高带宽通信。
-   **All-to-All 通信**：支持高效的点对点（P2P）和集合通信（如 AllReduce、AlltoAll）。
-   **异步操作**：允许计算与通信重叠，提升整体效率。

### **在 DeepEP 中的作用**

1.  **跨节点通信优化**：
2.  在 MoE 模型的分布式训练中，专家可能分布在多个节点上。nvshmem 通过 RDMA 技术实现跨节点的高速数据传输，解决节点间通信瓶颈。
3.  **示例**：DeepEP 的 `非对称域带宽转发`（如从 NVLink 域到 RDMA 域）依赖 nvshmem 的异构通信能力。  
    
4.  **低延迟推理解码**：  
    
5.  DeepEP 提供 **纯 RDMA 的低延迟内核**，适用于推理解码阶段。通过 nvshmem 的异步通信，解码任务的延迟可降低至 **163 微秒**（根据知识库数据）。  
    
6.  **动态资源控制**：  
    
7.  nvshmem 支持灵活的资源管理（如 SM 数量控制），DeepEP 利用这一特性优化 GPU 资源分配，适配不同规模的训练任务。  
    
8.  **与 FP8 压缩结合**：  
    
9.  DeepEP 的 FP8 分发操作通过 nvshmem 的 RDMA 传输，进一步压缩数据体积，减少带宽占用（节省 50% 带宽资源）。

### **3\. CUDA IPC 与 nvshmem 的协同作用**

在 DeepEP 中，CUDA IPC 和 nvshmem 通常 **分工协作**，覆盖节点内和节点间的通信需求：

| 场景 | 技术 | 作用 |
| --- | --- | --- |
| 节点内 GPU 通信 | CUDA IPC | 通过共享显存句柄，实现 NVLink 连接 GPU 的高速通信（单机带宽可达 158 GB/s）。 |
| 跨节点通信 | nvshmem | 利用 RDMA 技术，实现跨节点的低延迟、高带宽传输（单网卡带宽 43-47 GB/s）。 |
| 通信-计算重叠 | nvshmem | 异步通信与计算并行执行，减少 GPU 空闲时间（提升资源利用率）。 |
| 低延迟推理解码 | nvshmem | 通过纯 RDMA 内核，最小化延迟（<163 微秒），适合实时推理场景。 |

### **4\. 深度技术解析：DeepEP 的通信优化**

### **关键技术点**

1.  **All-to-All 通信内核**：
2.  DeepEP 的 `dispatch` 和 `combine` 操作基于 CUDA IPC 和 nvshmem 实现全对全通信，适配 MoE 的非对称数据分布。
3.  **示例代码片段**： `python from deep_ep import Buffer, EventOverlap buffer = Buffer(...) # 初始化通信缓冲区 buffer.dispatch(...) # 分发操作（利用 CUDA IPC 或 nvshmem） buffer.combine(...) # 合并操作（同上）`  
    
4.  **计算-通信重叠**：  
    
5.  DeepEP 通过 **hook-based 机制** 实现通信与计算的完全分离，无需占用 SM 资源。例如：  
    

-   在等待 RDMA 数据传输时，GPU 可继续执行其他计算任务。

1.  **异构网络优化**：  
    
2.  DeepEP 支持 **NVLink + RDMA** 的混合网络架构，通过 nvshmem 的异构通信接口，灵活适配不同硬件配置。

### **总结**

| 技术 | 核心作用 |
| --- | --- |
| CUDA IPC | 优化节点内 GPU 通信，实现 NVLink 高速数据共享，减少 CPU 中转。 |
| nvshmem | 优化跨节点通信，通过 RDMA 技术实现低延迟、高带宽传输，支持异步计算-通信重叠。 |
| DeepEP 协同 | CUDA IPC 与 nvshmem 结合，覆盖节点内和节点间通信，提升 MoE 模型的训练和推理效率。 |

* * *

## DeepEP 用于 MoE 的流程概述

### 1\. 硬件架构假设

假设系统由多台服务器组成，每台服务器包含：

-   8 个 GPU（通过 NVLink 互连）
-   每个 GPU 拥有 HBM 显存（高带宽内存）
-   服务器间通过 InfiniBand 网络连接
-   CPU 内存（用于数据暂存和控制流）

### 2\. 核心流程与软硬件分工

1.  初始化通信缓冲区：在开始训练或推理之前，需要初始化通信缓冲区。这涉及到设置缓冲区的大小和其他参数，以确保能够有效地处理数据的分发和组合。
2.  计算分发布局：在进行实际的分发操作之前，需要计算数据的分发布局。这包括确定每个 rank 接收的令牌数量、每个专家接收的令牌数量等信息。
3.  执行分发操作：根据计算得到的分发布局，执行实际的分发操作。这涉及到将数据从一个 rank 发送到其他 rank，并接收来自其他 rank 的数据。
4.  执行组合操作：在数据分发完成后，需要执行组合操作，将从其他 rank 接收到的数据组合成一个完整的张量。
5.  反向传播：在反向传播阶段，分发和组合操作的过程与正向传播相反。分发操作的反向过程实际上是组合操作，而组合操作的反向过程实际上是分发操作。

```python
# Python接口层 (CPU执行)
def moe_forward(input_tokens, router_logits):
    # 1. 路由器计算 (CPU/GPU)
    topk_experts, topk_weights = router(input_tokens, router_logits)

    # 2. 创建通信Buffer (CPU初始化)
    buffer = Buffer(group_size=8)  # 对应8个GPU

    # 3. 计算分发布局 (CPU/GPU)
    layout = buffer.get_dispatch_layout(topk_experts)

    # 4. 数据分发 (GPU执行，CPU同步)
    # 调用C++封装层
    dispatched_tokens, expert_indices = buffer.dispatch(
        input_tokens, topk_experts, layout
    )

    # 5. 专家计算 (GPU执行)
    expert_outputs = []
    for i in range(num_gpus):
        # 每个GPU执行本地专家计算
        expert_outputs.append(expert_forward(
            dispatched_tokens[i], 
            expert_weights[i]  # 本地专家参数
        ))

    # 6. 结果合并 (GPU执行，CPU同步)
    combined_output = buffer.combine(expert_outputs, expert_indices)

    return combined_output
```

### 3\. 数据流向与存储位置

| 组件 | 存储位置 | 说明 |
| --- | --- | --- |
| 原始 Tokens | GPU HBM | 输入序列，初始存储在发起计算的 GPU 显存中 |
| 分发布局 | GPU HBM + CPU 内存 | 布局元数据在 CPU 计算，最终传输到 GPU 用于指导数据分发 |
| 通信 Buffer | GPU HBM | 由Buffer类管理，用于暂存待发送 / 已接收的 Tokens，完全在 GPU 显存中 |
| 专家参数 | 各 GPU HBM | 每个 GPU 存储部分专家参数，例如 GPU0 存储专家 0-31，GPU1 存储专家 32-63 |
| 中间结果 | GPU HBM | 专家计算的输出结果，存储在执行计算的 GPU 中 |

### 4\. 专家分布策略

DeepEP 支持两种主要的专家分布模式：

### 模式 1：节点内专家并行

-   **场景**：单机多 GPU 系统（如 8 卡服务器）
-   **分布方式**：

-   每个 GPU 存储一部分专家（例如每个 GPU 存储总专家数的 1/8）
-   专家 ID 与 GPU 的映射关系为：`gpu_id = expert_id % num_gpus`

  
-   **数据流向**：

-   Tokens 根据路由结果被分发到对应的 GPU
-   例如，Token 被路由到专家 17，则该 Token 会被发送到 GPU1（17 % 8 = 1）

### 模式 2：节点间专家并行

-   **场景**：多机系统（例如 4 台服务器，每台 8 卡）
-   **分布方式**：

-   专家按层次分布：

-   第一层：跨服务器划分（例如服务器 0 存储专家 0-255，服务器 1 存储 256-511）
-   第二层：服务器内 GPU 划分（例如服务器 0 的 GPU0 存储专家 0-31）

  

-   映射关系：`server_id = expert_id // (experts_per_server)`

-   **数据流向**：

-   Tokens 先通过 InfiniBand 发送到目标服务器
-   再通过 NVLink 发送到服务器内的目标 GPU

### 5\. 通信流程详解

这一节可以在理解整体流程后再来理解通信加速设计。

假设一个 Token 被门控网络分配给 3 个专家：其中 2 个在 Machine 2（同机器），1 个在 Machine 5（跨机器）。DeepEP 的传输逻辑通过 “**跨节点合并传输 + 节点内高效分发**” 降低带宽压力：

### 1\. 跨节点传输（Infiniband，目标是减少数据量）

-   **合并同机器 Token**：发送到 Machine 2 的 Token 数据只传 1 份（而非 2 份），发送到 Machine 5 的 Token 传 1 份。
-   **通信载体**：通过 NVSHMEM 机制（节点间单向通信）传输，跳过传统 Send/Receive 的双向握手，减少延迟。

```text
Token传输第一步（跨节点）：
源机器（如Machine 0）
├─ 打包发往Machine 2的Token → 1份数据（供Machine 2内2个专家共享）
└─ 打包发往Machine 5的Token → 1份数据（供Machine 5内1个专家使用）
```

### 2\. 节点内分发（NVLink，利用高带宽）

-   **同机器内共享数据**：Machine 2 收到 Token 后，通过 NVLink（160GB/s 带宽，远高于 Infiniband 的 50GB/s）将 1 份 Token 数据分发到 2 个专家所在的 GPU。
-   **通信载体**：使用 IPC 机制（节点内单向通信），结合显存一致性，无需 CPU 参与，直接在 GPU 间传输。

```text
Token传输第二步（节点内）：
Machine 2接收端
├─ 收到1份Token数据（存在本地显存）
├─ 通过NVLink+IPC → 发送到Expert A所在GPU
└─ 通过NVLink+IPC → 发送到Expert B所在GPU（同一份数据复用，无额外传输）
```

### 3\. DeepEP 的通信核心机制为什么高效？

### 1\. 节点间：NVSHMEM（替代传统 NCCL/Send/Receive）

-   **单向操作**：无需等待接收方确认（传统 NCCL 是对称通信，需双向同步），发送方直接写入接收方的显存地址（基于显存一致性）。
-   **无锁同步**：通过显存一致性机制（如 GPU 显存的原子操作）实现同步，避免锁竞争，适合高并发场景。

### 2\. 节点内：IPC（替代传统 PCIe 通信）

-   **本地直连**：同一机器内 GPU 通过 NVLink 物理直连，IPC 机制直接映射显存地址，数据无需拷贝，直接访问。
-   **与 NVLink 协同**：收到 RDMA 数据后，立即通过 NVLink 分发（如技术文档提到 “RDMA 收到后马上用 NVLink 发送到专家”），减少中间缓存延迟。

### 3\. 对比传统通信（优势在哪里？）

| 通信场景 | 传统方式（NCCL/Send） | DeepEP（NVSHMEM+IPC） | 优势总结 |
| --- | --- | --- | --- |
| 节点间通信 | 双向握手，数据需多次拷贝 | 单向写入，直接显存访问 | 延迟降低 30%+，带宽利用率提升 |
| 节点内多专家 | 每个专家单独传输，重复数据 | 1 份数据复用，NVLink 分发 | 数据量减少（n→1），节省带宽 |

### 6\. 关键优化点

DeepEP 优化技术以表格形式：

| 优化维度 | 核心技术 | 具体实现 | 效果 / 优势 | 关联硬件 / 场景 |
| --- | --- | --- | --- | --- |
| 通信与计算重叠 | CUDA 流与事件管理 | 在等待 Tokens 接收时执行本地计算 | 隐藏通信延迟，提升 GPU 利用率 | 所有计算密集型场景 |
| 低延迟设计 | 纯 RDMA 模式 | 避免 CPU 参与，GPU 直接访问远程内存（internode_ll.cu） | 减少数据拷贝路径，延迟降低 50%+ | 跨节点通信（Infiniband） |
| 内存优化 | GPU HBM 直接使用 + 动态 Buffer 调整 | Buffer 常驻 GPU 显存，根据 Token 分布动态分配大小 | 减少 CPU-GPU 数据传输，内存利用率提升 20%+ | 显存受限场景（如大模型推理） |
| 数据传输优化 | 同节点专家数据共享 | 跨节点传输量从 “专家数” 降至 “机器数” | 减少跨节点通信量，降低 Infiniband 带宽压力 | 多专家并行模式 |
| 通信机制优化 | 单向通信（NVSHMEM/IPC） + 无锁同步 | 替代传统 Send/Receive，使用环形缓冲区和原子计数器 | 减少同步开销，提升并发能力 | 高并发通信场景 |
| 硬件特性利用 | NVLink 高带宽 + SM 专用分配 + PTX 指令优化 | 预留 20 个 SM 处理通信，使用 PTX 穿透 L2 缓存 | 充分发挥硬件性能，通信延迟降低 30%+ | NVIDIA GPU 集群 |
| 流水线处理 | RDMA 与 NVLink 衔接 | RDMA 接收未完成时即开始 NVLink 分发 | 隐藏传输延迟，提升整体吞吐量 | 多 GPU 节点间数据流动 |

### **1\. 硬件适配层**

-   **SM 专用分配**：将 GPU 的 Streaming Multiprocessors 分为计算核心和通信核心，避免资源竞争。
-   **PTX 指令优化**：直接操作 GPU 底层硬件，绕过 L2 缓存，减少数据访问延迟。

### **2\. 通信协议层**

-   **纯 RDMA 模式**：基于 InfiniBand 的远程直接内存访问，实现 GPU 间的高效数据传输。
-   **NVSHMEM/IPC 单向通信**：无需双向握手，降低通信协议开销。

### **3\. 内存管理层**

-   **HBM 直接使用**：将关键数据结构驻留在 GPU 高带宽内存中，避免 CPU 参与。
-   **动态 Buffer 调整**：根据实际 Token 分布，弹性分配内存资源。

### **4\. 算法优化层**

-   **同节点数据共享**：基于 MOE 模型的专家分配策略，减少冗余数据传输。
-   **流水线处理**：利用数据传输的时间窗口，提前进行后续处理。

### 7\. 不同场景的通信模式

DeepEP 针对不同场景设计了灵活的通信模式（低延迟模式与标准模式），并通过多种优化机制实现高效数据处理。以下结合具体技术细节展开说明：

### 1\. **低延迟模式（专为推理设计）**

-   **核心目标**：最小化单次请求的响应时间（Latency），适合实时交互场景（如聊天机器人、实时翻译）。
-   **技术特点**：

-   **纯 RDMA 通信**：跳过 CPU 和内核，直接通过网络适配器（如 InfiniBand）在 GPU 间传输数据，延迟可低至 163 微秒（处理 8 个专家时）。
-   **同步请求 - 响应**：采用类似 RPC 的模式，每个请求立即触发通信和计算，无需批量处理。
-   **预分配资源**：提前分配 GPU 内存和通信缓冲区，避免动态分配延迟。

  
-   **适用场景**：小批量请求（如单用户对话）、对延迟敏感的任务。

### 2\. **标准模式（适合训练与批处理）**

-   **核心目标**：最大化吞吐量（Throughput），适合大规模数据并行处理（如模型训练、离线推理）。
-   **技术特点**：

-   **批量操作**：将多个 token 请求合并为一个通信批次，减少通信次数。
-   **计算与通信重叠**：利用 CUDA 流（CUDA Stream）和非阻塞 MPI/NCCL 操作，在 GPU 计算时同时进行数据传输。
-   **动态负载均衡**：根据专家负载动态调整 token 分配，避免热点问题。

  
-   **适用场景**：大批量请求、对吞吐量要求高的任务。

### 8\. 多机多卡分布概述

要理解 DeepEP 在 MOE 模型专家并行中的作用，核心逻辑是：**将 “专家（Expert）” 分布在多卡 / 多机器上，通过 DeepEP 高效传递 “token（待处理数据）” 并聚合结果**。先明确核心概念：

-   **MOE 模型**：类似 “多个专家处理不同任务，门控决定谁来处理”—— 输入 token 先经过门控网络，得到每个 token 对专家的 “偏好分数”，再分配给 Top-K 个专家处理，最后聚合结果。
-   **专家并行（Expert Parallelism）**：将大量专家（比如 128 个）分散到多卡 / 多机器上（而非单卡容纳所有专家），解决单卡内存不足问题。
-   **DeepEP**：专为专家并行设计的 “通信桥梁”—— 负责 token 从输入设备到专家所在设备的分发（dispatch），以及专家输出结果的聚合（combine），底层优化了通信效率。

下面讲解从 Expert 分配到 Token 传递的流程。

### **第一步：Expert 如何分配到多卡 / 多机器？（类似 “工人分到不同车间”）**

假设场景：2 台机器（Machine A、Machine B），每台机器有 2 张 GPU 卡（Card 0、Card 1），共 16 个专家（Expert 0~15）。

| 设备层级 | 分配逻辑（DeepEP 支持灵活划分） | 可视化类比 |
| --- | --- | --- |
| 机器（Machine） | 按 “就近通信” 原则划分专家组（比如 Machine A 负责前 8 个专家，Machine B 负责后 8 个） | 2 个工厂，各负责一半任务 |
| GPU 卡（Card） | 单台机器内的 GPU 平均分配本组专家（Machine A 的 Card 0 负责 Expert 0~3，Card 1 负责 Expert 4~7） | 每个工厂有 2 个车间，各分一半工人 |
| 专家（Expert） | 每个 GPU 卡上的专家固定绑定设备（如 Card 0 上的 Expert 0~3 独占该卡的计算资源） | 每个车间有 4 个工人（专家） |

**分配结果图示**：

```text
Machine A（机器A）
├─ Card 0（GPU卡0）：Expert 0、Expert 1、Expert 2、Expert 3
└─ Card 1（GPU卡1）：Expert 4、Expert 5、Expert 6、Expert 7

Machine B（机器B）
├─ Card 0（GPU卡0）：Expert 8、Expert 9、Expert 10、Expert 11
└─ Card 1（GPU卡1）：Expert 12、Expert 13、Expert 14、Expert 15
```

_注：DeepEP 支持 “组限制门控”（Group-restricted Gating），可将专家按机器 / 卡分组，让 token 优先分配到同组专家（减少跨机器通信），比如限制 token 只能选择 Machine A 内的专家，降低跨机延迟。_

### **第二步：Token 如何通过 DeepEP 传递并处理？（类似 “零件分配给对应工人加工”）**

假设输入 10 个 token，门控网络为每个 token 选择 Top-2 个专家（比如 Token 0 选择 Expert 2 和 Expert 9），整个流程分 3 步：

### **阶段 1：门控网络生成 “分配指令”（类似 “零件分配清单”）**

-   输入 token 先经过门控网络（Gating Network），计算每个 token 对所有专家的 “偏好分数”（如 Token 0 对 Expert 2 的分数 0.8，对 Expert 9 的分数 0.7）。
-   筛选 Top-K 个专家（比如 K=2），生成 “token - 专家” 分配表（谁该去哪个专家）。

```text
门控输出（分配表）：
Token 0 → Expert 2（Machine A, Card 0）、Expert 9（Machine B, Card 0）
Token 1 → Expert 3（Machine A, Card 0）、Expert 8（Machine B, Card 0）
...（共10个Token，每个对应2个专家）
```

### **阶段 2：DeepEP 的 Dispatch（分发 token 到专家所在设备）**

这是 DeepEP 的核心功能之一：**根据分配表，将 token 从 “输入设备” 发送到 “专家所在设备”**（类似 “按清单把零件送到对应车间”）。

-   **输入设备**：假设所有 token 初始在 “输入卡”（比如 Machine A, Card 0，也可分散在多卡）。
-   **DeepEP 优化点**：

-   用高效通信内核（如基于 NVLink/RDMA）直接传输，跳过 CPU 中转；
-   按 “设备分组” 批量发送（比如发往 Machine A 的 token 打包，发往 Machine B 的 token 打包），减少通信次数。

```text
Dispatch流程（数据流向）：
输入卡（Machine A, Card 0）
├─ 批量发送Token 0、Token 1中去Machine A, Card 0的部分 → Expert 2、Expert 3（本地卡内通信，最快）
└─ 通过RDMA发送Token 0、Token 1中去Machine B, Card 0的部分 → Expert 9、Expert 8（跨机器通信，DeepEP优化带宽）
```

### **阶段 3：专家计算 + DeepEP 的 Combine（聚合专家输出）**

-   **专家计算**：每个专家在本地卡上处理收到的 token（比如 Expert 2 处理 Token 0，Expert 9 也处理 Token 0），输出 “专家对 token 的处理结果”。
-   **DeepEP 的 Combine**：将同一个 token 在不同专家的输出 “拉回原设备并聚合”（比如 Token 0 的两个专家输出，需传回原输入卡并加权求和）。

```text
Combine流程（数据流向）：
Expert 2（Machine A, Card 0）→ 输出Token 0的结果 → 通过本地通信传回输入卡
Expert 9（Machine B, Card 0）→ 输出Token 0的结果 → 通过RDMA传回输入卡
DeepEP在输入卡聚合：Token 0最终结果 = （Expert 2输出 × 0.8） + （Expert 9输出 × 0.7）
```

### **总结：DeepEP 在通信中的作用**

| 步骤 | 传统通信问题 | DeepEP 的优化（可视化理解） |
| --- | --- | --- |
| Dispatch | 小批量传输多、跨机延迟高 | 像 “快递批量分拣”：按设备分组打包，用专用通道（RDMA）传输 |
| Combine | 聚合时数据乱序、带宽浪费 | 像 “包裹汇总”：按 token ID 整理结果，高效合并传输 |
| 整体 | 通信占比高，拖慢训练 / 推理 | 通信与计算重叠（专家计算时提前传输），隐藏延迟 |

### **核心可视化总结图**

```text
┌───────────────┐      门控网络       ┌─────────────────────────────────────┐
│  输入Token    ├────────────────────►│ 生成Token-Expert分配表（Top-K）      │
└───────┬───────┘                     └───────────────┬─────────────────────┘
        │                                             │
        ▼                                             ▼
┌───────┬───────┐                     ┌───────────────┬─────────────────────┐
│输入设备│(多卡) │                     │  DeepEP       │                     │
└───────┼───────┘                     │  Dispatch     │                     │
        │                             └───────┬───────┘                     │
        └─────────────────────────────────────┘                             │
                                              │                              │
        ┌─────────────────────────────────────┼──────────────────────────────┘
        │                                     │
        ▼                                     ▼
┌───────────────┐               ┌───────────────────────┐
│Machine A      │               │ Machine B            │
│  Card 0       │               │  Card 0              │
│  Expert 2,3   │◄──────────────┤  Expert 8,9          │
│  (处理Token)  │               │  (处理Token)         │
└───────┬───────┘               └───────┬───────────────┘
        │                               │
        └───────────────┬───────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│ DeepEP Combine：聚合专家输出 → 输出最终Token结果      │
└─────────────────────────────────────────────────────┘
```

通过这个流程，DeepEP 解决了 MOE 专家并行中最核心的 “token - 专家” 通信问题 —— 让分散在多卡 / 多机器的专家能高效接收数据、返回结果，最终支撑 MOE 模型在大规模硬件上的训练和推理。

* * *

## `intranode` 测试详解

下面基于`tests/test_intranode.py`讲解测试通信库的步骤，这个测试文件就调用了`Buffer`提供的相关接口进行对比和性能测试。关于`Buffer`的具体实现后续讲解。

### 一、实验设置

### 参数配置

```python
# tests\test_intranode.py
@@ -261, 10
```

这一段是测试开始前的一些参数。

1.  **`'--num-processes', type=int, default=8`**

-   要启动的进程数量。在分布式训练或测试中，每个进程通常运行在不同的 GPU 或计算资源上，这些进程会协同工作完成任务。

1.  **`'--num-tokens', type=int, default=4096`**

-   ：输入的令牌数量。在自然语言处理等任务里，文本会被拆分成多个令牌，这些令牌是模型处理的基本单位。

1.  **`'--hidden', type=int, default=7168`**

-   隐藏层的维度大小。在深度学习模型中，隐藏层是输入层和输出层之间的中间层，该参数决定了隐藏层中神经元的数量。

1.  **`'--num-topk', type=int, default=8`**

-   每个令牌对应的 Top-K 专家数量。在混合专家（Mixture-of-Experts, MoE）模型中，每个令牌会根据得分选择与之最相关的 K 个专家进行处理。

1.  **`'--num-experts', type=int, default=256`**

-   模型中专家的总数。每个专家是一个独立的子模型，不同的专家擅长处理不同类型的任务。

### 数据分配

根据上面的参数，可以得到我们需要处理的输入数据的格式，以及会生成一些用于 router 的信息的格式。

-   **输入张量 `x`**：形状为 `(num_tokens, hidden)`，即 `(4096, 7168)`。每个进程会生成一个值为当前进程排名的 `x` 张量，其数据类型为 `torch.bfloat16`。
-   **得分矩阵 `scores`**：形状为 `(num_tokens, num_experts)`，即 `(4096, 256)`。每个进程生成一个随机的得分矩阵，用于确定每个令牌对应的 Top-K 专家。
-   **Top-K 索引 `topk_idx`**：通过对 `scores` 矩阵进行 `torch.topk` 操作得到，形状为 `(num_tokens, num_topk)`，即 `(4096, 8)`。它表示每个令牌对应的 Top-K 专家的索引。

### 专家分配

### 创建多进程节点内专家并行 EP 测试

```python
# tests\test_intranode.py
@@ -275, 1
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
```

`torch.multiprocessing.spawn` 是 PyTorch 提供的一个实用函数，用于在多进程环境下启动多个子进程并执行指定的函数。它简化了多进程编程的复杂性，尤其适用于分布式训练或测试场景。以下是该函数的基本语法和参数解释：

```python
torch.multiprocessing.spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn')
```

-   **`fn`**：需要在每个子进程中执行的目标函数。该函数的第一个参数必须是子进程的全局排名（`rank`）。
-   **`args`**：传递给目标函数 `fn` 的额外参数，以元组形式提供。
-   **`nprocs`**：要启动的子进程数量。
-   **`join`**：布尔值，指示是否等待所有子进程执行完毕。默认为 `True`。
-   **`daemon`**：布尔值，指示是否将子进程设置为守护进程。默认为 `False`。
-   **`start_method`**：启动子进程的方法，可选值有 `'spawn'`、`'fork'` 和 `'forkserver'`，默认为 `'spawn'`。

当前代码中 `spawn` 函数调用后的执行情况：

1.  **主进程创建子进程**：主进程调用 `torch.multiprocessing.spawn` 函数后，会创建 `num_processes` 个子进程。每个子进程都会执行 `test_loop` 函数。
2.  **传递参数**：`test_loop` 函数的第一个参数会被自动设置为该子进程的全局排名（`rank`），取值范围是 `0` 到 `num_processes - 1`。之后，`test_loop` 函数会接收到 `args` 元组中的参数，即 `num_processes` 和 `args`。
3.  **子进程执行**：每个子进程独立执行 `test_loop` 函数，进行分布式环境初始化、创建 `deep_ep.Buffer` 实例、执行测试逻辑等操作。在 `test_loop` 函数中，会调用 `init_dist` 函数初始化分布式环境，确保各个子进程之间可以进行通信。
4.  **同步与结束**：如果 `join` 参数为 `True`（默认值），主进程会等待所有子进程执行完毕后才会继续执行后续代码。当所有子进程执行完 `test_loop` 函数，释放资源并退出后，主进程也会结束。

在实际执行时，`test_loop` 函数接收到的参数情况如下：

```python
# tests/test_intranode.py
@@ -233, 24
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
```

-   `local_rank`：由 `spawn` 函数自动传入的子进程全局排名。
-   `num_local_ranks`：从 `args` 元组中获取的 `num_processes`。
-   `args`：从 `args` 元组中获取的命令行参数解析结果。

`init_dist` 函数初始化分布式环境，获取当前进程的全局排名 `rank`、总进程数 `num_ranks` 以及通信组 `group`。

-   `test_ll_compatibility`：一个布尔变量，用于控制是否进行低延迟功能测试，当前设置为 `False`，即不进行测试。
-   `num_rdma_bytes`：RDMA 缓冲区所需的字节数，初始化为 0。若进行低延迟测试，调用 `deep_ep.Buffer.get_low_latency_rdma_size_hint` 函数计算所需的 RDMA 缓冲区大小。

> 关于初始化分布式环境，详细见后续章节`utils.init_dist`。

### 创建 `Buffer` 实例

在这个例子中，由于设置了不进行低延迟相关测试，则剩余部分只包括：创建 `Buffer` 实例、设置随机种子、执行主测试函数、释放资源。

```python
buffer = deep_ep.Buffer(group, int(2e9), num_rdma_bytes, low_latency_mode=test_ll_compatibility,
                            num_qps_per_rank=(ll_num_experts // num_ranks if test_ll_compatibility else 1), explicitly_destroy=True)
```

创建 `deep_ep.Buffer` 实例，参数含义如下：

-   `group`：通信组。
-   `int(2e9)`：NVLink 缓冲区的字节数。
-   `num_rdma_bytes`：RDMA 缓冲区的字节数。
-   `low_latency_mode`：是否启用低延迟模式。
-   `num_qps_per_rank`：每个进程的队列对数量，低延迟模式下根据专家数量计算，否则为 1。
-   `explicitly_destroy`：是否需要显式调用 `destroy` 方法释放资源，设置为 `True`。

`Buffer` 的具体组成具体见后文，`intranode`测试只有`num_nvl_bytes = int(2e9)`会对后面的测试起作用，RDMA相关的参数可以不用考虑。

### 执行主测试函数

```python
for i in (24, ):
        test_main(args, i, local_rank, num_ranks, rank, buffer, group)
        if local_rank == 0:
            print('', flush=True)
```

-   遍历 `(24, )` 这个元组，将 `i` 作为多处理器数量传入 `test_main` 函数进行测试。
-   若当前进程是本地排名为 0 的进程，打印一个空行。

### 二、测试配置与数据生成

上面做好了实验基本的参数配置和分布式进程初始化，在`test_main`函数中就是DeepEP 主要测试功能函数的实现。这一部分定义了 `test_main` 函数，该函数的主要功能是对 `deep_ep.Buffer` 的分发（dispatch）和合并（combine）操作进行全面测试，同时对这些操作的性能进行调优。

下面解读 dispatch layout 计算，并和`buffer`实现对比，测试性能。

```python
# tests/test_intranode.py
@@ -15, 55
```

### 函数定义与初始化配置

```python
def test_main(args: argparse.Namespace, num_sms: int, local_rank: int, num_ranks: int, rank: int,
              buffer: deep_ep.Buffer, group: dist.ProcessGroup):
```

-   `args`：命令行参数解析后的命名空间对象。
-   `num_sms`：流多处理器（SM）的数量，这个测试中在调用时指定了`num_sums=24`。
-   `local_rank`：当前进程在本地节点的排名。
-   `num_ranks`：总进程数。
-   `rank`：当前进程的全局排名。
-   `buffer`：`deep_ep.Buffer` 实例，用于通信操作。
-   `group`：分布式进程组。

```python
assert num_experts % num_ranks == 0
    if local_rank == 0:
        print(f'[config] num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}', flush=True)
```

从命令行参数中获取令牌数量、隐藏维度大小、Top-K 专家数量和专家总数。确保专家总数能被进程数整除，并打印配置信息。

-   要求 `num_experts` 能被 `num_processes` 整除，在默认情况下 `256 % 8 == 0` 满足条件。每个进程负责一部分专家，专家数量为 `num_experts // num_processes = 256 // 8 = 32` 个。

### 数据生成

```python
# tests/test_intranode.py
@@ -25, 11
    # Random data
```

这一部分，每一个 rank 都会各自生成各种输入数据，包括全 1 张量、随机张量、FP8 格式的张量、得分矩阵、Top-K 索引和权重等，并计算每个令牌对应的进程排名。

### 1\. 原始数据

1\. `x`

```python
x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * rank
```

-   **数据说明**：生成一个形状为 `(num_tokens, hidden)` 的张量，数据类型为 `torch.bfloat16`，放置在 CUDA 设备上。张量的所有元素初始值为 1，再乘以当前进程的全局排名 `rank`。
-   **作用**：作为输入数据，用于测试分发和合并操作，其元素值与进程排名相关，方便后续验证数据分发和合并的正确性。

2\. `x_pure_rand`

```python
x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
```

-   **数据说明**：生成一个形状为 `(num_tokens, hidden)` 的张量，数据类型为 `torch.bfloat16`，放置在 CUDA 设备上。张量元素服从标准正态分布。
-   **作用**：作为纯随机输入数据，用于测试分发和合并操作在随机数据下的表现。

3\. `x_e4m3`

```python
x_e4m3 = per_token_cast_to_fp8(x) if deep_ep.Buffer.is_sm90_compiled() else None
x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T) if x_e4m3 is not None else None
```

-   **数据说明**：首先检查 `deep_ep.Buffer.is_sm90_compiled()` 是否为 `True`，若为 `True`，调用 DeepEP 库定义的 `per_token_cast_to_fp8` 函数将 `x` 转换为 FP8（E4M3 格式）；否则，`x_e4m3` 为 `None`。若 `x_e4m3` 不为 `None`，对其第二个元素进行转置再转置操作，确保内存连续。
-   **作用**：作为 FP8 格式的输入数据，用于测试在低精度计算下分发和合并操作的正确性。确保仅在兼容的设备上启用 FP8，避免不支持硬件的错误。
-   **硬件支持**：SM90（如 H100 GPU）引入了专门的 FP8 Tensor Core，对 E4M3 格式提供原生支持，计算速度比 BF16 快得多。
-   **性能对比**：在 H100 上，FP8 的矩阵乘法吞吐量是 BF16 的 2 倍，能效比更高。

> 关于 e4m3 格式转换，详细见后续章节`utils.per_token_cast_to_fp8`。

### 2\. 计分统计

```python
scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
```

-   **数据说明**：生成一个形状为 `(num_tokens, num_experts)` 的张量，数据类型为 `torch.float32`，放置在 CUDA 设备上。张量元素服从标准正态分布，取绝对值后加 1，避免负值影响后续排序。
-   **作用**：作为每个令牌对应每个专家的得分，用于确定每个令牌对应的 Top-K 专家。

```python
topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
```

-   **数据说明**：对 `scores` 张量在最后一个维度（即每一行）上取前 `num_topk` 个最大值，返回这些值的索引。形状为 `(num_tokens, num_topk)`。
-   **作用**：表示每个令牌对应的 Top-K 专家的索引。

```python
topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='cuda') * rank
```

-   **数据说明**：生成一个形状为 `(num_tokens, num_topk)` 的张量，数据类型为 `torch.float32`，放置在 CUDA 设备上。张量元素初始值为 1，再乘以当前进程的全局排名 `rank`。
-   **作用**：作为每个令牌对应的 Top-K 专家的权重，用于测试分发和合并操作中权重的处理。

```python
topk_weights_pure_rand = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')
```

-   **数据说明**：生成一个形状为 `(num_tokens, num_topk)` 的张量，数据类型为 `torch.float32`，放置在 CUDA 设备上。张量元素服从标准正态分布。
-   **作用**：作为纯随机的 Top-K 专家权重，用于测试在随机权重下分发和合并操作的表现。

### 3\. 专家索引到计算设备

这一步计算就把 token 到 expert 的选择关系`topk_idx`转化为了 **token 到 rank/device 的选择关系**。

```python
rank_idx = topk_idx // (num_experts // num_ranks)
rank_idx.masked_fill_(topk_idx == -1, -1)
inplace_unique(rank_idx, num_ranks)
```

-   **数据说明**：

-   第一行：计算每个 Top-K 专家所在的进程排名。
-   第二行：将 `topk_idx` 中值为 -1 的位置对应的 `rank_idx` 元素也设为 -1。
-   第三行：调用 DeepEP 库定义的 `inplace_unique` 函数对 `rank_idx` 进行原地去重操作，确保每个令牌对应的进程排名唯一。

  
-   **作用**：表示每个令牌对应的 Top-K 专家所在的进程排名，用于后续计算每个进程的令牌数量和布局信息。  
    
-   `rank_idx` 表示每个 Top-K 专家所在的进程排名，通过 `topk_idx // (num_experts // num_processes)` 计算得到。后续会根据 `rank_idx` 进行数据的分发和合并操作，确保每个令牌能被正确分配到对应的进程和专家进行处理。  
    
-   这里的专家映射情况就是直接的线性映射，如：

-   8 个专家（`num_experts = 8`），4 个计算设备（`num_ranks = 4`），因此每个 rank 负责 2 个专家（`8 // 4 = 2`） `专家索引 | 对应rank 0 1 | 0 2 3 | 1 4 5 | 2 6 7 | 3`

### 全局元数据计算 layout

这一部分主要是调用了`deep_ep.cpp`接口的`buffer.get_dispatch_layout`。

```python
# tests/test_intranode.py
@@ -38, 20
    # Expert meta
    # Rank layout meta
```

### 令牌计算

每个 rank 都各自计算每个专家和每个进程的令牌数量，并通过 `dist.all_reduce` 进行全局同步。

1.  **计算每个专家的令牌数量**

````python
num_tokens_per_expert = torch.zeros((num_experts, ), dtype=torch.int, device='cuda')
for i in range(num_experts):
    num_tokens_per_expert[i] = (topk_idx == i).sum()
    ```

创建一个形状为 `(num_experts,)` 的全零张量，数据类型为 `torch.int`，放置在 CUDA 设备上，用于存储每个专家对应的令牌数量。
遍历每个专家，对于每个专家 `i`，统计 `topk_idx` 中值等于 `i` 的元素数量，即该专家对应的令牌数量。`topk_idx` 是一个形状为 `(num_tokens, num_topk)` 的张量，表示每个令牌对应的 Top-K 专家的索引。

3. **全局同步**

```python
gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
dist.all_reduce(gbl_num_tokens_per_expert, group=group)
````

先克隆 `num_tokens_per_expert` 张量得到 `gbl_num_tokens_per_expert`，然后使用 `dist.all_reduce` 函数对 `gbl_num_tokens_per_expert` 进行全局归约操作，将所有进程中每个专家的令牌数量累加起来，得到全局每个专家的令牌数量。`group` 是进程组，确保归约操作在指定的进程组内进行。

### layout 计算

随后计算每个进程的令牌数量、令牌在进程中的索引，以及判断每个令牌是否在某个进程中，并进行全局同步。

1.  **初始化张量**

```python
num_tokens_per_rank = torch.empty((num_ranks, ), dtype=torch.int, device='cuda')
token_idx_in_rank = torch.full((num_ranks, num_tokens), -1, dtype=torch.long, device='cuda')
```

-   `num_tokens_per_rank`：创建一个形状为 `(num_ranks,)` 的空张量，数据类型为 `torch.int`，放置在 CUDA 设备上，用于存储每个进程的令牌数量。
-   `token_idx_in_rank`：创建一个形状为 `(num_ranks, num_tokens)` 的张量，数据类型为 `torch.long`，放置在 CUDA 设备上，初始值全为 -1，用于存储每个令牌在进程中的索引。  
    
-   **计算每个进程的令牌数量和令牌索引**  
    `python for i in range(num_ranks): num_tokens_per_rank[i] = (rank_idx == i).sum() token_sel = (rank_idx == i).max(dim=-1)[0] count = token_sel.sum().item() tokens = torch.sort(token_sel.to(torch.int), descending=True)[1] tokens[:count] = torch.sort(tokens[:count])[0] token_idx_in_rank[i][tokens[:count]] = torch.arange(count, dtype=torch.long, device='cuda')`  
    

遍历每个进程，对于每个进程 `i`：

-   `num_tokens_per_rank[i] = (rank_idx == i).sum()`：统计 `rank_idx` 中值等于 `i` 的元素数量，即该进程对应的令牌数量。`rank_idx` 是一个形状为 `(num_tokens, num_topk)` 的张量，表示每个令牌对应的 Top-K 专家所在的进程排名。
-   `token_sel = (rank_idx == i).max(dim=-1)[0]`：对于每个令牌，判断其是否有对应的 Top-K 专家在进程 `i` 中，得到一个形状为 `(num_tokens,)` 的布尔张量。

-   `rank_idx`维度`(#tokens, #topk)`表示每个tokens选择的专家索引。
-   第一步比较生成 `bool` 类型，`True`表示 token 选择了`rank[i]`，布尔类型在`max`计算时会被转为0、1，所以等价于在`num_topk`维度上取`or`操作。

-   结果是一个元组 `(values, indices)`，其中 `values` 是每个 token 在 `num_topk` 维度上的最大值（`1` 或 `0`），形状为 `(num_tokens,)`。
-   **`[0]`**：取元组的第一个元素（即 `values`），得到形状为 `(num_tokens,)` 的布尔张量（`1` 对应 `True`，`0` 对应 `False`）。

  

-   `token_sel`维度`(#tokens, )`，每一个元素表示，对`rand_idx`每一行/每一个`token`判断该行的`topk`中是否有`rank i`。

-   `count = token_sel.sum().item()`：统计 `token_sel` 中 `True` 的数量，即该进程对应的令牌数量。
-   `tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]`：将 `token_sel` 转换为整数类型，然后进行降序排序，得到排序后的索引。

-   排序后返回一个元组 `(sorted_values, sorted_indices)`：

-   `sorted_values`：排序后的值（例如 `[1,1,0,0]`）；
-   `sorted_indices`：原始张量中元素的索引（即 “哪些 token 是 1，哪些是 0”）。

  

-   **`[1]`**：取元组的第二个元素（即 `sorted_indices`），得到形状为 `(num_tokens,)` 的索引张量。

-   `tokens[:count] = torch.sort(tokens[:count])[0]`：对前 `count` 个索引进行升序排序。
-   `token_idx_in_rank[i][tokens[:count]] = torch.arange(count, dtype=torch.long, device='cuda')`：利用 python **索引**语法将排序后的前 `count` 个令牌的索引依次赋值给 `token_idx_in_rank` 中对应进程的位置。  
    

-   `token_idx_in_rank` 是形状为 `(num_ranks, num_tokens)` 的张量，这里给第 `i` 行（对应 rank `i`）中 “需要处理的 token” 的位置，赋值为连续的本地索引。

  
-   **调整 `token_idx_in_rank` 张量**  
    

```python
token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
```

对 `token_idx_in_rank` 进行转置操作，使其形状变为 `(num_tokens, num_ranks)`，然后调用 `contiguous` 方法确保张量在内存中是 “连续存储” 的，最后将数据类型转换为 `torch.int`。

1.  **判断每个令牌是否在某个进程中**

```python
is_token_in_rank = token_idx_in_rank >= 0
```

创建一个布尔张量 `is_token_in_rank`，判断 `token_idx_in_rank` 中每个元素是否大于等于 0，即判断每个令牌是否在某个进程中。

1.  **全局同步**

```python
gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
dist.all_reduce(gbl_num_tokens_per_rank, group=group)
```

先克隆 `num_tokens_per_rank` 张量得到 `gbl_num_tokens_per_rank`，然后使用 `dist.all_reduce` 函数对 `gbl_num_tokens_per_rank` 进行全局归约操作，将所有进程中每个进程的令牌数量累加起来，得到全局每个进程的令牌数量。

### 步骤可视化

下面以一个可视化例子解释这个循环的步骤。假设我们有：

-   4 个 token（`num_tokens = 4`）
-   3 个 rank（`num_ranks = 3`）
-   `rank_idx` 内容如下：

```python
rank_idx = torch.tensor([
    [0, 1],  # Token 0选择了rank 0和1上的专家
    [1, 2],  # Token 1选择了rank 1和2上的专家
    [0, 2],  # Token 2选择了rank 0和2上的专家
    [1, 0]   # Token 3选择了rank 1和0上的专家
])
```

### 1\. 统计每个 rank 负责的 token 数量

```text
对于rank 0：token 0、2、3 → num_tokens_per_rank[0] = 3
对于rank 1：token 0、1、3 → num_tokens_per_rank[1] = 3
对于rank 2：token 1、2   → num_tokens_per_rank[2] = 2
```

### 2\. 构建 token 在 rank 内的索引映射

```python
# 初始token_idx_in_rank（全-1）
token_idx_in_rank = [
    [-1, -1, -1, -1],  # rank 0
    [-1, -1, -1, -1],  # rank 1
    [-1, -1, -1, -1]   # rank 2
]

# 处理rank 0
token_sel = [True, False, True, True]
tokens = [0, 2, 3, 1]  # 排序后的token索引
count = 3
token_idx_in_rank[0][[0, 2, 3]] = [0, 1, 2]  # 设置为0,1,2

# 处理rank 1
token_sel = [True, True, False, True]
tokens = [0, 1, 3, 2]
count = 3
token_idx_in_rank[1][[0, 1, 3]] = [0, 1, 2]

# 处理rank 2
token_sel = [False, True, True, False]
tokens = [1, 2, 0, 3]
count = 2
token_idx_in_rank[2][[1, 2]] = [0, 1]

# 转置后
token_idx_in_rank = [
    [0, -1, -1],  # token 0
    [-1, 0, 1],   # token 1
    [1, -1, 0],   # token 2
    [2, 1, -1]    # token 3
]

# 有效token掩码
is_token_in_rank = [
    [True, False, False],
    [False, True, True],
    [True, False, True],
    [True, True, False]
]
```

### layout 验证

验证 `buffer.get_dispatch_layout` 方法计算得到的布局信息是否与手动计算的布局信息一致，同时测量该方法的执行性能。

1.  调用 `get_dispatch_layout` 方法获取参考布局信息

```python
ref_num_tokens_per_rank, _, ref_num_tokens_per_expert, ref_is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)
```

输入参数：

-   `buffer.get_dispatch_layout(topk_idx, num_experts)`：调用 `deep_ep.Buffer` 实例 `buffer` 的 `get_dispatch_layout` 方法，传入 `topk_idx`（每个令牌对应的 Top-K 专家索引）和 `num_experts`（专家总数）作为参数，该方法会返回一系列布局相关的信息。

intranode 的输出只有三个有效：

-   `ref_num_tokens_per_rank`：参考的每个进程的令牌数量。
-   `ref_num_tokens_per_expert`：参考的每个专家的令牌数量。
-   `ref_is_token_in_rank`：参考的每个令牌是否在某个进程中的布尔张量。
-   `_`：表示忽略该位置返回的值。  
    
-   验证参考布局信息与手动计算的布局信息是否一致  
    

```python
assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
    assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
    assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)
```

-   `torch.allclose`：用于比较两个张量的所有元素是否接近。如果两个张量对应位置的元素差值在一定的容忍范围内，则认为它们接近。
-   这三个 `assert` 语句分别验证参考的每个进程的令牌数量、每个专家的令牌数量以及每个令牌是否在某个进程中的布尔张量是否与手动计算的结果一致。如果不一致，程序会抛出 `AssertionError` 异常，表明 `get_dispatch_layout` 方法的计算结果可能存在问题。  
    
-   测量 `get_dispatch_layout` 方法的执行性能并打印  
    

```python
t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
```

-   `bench`：这是一个 DeepEP 自定义的性能测试函数，用于测量传入的函数的执行时间。
-   `lambda: buffer.get_dispatch_layout(topk_idx, num_experts)`：定义了一个匿名函数，该函数调用 `buffer.get_dispatch_layout` 方法。
-   `t`：获取 `bench` 函数返回结果的第一个元素，即 `get_dispatch_layout` 方法的执行时间。

最后调用进程组 `group` 的 `barrier` 方法，该方法会阻塞当前进程，直到进程组内的所有进程都调用了该方法，确保所有进程在这一步完成同步，然后短暂等待。

### 配置对象设置

```python
# Config
    nvl_buffer_size = 256
    config = deep_ep.Config(num_sms, 8, nvl_buffer_size)
```

调用`csrc\config.hpp`初始化配置对象，设置 SM 数量、NVL 块大小和 NVL 缓冲区大小。

### 三、dispatch 测试

上面生成数据和配置后，下面就调用了`deep_ep.cpp`接口的`buffer.dispatch`来进行测试分发和合并。下面解读三个不同情形下的合并测试。

### 循环测试

```python
# tests/test_intranode.py
@@ -86, 85
    for previous_mode in (False, True):
        for async_mode in (False, True):
            for current_x in filter(lambda elem: elem is not None, (x_pure_rand, x, x_e4m3)):
                for with_topk in (False, True):
                    # ... 分发操作 ...
                    # ... 数据检查 ...
                    # ... 合并操作 ...
                    # ... 数据检查 ...
```

这一部分通过四重循环遍历不同的测试模式，包括_是否使用之前的事件、是否异步执行、不同的数据类型以及是否包含 Top-K 信息_，对 `buffer.dispatch` 和 `buffer.combine` 方法进行测试，并对结果进行检查。

这里使用了四重嵌套循环，每个循环代表一个测试维度，组合起来能覆盖多种测试场景：

-   `previous_mode`：布尔值，代表是否使用之前捕获的事件，用于测试 `dispatch` 方法在依赖先前事件时的表现。
-   `async_mode`：布尔值，代表是否使用异步模式调用 `dispatch` 方法，测试异步和同步模式下的功能。
-   `current_x`：输入数据，可能是纯随机张量 `x_pure_rand`、与进程排名相关的全 1 张量 `x` 或 FP8 格式的张量 `x_e4m3`（若支持），测试不同数据类型和特征下的 `dispatch` 功能。
-   `with_topk`：布尔值，代表是否在 `dispatch` 调用中包含 `topk_idx` 和 `topk_weights` 参数，测试有无 Top-K 信息时的功能。

循环开始先打印信息。

语法解释：

1.  `filter(lambda elem: elem is not None, (x_pure_rand, x, x_e4m3))`

这是一个**过滤迭代器**，用于筛选出非`None`的输入张量。

-   **`(x_pure_rand, x, x_e4m3)`**：一个元组，包含三个可能的输入张量：

-   `x_pure_rand`：纯随机初始化的张量（bfloat16 格式）；
-   `x`：固定值初始化的张量（bfloat16 格式）；
-   `x_e4m3`：FP8 格式的张量（可能为`None`，例如当 GPU 不支持 SM90 架构时）。

  
-   **`lambda elem: elem is not None`**：一个匿名函数（lambda 表达式），作为过滤条件：

-   输入`elem`为元组中的每个元素；
-   返回`True`如果`elem`不是`None`，否则返回`False`。

-   **`filter(...)`**：Python 内置函数，根据 lambda 的返回值过滤元组：

-   保留所有`elem is not None`的元素；
-   返回一个迭代器，遍历所有非`None`的张量。

**作用**：只处理有效的输入张量（跳过`x_e4m3`为`None`的情况），避免后续代码报错。

1.  `recv_x, ... = buffer.dispatch(** dispatch_args)`

这是**函数调用与参数解包**，调用`buffer.dispatch`方法并接收返回值。

-   `**dispatch_args`：字典解包语法，将`dispatch_args`中的键值对转换为函数的关键字参数，等价于：  
    `python buffer.dispatch(x=current_x, num_tokens_per_rank=num_tokens_per_rank, is_token_in_rank=is_token_in_rank)`  
    
-   `event.current_stream_wait() if async_mode else ()`  
    

这是一个**三元表达式**，等价于：

```python
if async_mode:
    event.current_stream_wait()
else:
    pass  # 空元组 () 表示不执行任何操作
```

1.  **`*recv_x`**：元组解包语法，将元组中的元素展开作为参数传递给函数。例如：

```python
# 若 recv_x = (data, scale)
per_token_cast_back(*recv_x)  # 等价于 per_token_cast_back(data, scale)
```

### 构建 `dispatch` 方法的参数

```python
dispatch_args = {'x': current_x, 'num_tokens_per_rank': num_tokens_per_rank,  'is_token_in_rank': is_token_in_rank,
                 'num_tokens_per_expert': num_tokens_per_expert, 'config': config, 'async_finish': async_mode}
if with_topk:
    dispatch_args.update({'topk_idx': topk_idx, 'topk_weights': topk_weights_pure_rand if current_x is x_pure_rand else topk_weights})
if previous_mode:
    dispatch_args.update({'previous_event': buffer.capture()})
```

-   构建 `dispatch` 方法的基本参数，包含输入数据、每个进程的令牌数量、令牌是否在进程中的信息、每个专家的令牌数量、配置对象以及是否异步执行。
-   若 `with_topk` 为 `True`，添加 `topk_idx` 和 `topk_weights` 参数。
-   若 `previous_mode` 为 `True`，添加之前捕获的事件参数。

### 调用 `dispatch` 方法

```python
recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, event = buffer.dispatch(**dispatch_args)
event.current_stream_wait() if async_mode else ()
recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x
```

-   调用 `buffer.dispatch` 方法进行数据分发操作，获取返回的接收数据、接收的 Top-K 索引、接收的 Top-K 权重、每个专家接收的令牌数量列表、句柄和事件。
-   若 `async_mode` 为 `True`，等待事件完成。
-   若 `recv_x` 是元组，调用 `per_token_cast_back` 函数将其转换回原始数据类型。

### 接收检查

这段代码的主要功能是对 `buffer.dispatch` 方法返回的接收数据、Top-K 索引和 Top-K 权重进行了全面的检查，确保分发操作的正确性和数据的一致性。

1.  提取前缀矩阵

```python
rank_prefix_matrix = handle[0]
```

从 `dispatch` 方法返回的 `handle` 元组中提取第一个元素作为 `rank_prefix_matrix`，这个矩阵后续会用于验证数据的一致性。

1.  验证接收数据的数量

```python
assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(0), f'{gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}'
```

-   `gbl_num_tokens_per_rank[rank].item()`：获取当前进程的全局令牌数量。
-   `recv_x.size(0)`：获取 `dispatch` 方法返回的接收数据 `recv_x` 的第一维大小，即接收的令牌数量。
-   `assert` 语句检查这两个值是否相等，如果不相等，会抛出 `AssertionError` 异常，并打印具体的错误信息。  
    
-   验证每个专家接收的令牌数量  
    

```python
assert gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist() == recv_num_tokens_per_expert_list
```

-   `gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()`：将全局每个专家的令牌数量张量 `gbl_num_tokens_per_expert` 重塑为 `(num_ranks, -1)` 的形状，然后提取当前进程对应的部分并转换为列表。
-   `recv_num_tokens_per_expert_list`：`dispatch` 方法返回的当前进程中每个专家接收的令牌数量列表。
-   `assert` 语句检查这两个列表是否相等，确保每个专家接收的令牌数量正确。  
    
-   检查接收数据的一致性  
    

```python
if current_x is not x_pure_rand:
                        check_data(recv_x, rank_prefix_matrix)
```

-   如果当前输入数据 `current_x` 不是纯随机数据 `x_pure_rand`，则调用 `check_data` 函数对接收数据 `recv_x` 进行检查。
-   `check_data` 函数用于验证 `recv_x` 的一致性，确保每个进程的数据符合预期。  
    
-   初始化克隆的 Top-K 权重  
    

```python
recv_topk_weights_clone = None
```

初始化 `recv_topk_weights_clone` 为 `None`，后续在需要时会对 `recv_topk_weights` 进行克隆。

1.  检查 Top-K 索引

```python
if with_topk:
                        # Check `topk_idx`
                        assert (recv_topk_idx.eq(-1) | ((recv_topk_idx >= 0) & (recv_topk_idx < (num_experts // num_ranks)))).sum().item() == recv_topk_idx.numel()
                        for i, count in enumerate(recv_num_tokens_per_expert_list):
                            assert recv_topk_idx.eq(i).sum().item() == count
```

-   **第一个 `assert` 语句**：

-   `recv_topk_idx.eq(-1)`：检查 `recv_topk_idx` 中值为 -1 的元素。
-   `(recv_topk_idx >= 0) & (recv_topk_idx < (num_experts // num_ranks))`：检查 `recv_topk_idx` 中值在有效范围内（大于等于 0 且小于每个进程的专家数量）的元素。
-   `(recv_topk_idx.eq(-1) | ((recv_topk_idx >= 0) & (recv_topk_idx < (num_experts // num_ranks))))`：对上述两个条件进行逻辑或运算，得到所有有效元素。
-   `(recv_topk_idx.eq(-1) | ((recv_topk_idx >= 0) & (recv_topk_idx < (num_experts // num_ranks)))).sum().item()`：计算有效元素的数量。
-   `recv_topk_idx.numel()`：计算 `recv_topk_idx` 中元素的总数。
-   该 `assert` 语句确保 `recv_topk_idx` 中的所有元素都是有效元素。

  
-   **第二个 `assert` 语句**：  
    

-   遍历 `recv_num_tokens_per_expert_list`，对于每个专家 `i`，检查 `recv_topk_idx` 中值等于 `i` 的元素数量是否等于该专家接收的令牌数量 `count`。

-   检查 Top-K 权重  
    

```python
# Check `topk_weights`
                        recv_topk_weights_clone = recv_topk_weights.clone()
                        if current_x is not x_pure_rand:
                            recv_topk_weights[recv_topk_idx.eq(-1)] = recv_topk_weights.amax(dim=1, keepdim=True).expand_as(recv_topk_weights)[recv_topk_idx.eq(-1)]
                            check_data(recv_topk_weights, rank_prefix_matrix)
```

-   **克隆 Top-K 权重**：对 `recv_topk_weights` 进行克隆，保存原始数据。
-   **处理无效索引对应的权重**：

-   如果当前输入数据 `current_x` 不是纯随机数据 `x_pure_rand`，则将 `recv_topk_idx` 中值为 -1 的元素对应的 `recv_topk_weights` 替换为该行的最大值。
-   `recv_topk_weights.amax(dim=1, keepdim=True)`：计算 `recv_topk_weights` 每行的最大值。
-   `expand_as(recv_topk_weights)`：将最大值扩展为与 `recv_topk_weights` 相同的形状。
-   `recv_topk_weights[recv_topk_idx.eq(-1)] = ...`：将无效索引对应的权重替换为最大值。

  
-   **检查处理后的 Top-K 权重**：调用 `check_data` 函数对处理后的 `recv_topk_weights` 进行检查，确保其一致性。

### dispatch 不同情景检查

这两段代码分别对不同场景下 `buffer.dispatch` 方法的功能进行测试。

-   `# Test num_worst_tokens != 0` 部分在包含 Top-K 信息的场景下，测试 `dispatch` 方法处理 `num_worst_tokens` 参数的功能。
-   `# Test cached dispatch (must without top-k staffs)` 部分在不包含 Top-K 信息的场景下，测试 `dispatch` 方法的缓存分发功能。

### `# Test num_worst_tokens != 0` 部分

```python
# tests/test_intranode.py
@@ -121, 14
    # Test `num_worst_tokens != 0
```

此部分代码在 `with_topk` 为 `True` 的情况下，测试 `dispatch` 方法在传入 `num_worst_tokens` 参数时的行为，以此验证 `dispatch` 方法处理最差令牌数量的能力。

1.  **设置 `num_worst_tokens` 参数**：  
    `python num_worst_tokens = num_tokens * num_ranks dispatch_args.update({'num_worst_tokens': num_worst_tokens})`  
    计算 `num_worst_tokens` 的值并将其添加到 `dispatch_args` 字典中，后续会将其作为参数传递给 `dispatch` 方法。  
    
2.  **调用 `dispatch` 方法**：  
    `python recv_worst_x, recv_worst_topk_idx, recv_worst_topk_weights, empty_list, _, event = buffer.dispatch(**dispatch_args)`  
    调用 `buffer.dispatch` 方法，传入更新后的参数，获取接收数据、Top-K 索引、Top-K 权重等返回值。  
    
3.  **处理异步操作**：  
    `python event.current_stream_wait() if async_mode else ()`  
    若处于异步模式，等待事件完成，确保数据处理完成。  
    
4.  **数据类型转换**：  
    `python recv_worst_x = per_token_cast_back(*recv_worst_x) if isinstance(recv_worst_x, tuple) else recv_worst_x`  
    若 `recv_worst_x` 是元组，调用 `per_token_cast_back` 函数将其转换回原始数据类型。  
    
5.  **结果验证**：  
    `python assert len(empty_list) == 0 assert num_worst_tokens == recv_worst_x.size(0) assert num_worst_tokens == recv_worst_topk_idx.size(0) assert num_worst_tokens == recv_worst_topk_weights.size(0) assert torch.equal(recv_x, recv_worst_x[:recv_x.size(0)]) assert torch.equal(recv_topk_idx, recv_worst_topk_idx[:recv_x.size(0)]) assert torch.equal(recv_topk_weights_clone, recv_worst_topk_weights[:recv_x.size(0)]) assert torch.all(recv_worst_topk_idx[recv_x.size(0):] == -1).item()`  
    通过一系列 `assert` 语句验证返回结果的正确性，包括 `empty_list` 是否为空、接收数据的大小是否符合预期，以及前 `recv_x.size(0)` 个元素是否与之前的结果一致等。  
    

### `# Test cached dispatch (must without top-k staffs)` 部分

```python
# tests/test_intranode.py
@@ -137, 9
    # Test cached dispatch (must without top-k staffs)
```

这部分代码在 `with_topk` 为 `False` 的情况下，测试 `dispatch` 方法的缓存分发功能，即不使用 Top-K 相关参数时的行为。

1.  **构建参数**：  
    `python dispatch_args = {'x': current_x, 'handle': handle, 'config': config, 'async_finish': async_mode} if previous_mode: dispatch_args.update({'previous_event': buffer.capture()})`  
    构建调用 `dispatch` 方法所需的参数，若 `previous_mode` 为 `True`，添加之前捕获的事件参数。  
    
2.  **调用 `dispatch` 方法**：  
    `python recv_x, _, _, _, _, event = buffer.dispatch(**dispatch_args)`  
    调用 `buffer.dispatch` 方法，传入构建好的参数，获取接收数据和事件。  
    
3.  **处理异步操作**：  
    `python event.current_stream_wait() if async_mode else ()`  
    若处于异步模式，等待事件完成，确保数据处理完成。  
    

### 四、combine 测试

```python
# tests/test_intranode.py
@@ -148, 14
    # Test combine
```

在同样的循环里面，这段代码通过构建不同参数组合调用 `combine` 方法，该方法用于将之前 `dispatch` 分发出去的数据合并回来，对合并后的数据和 Top-K 权重进行处理，并与参考数据进行比较，验证 `combine` 方法的正确性。通过设置不同的测试条件（如是否包含 Top-K 信息、是否启用 `previous_mode` 等），确保合并操作的正确性。

### 1\. 构建 `combine` 方法的参数

```python
combine_args = {'x': recv_x, 'handle': handle, 'config': config, 'async_finish': async_mode}
if with_topk:
    combine_args.update({'topk_weights': recv_topk_weights})
if previous_mode:
    combine_args.update({'previous_event': buffer.capture()})
```

-   `combine_args`：构建一个字典，包含 `combine` 方法的基本参数。

-   `x`：需要合并的数据，即之前 `dispatch` 方法返回的 `recv_x`。
-   `handle`：`dispatch` 方法返回的句柄，包含分发操作的布局信息。
-   `config`：配置对象，用于指定合并操作的配置参数。
-   `async_finish`：布尔值，指定是否以异步模式执行合并操作。

  
-   `if with_topk`：如果在测试中包含 Top-K 信息，将 `recv_topk_weights`（`dispatch` 方法返回的 Top-K 权重）添加到参数中。
-   `if previous_mode`：如果启用了 `previous_mode`，调用 `buffer.capture()` 捕获当前事件，并将其添加到参数中，用于事件同步。

### 2\. 调用 `combine` 方法

```python
combined_x, combined_topk_weights, event = buffer.combine(**combine_args)
                    event.current_stream_wait() if async_mode else ()
```

-   `buffer.combine(**combine_args)`：调用 `combine` 方法进行数据合并操作，返回合并后的数据 `combined_x`、合并后的 Top-K 权重 `combined_topk_weights` 以及一个事件对象 `event`。
-   `event.current_stream_wait() if async_mode else ()`：如果以异步模式执行合并操作，等待事件完成，确保数据合并操作已经结束。

### 3\. 验证合并后的数据

```python
check_x = combined_x.float() / is_token_in_rank.sum(dim=1).unsqueeze(1)
                    ref_x = x_pure_rand if current_x is x_pure_rand else x
                    assert calc_diff(check_x, ref_x) < 5e-6
```

-   `check_x`：对合并后的数据 `combined_x` 进行处理，将其转换为 `float` 类型，并除以每个令牌所属进程的数量（`is_token_in_rank.sum(dim=1).unsqueeze(1)`），得到用于验证的数据。
-   `ref_x`：根据当前输入数据 `current_x` 的类型，选择参考数据。如果 `current_x` 是纯随机数据 `x_pure_rand`，则参考数据为 `x_pure_rand`；否则，参考数据为 `x`。
-   `assert calc_diff(check_x, ref_x) < 5e-6`：调用 `calc_diff` 函数计算 `check_x` 和 `ref_x` 之间的差异，并使用 `assert` 语句确保差异小于 `5e-6`。如果差异超过该阈值，测试将失败。

### 4\. 验证合并后的 Top-K 权重（如果包含 Top-K 信息）

```python
if with_topk:
                        check_topk_weights = combined_topk_weights if (current_x is x_pure_rand) else (combined_topk_weights / is_token_in_rank.sum(dim=1).unsqueeze(1))
                        ref_topk_weights = topk_weights_pure_rand if current_x is x_pure_rand else topk_weights
                        assert calc_diff(check_topk_weights, ref_topk_weights) < 1e-9
```

-   `check_topk_weights`：根据当前输入数据 `current_x` 的类型，对合并后的 Top-K 权重 `combined_topk_weights` 进行处理。如果 `current_x` 是纯随机数据 `x_pure_rand`，则直接使用 `combined_topk_weights`；否则，将其除以每个令牌所属进程的数量。
-   `ref_topk_weights`：根据当前输入数据 `current_x` 的类型，选择参考的 Top-K 权重。如果 `current_x` 是纯随机数据 `x_pure_rand`，则参考权重为 `topk_weights_pure_rand`；否则，参考权重为 `topk_weights`。
-   `assert calc_diff(check_topk_weights, ref_topk_weights) < 1e-9`：调用 `calc_diff` 函数计算 `check_topk_weights` 和 `ref_topk_weights` 之间的差异，并使用 `assert` 语句确保差异小于 `1e-9`。如果差异超过该阈值，测试将失败。

### 五、Tune 处理

```python
# For later tuning
dispatch_bf16_nvl_recv_bytes = recv_x.numel() * 2
combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes
```

在分布式训练里，了解数据传输量对于性能调优至关重要。此代码计算分发和合并操作期间通过 NVLink（NVIDIA 高速互联技术）传输的数据字节数，为后续性能优化提供依据。

-   `recv_x.numel()`：该方法返回 `recv_x` 张量中的元素总数。`recv_x` 是分发操作后接收到的数据。
-   `* 2`：由于 `recv_x` 的数据类型是 `torch.bfloat16`，每个 `bfloat16` 数据元素在内存中占用 2 字节，将元素总数乘以 2 就得到了分发操作期间通过 NVLink 接收的总字节数。
-   `dispatch_bf16_nvl_recv_bytes`：这个变量存储计算得到的接收字节数。  
    
-   `combine_bf16_nvl_send_bytes`：该变量表示合并操作期间通过 NVLink 发送的字节数。  
    
-   在理想情况下，合并操作发送的数据量应与分发操作接收的数据量相等。所以，这行代码直接将 `dispatch_bf16_nvl_recv_bytes` 的值赋给 `combine_bf16_nvl_send_bytes`。

这两个变量会在后续的性能调优代码里用于计算分发和合并操作的数据传输速率（GB/s），帮助开发者找到最优的配置参数。例如：

```python
# ...
t = bench(lambda: buffer.dispatch(**tune_args))[0]
if local_rank == 0:
    print(f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size if nvl_chunk_size else "default"}: '
          f'{dispatch_bf16_nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL), avg_t: {t * 1e6:.2f} us', flush=True)
# ...
```

### 六、分发和合并性能调优

```python
# Tune dispatch performance
    best_dispatch_results = None
    fp8_factor = (1 + 4 / 128) / 2
    for current_x in filter(lambda elem: elem is not None, (x_e4m3, x)):
        best_time, best_results = 1e10, None
        for nvl_chunk_size in tuple(range(4, 33, 2)) + (0, ):
            # ... 测试不同配置下的分发性能 ...
        # ... 记录最佳配置 ...

    # Tune combine performance
    best_time, best_results = 1e10, None
    for nvl_chunk_size in tuple(range(1, 17, 1)) + (0, ):
        # ... 测试不同配置下的合并性能 ...
    # ... 记录最佳配置 ...
```

分别对分发和合并操作进行性能调优，尝试不同的 NVL 块大小，记录最佳性能配置并打印结果。

* * *

## utils TODO

### `init_dist`

### **1\. `init_dist(local_rank, num_local_ranks)` 函数作用**

`init_dist` 是 DeepEP 库自定义的一个**分布式环境初始化函数**，主要负责初始化多进程通信所需的基础组件，为后续分布式操作（如数据同步、集体通信）做准备。具体作用包括：

-   **初始化通信后端**：启动分布式通信所需的后端（如 NCCL 或 Gloo，通常用于 GPU 间通信）。
-   **设置进程标识**：确定当前进程的全局 `rank`（唯一标识）和总进程数 `num_ranks`（world size）。
-   **绑定设备**：根据 `local_rank` 将当前进程绑定到指定 GPU（避免多进程抢占同一 GPU）。
-   **创建通信组**：构建进程间的通信组 `group`，用于后续集体通信操作（如 `dist.all_reduce`、`dist.all_gather`）。

### **2\. 返回值 `group` 的作用**

`group` 是 `torch.distributed.ProcessGroup` 类型的对象，代表**一组参与通信的进程集合**，是分布式通信的核心协调者。其具体作用如下：

### **（1）限定集体通信的作用范围**

`group` 定义了哪些进程参与集体通信操作（如 `dist.all_reduce`、`dist.all_gather`）。例如：

```python
# 使用 group 限定 all_reduce 仅在该组内的进程间执行
gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
dist.all_reduce(gbl_num_tokens_per_expert, group=group)  # 仅 group 内的进程参与数据同步
```

若不指定 `group`，PyTorch 会默认使用全局通信组（包含所有进程），但 `init_dist` 返回的 `group` 可能是根据测试需求定制的子通信组（如仅包含当前节点内的进程，符合 `test_intranode.py` 的“节点内测试”场景）。

### **（2）作为 `deep_ep.Buffer` 的通信句柄**

在创建 `deep_ep.Buffer`（专家并行通信缓冲区）时，`group` 被作为参数传入：

```python
buffer = deep_ep.Buffer(group, int(2e9), num_rdma_bytes, ...)
```

`Buffer` 类需要通过 `group` 获取进程间的通信上下文（如通信后端、进程拓扑），以实现跨进程的数据分发（dispatch）和聚合（combine），这是专家并行（EP）中“跨进程路由 token 到对应专家”的核心依赖。

### **（3）确保分布式操作的一致性**

后续所有依赖分布式协调的操作（如性能调优时的配置同步、测试结果验证）均基于 `group` 进行。例如，在收集最佳配置时：

```python
# 基于 group 收集所有进程的最佳配置
dist.all_gather(all_best_fp8_results_list, best_dispatch_results, group=group)
```

`group` 确保了不同进程间的数据交换仅在预设的通信范围内生效，避免跨组干扰，同时保证通信效率。

### 3\. torch 进程组

`torch.distributed.ProcessGroup` 是 PyTorch 中实现分布式训练的核心抽象概念，它定义了一组参与通信的进程（processes）以及它们之间的通信方式。通过 `ProcessGroup`，PyTorch 能够在多机多卡环境中高效协调计算资源，实现数据并行、模型并行等多种分布式训练策略。一个分布式作业里可以有多个进程组，每个进程组包含不同的进程子集，各进程组相互独立。**核心作用**如下：

\##### **(1) 通信范围划分**

-   在分布式训练中，进程组定义了通信的范围。将多个进程划分为不同的组（例如，将 8 个进程分为 2 个组，每组 4 个进程），每个组形成一个独立的通信域。不同组的进程可以独立进行通信，互不干扰。
-   例如：

-   **默认进程组**：通过 `torch.distributed.init_process_group` 初始化的全局进程组，通常包含所有参与训练的进程。
-   **自定义进程组**：通过 `torch.distributed.new_group` 创建的子组，用于特定的通信需求（如混合并行（数据并行 + 模型并行）中，不同层的模型参数在独立的组内同步）。

### **(2) 支持通信操作**

-   封装底层通信实现（如 NCCL、GLOO、MPI），提供一致的 API（如 `all_reduce`、`broadcast`、`send/recv`），使代码不依赖于具体的通信后端。

-   **NCCL**：NVIDIA GPU 间的高性能通信，支持 GPUDirect RDMA。
-   **GLOO**：跨平台（CPU/GPU）通信，适合小规模集群和快速原型。
-   **MPI**：支持异构环境和复杂网络拓扑。

  
-   所有集合通信操作（如 `all_reduce`、`broadcast`、`all_gather` 等）都依赖于 `ProcessGroup` 来指定通信的进程范围。

-   例如，`dist.all_reduce(tensor, group=group)` 会将 `tensor` 在指定的 `group` 中进行归约操作。

-   点对点通信操作 **Send/Recv**：进程间直接发送和接收数据。  
    _应用_：流水线并行中不同阶段之间的数据传输。

### **(3) 灵活的分布式策略**

-   通过划分不同的进程组，可以实现更复杂的分布式策略，例如：

-   **数据并行**：所有进程属于同一组，同步梯度。
-   **模型并行**：不同组处理模型的不同部分（如不同层），独立通信。

### `per_token_cast_to_fp8`

### **1\. 输入输出概述**

-   **输入**：`x: torch.Tensor`（2D 张量，形状 `[num_tokens, hidden]`，`hidden % 128 == 0`，数据类型为 BF16/FP32）。
-   **输出**：元组 `(fp8_data_tensor, scales_tensor)`，其中：

-   `fp8_data_tensor`：量化后的 FP8 数据（E4M3 格式），形状与输入 `x` 一致 `[num_tokens, hidden]`。
-   `scales_tensor`：每个量化块的缩放因子，形状 `[num_tokens, num_groups]`（`num_groups = hidden / 128`）。

### **2\. 核心变量形状与变换逻辑**

（`m=num_tokens`, `n=hidden`, `g=num_groups=n/128`）

| 变量名 | 形状 | 作用与变换逻辑 |   |
| --- | --- | --- | --- |
| x | (m, n) | 输入张量，需满足 n % 128 == 0（按 128 元素分块量化的前提）。 |   |
| x_view | (m, g, 128) | 将 x 按隐藏维度分块：x.view(m, -1, 128)，其中 -1 自动计算为 g = n/128。例如 n=512 时 g=4，x_view 形状为 (m, 4, 128)。 |   |
| x_amax | (m, g) | 计算每个 128 元素块的绝对值最大值（amax）：① x_view.abs().float()：转 FP32 避免精度损失；② amax(dim=2)：沿第 2 维（128 元素块）取最大值，得到 (m, g)；③ clamp(1e-4)：限制最小值为 1e-4，避免后续除零错误。 |   |
| fp8_data_tensor | (m, n) | 量化后的数据：① x_view * (448.0 / x_amax.unsqueeze(2))：将每个块缩放到 E4M3 范围（[-448, 448]），x_amax.unsqueeze(2) 扩展为 (m, g, 1) 以广播到 x_view 的 (m, g, 128)；② .to(torch.float8_e4m3fn)：转换为 E4M3 FP8 格式；③ .view(m, n)：恢复原始形状。 |   |
| scales_tensor | (m, g) | 缩放因子（用于反量化）：(x_amax / 448.0).view(m, -1)：计算量化时的缩放系数倒数（1/scale），形状保持 (m, g)。 |   |

### **3\. 为何返回元组？**

FP8 量化是**有损压缩**，需同时存储：

-   **量化后的数据**（`fp8_data_tensor`）：用 1 字节/元素存储，相比 BF16（2 字节）节省 50% 内存。
-   **缩放因子**（`scales_tensor`）：记录每个 128 元素块的动态范围（`x_amax / 448.0`），用于反量化时恢复原始数据精度（通过 `per_token_cast_back` 函数）。

二者缺一不可，因此返回元组 `(fp8_data_tensor, scales_tensor)`。

### **4\. 关键设计细节**

-   **分块量化（128 元素/块）**：隐藏维度按 128 元素分块（`x_view`），平衡量化精度与计算效率（块太小则缩放因子存储开销大，块太大则精度损失严重）。
-   **E4M3 格式适配**：E4M3 FP8 的动态范围为 `[-448, 448]`，因此通过 `448.0 / x_amax` 将每个块的最大值归一化到 448，确保数据能被 FP8 精确表示。
-   **数值稳定性**：`clamp(1e-4)` 避免 `x_amax` 过小导致的除零错误，`float()` 转换确保 `amax` 计算精度。

### **总结**

该函数通过**分块量化**将高Precision张量（BF16/FP32）压缩为 FP8（E4M3）格式，同时记录缩放因子，实现内存高效存储与后续精确恢复。返回元组是为了同时保留量化数据和反量化所需的动态范围信息。

* * *

## 参考

1.  [https://zhuanlan.zhihu.com/p/1890067712996270654](https://zhuanlan.zhihu.com/p/1890067712996270654)