# DeepSeek MoE 个人解读

**Author:** Turing

**Date:** 2025-07-31

**Link:** https://zhuanlan.zhihu.com/p/1933622437737666472

* * *

cssclasses: - wide-page

* * *

参考了DeepSeek 源代码、文档，网上的许多文章、AI 工具整理了关于[DeepSeek MoE](https://zhida.zhihu.com/search?content_id=260986356&content_type=Article&match_order=1&q=DeepSeek+MoE&zhida_source=entity)的解读。

## MoE 原理概述

### 一、MoE 核心概念与架构本质

MoE（Mixture of Expert，混合专家模型）是一种为提升大模型性能与效率而设计的[稀疏激活](https://zhida.zhihu.com/search?content_id=260986356&content_type=Article&match_order=1&q=%E7%A8%80%E7%96%8F%E6%BF%80%E6%B4%BB&zhida_source=entity)架构，核心逻辑是 “按需调用计算资源”，而非传统模型的 “全量计算”。**“并非所有参数在每次推理时都需要被激活”**，即“稀疏激活”（Sparsity）。它最早在 1990 年代提出，近年来因在大模型（如 Google 的 **GLaM**、Meta 的 **Llama 3-MoE**、NVIDIA 的 **Mixtral**）中的成功应用而重新受到广泛关注。

MoE 通过 “稀疏激活 + 专家并行” 的核心逻辑，打破了传统稠密模型 “参数量与计算量绑定” 的限制，成为超大模型训练的 “效率神器”。而 [DeepSeek V3](https://zhida.zhihu.com/search?content_id=260986356&content_type=Article&match_order=1&q=DeepSeek+V3&zhida_source=entity) 等模型通过专家细分（Routed/[Shared Expert](https://zhida.zhihu.com/search?content_id=260986356&content_type=Article&match_order=1&q=Shared+Expert&zhida_source=entity)）进一步释放了 MoE 的潜力，结合 DeepEP 等专用通信库的支撑，最终实现了 “大参数、小计算、高性能” 的目标。随着模型参数量向万亿级突破，MoE 架构及其配套技术（通信优化、[路由算法](https://zhida.zhihu.com/search?content_id=260986356&content_type=Article&match_order=1&q=%E8%B7%AF%E7%94%B1%E7%AE%97%E6%B3%95&zhida_source=entity)）将成为大模型研发的核心竞争力。

-   **架构定位**：通常集成在 [Transformer Block](https://zhida.zhihu.com/search?content_id=260986356&content_type=Article&match_order=1&q=Transformer+Block&zhida_source=entity) 的输出阶段，作为 “最后一层特征提取器”。在 Transformer 的自注意力和前馈网络之后，MoE 通过门控逻辑对特征进行精细化加工，既保留大模型的表征能力，又避免全量参数计算的冗余。
-   **稀疏激活机制**：模型包含多个 “专家网络”（Expert，可理解为独立的 Transformer 子网络），以及一个 “[门控网络](https://zhida.zhihu.com/search?content_id=260986356&content_type=Article&match_order=1&q=%E9%97%A8%E6%8E%A7%E7%BD%91%E7%BB%9C&zhida_source=entity)”（Gating Network）。对于每个输入 Token，门控网络仅选择部分专家（如 Top-2、Top-4）进行激活，其他专家处于 “休眠” 状态。这种设计使模型在保持超大参数量（通过大量专家扩展）的同时，将实际计算量控制在较低水平（仅激活部分专家）。

> **“用专家并行 + 动态路由 + [All-to-All 通信](https://zhida.zhihu.com/search?content_id=260986356&content_type=Article&match_order=1&q=All-to-All+%E9%80%9A%E4%BF%A1&zhida_source=entity)” 实现“超大模型 + 稀疏激活 + 高效计算”的统一。**

* * *

### 二、MoE 的核心优势：效率与性能的平衡

MoE 之所以成为超大模型训练的核心技术，关键在于解决了 “参数量扩展” 与 “计算成本” 的矛盾：

-   **高参数量与低计算量的兼容**：  
    传统稠密模型的参数量与计算量呈正相关（参数量翻倍，计算量也翻倍），而 MoE 通过 “专家并行” 将参数分散到多个专家中，参数量可随专家数量线性扩展，但计算量仅取决于被激活的专家数量。例如 DeepSeek V3 的模型参数量达 671B，但通过 MoE 的稀疏激活，实际计算量被控制在 37B（接近传统 37B 稠密模型），实现了 “大参数性能” 与 “小计算成本” 的兼顾。
-   **可扩展性**：  
    当需要提升模型性能时，无需重构整个网络，只需增加专家数量（每个专家专注于特定知识域），门控网络会自动将 Token 分配给最匹配的专家。这种 “模块化扩展” 方式比传统稠密模型的 “整体扩容” 更灵活，也更易适配分布式训练。
-   **推理效率优化**：  
    推理阶段，MoE 仅需加载被激活专家的参数（或按需调度），减少内存占用；同时，稀疏计算可降低单 Token 处理的计算耗时，尤其适合长文本生成等场景。

### 三、MoE 分布式计算原理

MoE（[Mixture of Experts](https://zhida.zhihu.com/search?content_id=260986356&content_type=Article&match_order=1&q=Mixture+of+Experts&zhida_source=entity)）是一种高效的参数化模型架构，通过动态路由将输入分配给不同的 “专家” 网络处理。在大规模应用中，MoE 通常需要分布式计算来扩展性能。

### **结构组成**

一个典型的 MoE 层由以下几个部分组成：

1.  **Experts（专家）**  
    

-   每个“专家”是一个独立的前馈神经网络（FFN），例如一个 MLP。
-   假设有 EE 个专家，每个专家负责处理某一类输入特征或任务。
-   所有专家并行存在，但每次前向传播只激活其中一部分。

1.  **Gating Network（门控网络）**  
    

-   一个小的神经网络（通常是一个线性层 + softmax），根据当前输入 token 决定“哪个专家最适合处理这个输入”。
-   输出是一个概率分布，表示每个专家对该输入的“权重”或“重要性”。

1.  **Routing Algorithm（路由算法）**

-   根据门控网络的输出，选择 Top-K 个专家来处理当前输入（通常 K=1 或 2）。
-   实现**稀疏激活**：只有被选中的专家参与计算，其余不运行。

1.  **Output Combination（输出融合）**

-   将被激活专家的输出按门控权重加权求和，得到最终输出。

### MoE 分布式

MoE 为什么要分布式计算：

1.  **参数量巨大**：MoE 模型可以拥有数万亿甚至更多的参数，远远超过了单个GPU或TPU的存储容量。
2.  **计算需求高**：虽然每次前向传播只激活部分专家，但由于专家数量庞大，总的计算需求仍然非常高，单设备难以承担。
3.  **数据并行不足**：传统的数据并行方式（即将批次的数据分割到不同的设备上进行处理）不足以应对MoE模型的需求，因为每个样本可能需要访问不同的专家。

分布式现有的策略：

-   **专家并行（Expert Parallelism）**：将不同的专家分配到不同的设备上。这种方式适合于当模型中的专家数量超过单一设备的处理能力时使用。
-   **数据并行（Data Parallelism）**：尽管单独的数据并行不足以满足MoE模型的需求，但它仍然可以与专家并行结合使用，通过增加批次大小来加速训练过程。
-   **模型并行（Model Parallelism）**：除了专家并行外，对于那些不能完全被分割成独立专家的部分模型，还可以应用传统的模型并行技术。
-   **混合并行（Hybrid Parallelism）**：结合上述多种并行方法，以充分利用不同设备间的计算资源和带宽优势。

### 分布式 MoE 的执行流程

### 1\. **前向传播示例**

1.  **输入处理**：  
    每个设备接收部分输入 Token（如设备 0 处理 Token 0~511，设备 1 处理 Token 512~1023）。
2.  **门控计算**：  
    每个设备本地计算门控结果，确定每个 Token 的 Top-K 专家。
3.  **Token 路由（All-to-All）**：

-   设备 0 将 Token 0~511 中需要专家 3 处理的 Token 发送给负责专家 3 的设备 2。
-   所有设备同时进行类似操作，形成全互联通信。

1.  **专家计算**：  
    每个设备使用本地专家处理接收到的 Token。
2.  **结果收集**：  
    通过反向 All-to-All 通信，将专家输出结果返回给对应设备。
3.  **合并输出**：  
    每个设备根据门控权重合并结果，生成最终输出。

### 2\. **反向传播与负载均衡**

-   **梯度计算**：类似前向传播，但数据流向相反。
-   **负载均衡**：  
    通过专家复制（Duplication）或动态调整门控策略，避免部分专家过载。

### 通信优化

MoE（Mixture of Experts）模型在处理大规模数据时，通信优化是确保其高效运行的关键。

-   **All-to-All 通信**：  
    所有设备同时发送和接收数据（每个设备将 Token 发送给对应专家所在设备）。
-   **分块传输**：  
    将大量 Token 分成小块传输，避免内存溢出（如`num_max_nvl_chunked_send_tokens`）。
-   **异步流**：  
    使用 CUDA Stream 并行执行计算和通信，隐藏延迟。
-   **RDMA/NVLink**：  
    利用高速网络和 GPU 直接通信，减少 CPU 参与。
-   **量化与压缩**：  
    降低通信数据精度（如 FP16/INT8），减少传输量。
-   **批处理**：  
    累积多个小批次的路由结果，批量传输以提高吞吐量。

### **主流框架**

-   **DeepSpeed-MoE**：  
    微软开发的 MoE 优化框架，支持专家并行、动态负载均衡。
-   **Megatron-LM**：  
    NVIDIA 的大规模 Transformer 训练框架，支持 MoE 分布式训练。
-   **JAX/Pax**：  
    Google 的自动并行框架，对 MoE 有良好支持。

* * *

## DeepSeek MoE 模型解析

### 一、DeepSeek V3 的 MoE 创新：[Routed Expert](https://zhida.zhihu.com/search?content_id=260986356&content_type=Article&match_order=1&q=Routed+Expert&zhida_source=entity) 与 Shared Expert 划分

DeepSeek V3 在标准 MoE 架构（如 GShard）基础上进行了优化，将专家细分为 “Routed Expert（路由专家）” 和 “Shared Expert（共享专家）”（见图1上半部分），进一步提升专家的专业化程度和知识利用效率：

> \[!note\] **Routed Expert（路由专家）** 每个路由专家专注于特定知识域（如逻辑推理、语义理解、事实问答等），门控网络通过精细化路由（如分组受限门控）将 Token 定向分配给最匹配的专家。这种 “专业化分工” 避免了单一专家承载过多知识导致的性能稀释，提升了特征提取的精准度。  
> \[!note\] **Shared Expert（共享专家）** 共享专家负责处理跨域通用知识（如基础语法、常见语义模式），所有 Token 都可能被路由到共享专家，避免路由专家因 “过度专业化” 导致的知识覆盖不足。同时，共享专家可隔离不同路由专家之间的知识冗余（如多个路由专家可能涉及的共性知识），减少重复计算。

-   **共享专家（Shared Experts）**：

-   所有 Token 都必须经过的专家，通常实现基础语言理解功能
-   常见 Kernel 操作：

-   多层 Transformer（自注意力 + 前馈网络）
-   词法分析与基础语义提取
-   位置编码与上下文整合

  
-   **路由专家（Routed Experts）**：

-   根据 Token 内容动态选择的专家，实现专业化功能
-   常见 Kernel 操作：

-   领域特定知识处理（如数学、代码、医学）
-   复杂推理与逻辑计算
-   长文本建模与连贯性维护

这种设计的优势在于：相比 GShard 等标准 MoE（所有专家平等参与路由），DeepSeek V3 的专家划分更符合 “知识分层” 逻辑 —— 通用知识由共享专家统一处理，专业知识由路由专家精准承载，最终在相同计算量下实现更高的模型性能。对于专家在多 GPU 上的分配策略，DeepSeek 采用**混合层次化分布**（见五 DeepSeek Auxiliary Loss for Load Balance）。

图1. DeepSeek-V3 Figure 2

!\[\[DeepSeek-V3 Fig2.png\]\]

### 专家在多 GPU 上的分配策略

DeepSeek 采用**混合层次化分布**：

```text
专家空间 = [
    0-31: 服务器0 GPU0 (共享专家)
    32-63: 服务器0 GPU1 (路由专家)
    64-95: 服务器0 GPU2 (路由专家)
    ...
    2048-2079: 服务器1 GPU0 (共享专家)
    2080-2111: 服务器1 GPU1 (路由专家)
    ...
]
```

-   **分配规则**：

1.  **共享专家**：每个服务器的首 GPU 存储完整副本（确保低延迟访问）
2.  **路由专家**：按哈希或轮询方式均匀分布
3.  **节点内分布**：通过 NVLink 互连的 GPU 组内，专家按 ID 取模分配 `gpu_id = expert_id % num_gpus_per_server`
4.  **节点间分布**：通过 InfiniBand 连接的服务器间，按块分配`server_id = expert_id // (experts_per_server)`

* * *

### 二、**分组受限门控 MoE** 机制数学表达

本节对应\[1\] 2.1.2 第一部分，是 DeepSeek - V3 中 **分组受限门控混合专家（MoE）** 机制的核心数学表达，用于实现 “稀疏化专家激活 + 分层知识路由”。

公式 DeepSeek-V3 (12-15) ：

$ \begin{align} \mathbf{h}_t' &= \mathbf{u}_t + \sum_{i=1}^{N_s} \text{FFN}_i^{(s)}(\mathbf{u}_t) + \sum_{i=1}^{N_r} g_{i,t} \cdot \text{FFN}_i^{(r)}(\mathbf{u}_t),\\ g_{i,t} &= \frac{g_{i,t}'}{\sum_{j=1}^{N_r} g_{j,t}'},\\ g_{i,t}' &= \begin{cases} s_{i,t}, & s_{i,t} \in \text{Topk}\left( \{s_{j,t} | 1 \leqslant j \leqslant N_r\}, K_r \right), \\ 0, & \text{otherwise}, \end{cases}, \\  s_{i,t} &= \text{Sigmoid}\left( \mathbf{u}_t^T \mathbf{e}_i \right), \\   \end{align} $

### 一、符号与变量定义

| 符号 | 含义 | 维度 / 类型 |
| --- | --- | --- |
| $\mathbf{h}_t'$ | 经过 MoE 层处理后的输出特征 | [batch_size,hidden_dim] |
| $\mathbf{u}_t$ | Transformer Block 输出的中间特征（MoE 层输入） | [batch_size,hidden_dim] |
| $\text{FFN}_i^{(s)}$ | 共享专家（Shared Expert）的前馈网络（全连接层 + 激活） | 函数（输入 $\mathbf{u}_t$，输出特征） |
| $\text{FFN}_i^{(r)}$ | 路由专家（Routed Expert）的前馈网络 | 函数（输入 $\mathbf{u}_t$，输出特征） |
| $N_s$ | 共享专家的数量 | 标量（如 4、8） |
| $N_r$ | 路由专家的数量 | 标量（如 64、128） |
| $g_{i,t}$ | 路由专家 $i$ 对 Token $t$ 的归一化门控权重 | $[0, 1]$ 标量 |
| $g_{i,t}'$ | 路由专家 $i$ 对 Token $t$ 的原始门控得分（未归一化、稀疏化后的值） | 标量（0 或原始得分） |
| $s_{i,t}$ | 路由专家 $i$ 对 Token $t$ 的原始门控得分（Sigmoid 激活前的 logit） | 标量（实数） |
| $\mathbf{e}_i$ | 路由专家 $i$ 的门控向量（用于计算 $s_{i,t}$ 的关键参数） | [hidden_dim] |
| $Topk(\cdot)$ | 取 Top-K 元素的操作（保留得分最高的 $K_r$ 个专家） | 函数 |
| $K_r$ | 每个 Token 激活的路由专家数量（如 Top-2、Top-4） | 标量（通常 $\ll N_r$） |
| FNN（Feed-Forward Networks） |   |   |
| ### 二、公式流程拆解：从门控计算到特征输出 |   |   |

从（公式）下往上、从（网络）后往前推导，这组公式可拆解为 **“门控得分计算 → 稀疏化与归一化 → 专家激活与特征聚合”** 三个核心步骤，完整描述了 MoE 层的前向传播逻辑：

### 1\. 步骤 1：计算原始门控得分（$s\_{i,t}$）

$ s_{i,t} = \text{Sigmoid}\left( \mathbf{u}_t^T \mathbf{e}_i \right) $

-   **作用**：衡量 Token $\\mathbf{u}\_t$ 与路由专家 $i$ 的 “匹配度”。
-   **实现细节**：

-   $\\mathbf{u}\_t^T \\mathbf{e}\_i$ 是向量点积，计算 $\\mathbf{u}\_t$ 与专家 $i$ 门控向量 $\\mathbf{e}\_i$ （对专家擅长的特征空间、数据分布的一种中心表示）的相似度（affinity scores，可理解为 “知识匹配度”）；
-   与 DeepSeek-V2 不同，DeepSeek-V3 采用了 Sigmoid 激活将相似度映射到 $\[0, 1\]$ 区间，得到原始门控得分 $s\_{i,t}$（值越大，专家 $i$ 越适配当前 Token）。

### 2\. 步骤 2：门控得分稀疏化（$g\_{i,t}'$)）与归一化（$g\_{i,t}$）

$ \begin{align} g_{i,t}' &= \begin{cases} s_{i,t}, & s_{i,t} \in \text{Topk}\left( \{s_{j,t} | 1 \leqslant j \leqslant N_r\}, K_r \right), \\ 0, & \text{otherwise}, \end{cases}\\  g_{i,t} &= \frac{g_{i,t}'}{\sum_{j=1}^{N_r} g_{j,t}'} \end{align} $

-   **作用**：实现 “稀疏激活”—— 仅保留得分最高的 $K\_r$ 个专家，其他专家贡献置 0，大幅减少计算量。
-   **实现细节**：

-   **稀疏化**：通过 $Topk$ 操作筛选出与当前 Token 最匹配的 $K\_r$ 个路由专家（如 $K\_r=2$ 时，仅保留得分前 2 的专家），未选中的专家 $g\_{i,t}'=0$；
-   **归一化**：对稀疏化后的得分做 Softmax（分母是所有路由专家稀疏化得分的和），确保 $g\_{i,t}$ 是概率分布（和为 1），用于加权聚合专家输出。

### 3\. 步骤 3：专家激活与特征聚合（$\\mathbf{h}\_t'$）

$ \mathbf{h}_t' = \mathbf{u}_t + \sum_{i=1}^{N_s} \text{FFN}_i^{(s)}(\mathbf{u}_t) + \sum_{i=1}^{N_r} g_{i,t} \cdot \text{FFN}_i^{(r)}(\mathbf{u}_t) $

-   **作用**：聚合 “共享专家” 和 “稀疏激活的路由专家” 的输出，生成最终特征。
-   **实现细节**：

-   **共享专家（$\\text{FFN}\_i^{(s)}$）**：所有 Token 都会激活全部共享专家（$N\_s$ 个），提供 “通用知识” 基础特征；
-   **路由专家（$\\text{FFN}\_i^{(r)}$）**：仅激活 $Top-K\_r$ 个专家，且通过 $g\_{i,t}$ 加权（匹配度越高的专家，权重越大）；
-   **残差连接**：$\\mathbf{u}\_t$ 直接相加（符合 Transformer 残差设计），避免信息丢失。

### 三、设计意图与技术创新

这组公式的核心目标是 **“在保持大模型性能的同时，通过稀疏化降低计算成本”**，对应 DeepSeek - V3 的两大创新：

1.  分层专家设计（共享专家 + 路由专家）  
    
2.  **共享专家（$\\text{FFN}\_i^{(s)}$）**：承担 “基础通用知识” 的计算，确保模型对所有 Token 都有稳定的特征提取能力，避免路由专家稀疏化导致的 “知识覆盖不足”；  
    
3.  **路由专家（$\\text{FFN}\_i^{(r)}$）**：通过稀疏激活聚焦 “个性化知识”（如特定领域、复杂模式），用少量计算量补充模型的专业化能力。

这种分层设计比传统 MoE（仅单一专家池）更高效，平衡了 “通用 vs 专用” 知识的计算成本。

1.  稀疏门控机制（Topk + 归一化）  
    
2.  **稀疏化（Topk）**：将路由专家的计算量从 $O(N\_r)$ 降至 $O(K\_r)$（通常 $K\_r \\ll N\_r$)，如 $K\_r=2$ 时计算量仅为 $2/N\_r$），使模型参数量可随 $N\_r$ 扩展（专家数量增加），但计算量不线性增长；  
    
3.  **归一化（Softmax）**：确保激活的专家权重合理分配，避免 “单一高得分专家主导输出”，提升特征融合的多样性。  
    
4.  与传统 MoE 的差异  
    

| 特性 | 传统 MoE（如 GShard） | DeepSeek - V3 公式设计 |
| --- | --- | --- |
| 专家类型 | 单一专家池（所有专家平等参与路由） | 分层设计（共享专家 + 路由专家） |
| 门控激活方式 | 通常用 Softmax 直接归一化所有专家得分 | 先 Topk 稀疏化，再 Softmax 归一化 |
| 计算量控制 | 依赖专家数量 N 和激活数 K（$O(K)$） | 分层控制（共享专家全激活 + 路由专家 $O(K_r)$） |
| 知识覆盖保障 | 依赖专家多样性，易出现 “冷门知识丢失” | 共享专家兜底通用知识，路由专家聚焦专用知识 |

* * *

### 三、 DeepSeek Auxiliary Loss for Load Balance无辅助损耗负载平衡

本节对应\[1\] 2.1.2 第二部分、\[2\] 2.2.2 2.2.3，详细拆解了 DeepSeek 中 **“设备级路由筛选 + 负载均衡 Bias”** 的设计逻辑，解释了 MoE 架构如何通过工程优化解决 “专家爆炸” 和 “负载不均” 问题：

### 一、Device-Limited Routing 设备限制路由机制

在使用专家并行（expert parallelism）的混合专家（MoE）模型场景中，由于 DeepSeekMoE 采用了细粒度的专家切分，激活的专家数量可能会较多。当为每个 token 选择路由专家（routed experts）时，如果这些目标专家分布在大量不同设备上，那么与 MoE 相关的通信成本会很高，因为 token 的 MoE 相关通信频率和其目标专家覆盖的设备数量成正比。为了限制这种通信成本，设计了设备限制路由（device - limited routing）机制。

1.  **核心目标**：确保每个 token 的目标专家最多分布在 M 个设备上，以此来约束与 MoE 相关的通信成本。
2.  **执行步骤**：

-   对于每个 token，首先从所有设备中筛选出 M 个设备，这些设备上有着亲和度分数（affinity scores，可理解为衡量 token 与专家匹配程度的指标 ）最高的专家。
-   然后，在这 M 个设备上的专家中，进行 Top - K 选择（即选出 K 个亲和度分数最高的专家作为该 token 实际要路由到的专家 ）。

1.  **实践效果**：在实际应用中发现，当 $M\\geqslant3$ 时，这种设备限制路由机制能够取得良好的性能，其表现大致与无限制的 Top - K 路由（即不限制设备数量，直接在所有专家中进行 Top - K 选择 ）相当。这意味着在控制通信成本（通过限制设备数量）的同时，并没有过多牺牲模型性能，实现了通信成本与模型性能之间的较好平衡。

**专家在多 GPU 上的分配策略分配规则**：

1.  **共享专家**：每个服务器的首 GPU 存储完整副本（确保低延迟访问）
2.  **路由专家**：按哈希或轮询方式均匀分布
3.  **节点内分布**：通过 NVLink 互连的 GPU 组内，专家按 ID 取模分配  
    `python gpu_id = expert_id % num_gpus_per_server`  
    
4.  **节点间分布**：通过 InfiniBand 连接的服务器间，按块分配  
    `python server_id = expert_id // (experts_per_server)`  
    

### 二、公式解析：带 Bias 的 Topk 路由

公式 (16) 如下，公式本质是 **“先设备筛选，再专家筛选”** 的两级路由机制，同时通过 $b\_i$ 实现负载均衡：

$ g_{i,t}' = \begin{cases} s_{i,t}, & s_{i,t} + b_i \in \text{Topk}\left( \{s_{j,t} + b_j | 1 \leqslant j \leqslant N_r\}, K_r \right), \\ 0, & \text{otherwise}, \end{cases} $

### 1\. 符号补充定义

| 符号 | 含义 |
| --- | --- |
| $b_i$ | 第 $i$ 个专家的 负载均衡偏置（Bias），用于调整专家的选中概率 |
| $s_{j,t} + b_j$ | 专家 $j$ 对 Token $t$ 的 带偏置亲和度（原始亲和度 + 负载均衡修正） |
| $M$ | 设备筛选阈值 |

### 2\. 两级路由逻辑

DeepSeek 的路由分两步：

-   **第一步：设备级筛选（隐式逻辑）** 限制专家分布在 M 个设备中，因此在计算 $s\_{j,t} + b\_j$ 前，会先筛选出 **亲和度最高的 M 个设备**，仅在这些设备内的专家参与后续竞争。 这一步是工程优化（未显式公式化），目的是 **避免专家切分过细导致的设备通信爆炸**（若专家分散在数百设备，跨设备通信成本会指数级上升）。
-   **第二步：专家级 Topk（显式公式）** 在筛选后的 M 个设备内，对专家计算 **“带偏置亲和度”** $s\_{j,t} + b\_j$，再取 Top-$K\_r$ 个专家激活。 这里的 $b\_i$ 是关键：若某个专家负载过高（被频繁选中），会通过调整 $b\_i$ 降低其 $s\_{i,t} + b\_i$ 的值，减少被选中的概率，从而 **平衡专家负载**。

### 三、核心设计：解决两大 MoE 痛点

### 1\. 痛点 1：专家切分过细 → 设备通信爆炸

-   **问题本质**： MoE 专家数量 $N\_r$ 增加时，若专家分散在大量设备（如 $N\_r=1024$ 分布在 128 设备），跨设备通信会成为瓶颈（每个 Token 需与多设备交互）。
-   **DeepSeek 解法**： 通过 **“设备级预筛选”**（限制最多 M 个设备），将专家路由限制在少数设备内，大幅减少跨设备通信次数。 例如：$M=8$ 时，Token 仅需与 8 个设备交互，而非全部 128 个，通信成本降为 $1/16$。

### 2\. 痛点 2：专家负载不均 → Routing 崩塌

-   **问题本质**： 若某些专家（如通用知识专家）被大多数 Token 选中，会导致：

-   **计算负载不均**：少数专家 GPU 满负载，其他闲置；
-   **Routing 崩塌**：模型过度依赖少数专家，知识表达能力下降（多样性不足）。

  
-   **传统解法**： 引入 **辅助损失（auxiliary loss）**，惩罚负载不均的专家（如让专家的选中概率尽可能均匀）。但辅助损失会增加训练复杂度，且可能与主任务（如语言建模）冲突。
-   **DeepSeek 解法**： 通过 **动态 Bias $b\_i$** 隐式调整专家的选中概率：

-   若专家 $i$ 负载过高（被选中次数多），则增大 $b\_i$ 的负值（或减小正值），降低 $s\_{i,t} + b\_i$，减少被 Topk 选中的概率；
-   若专家 $i$ 负载过低，则增大 $b\_i$ 的正值，提升被选中概率。 这种方式 **无需额外损失函数**，直接通过路由逻辑实现负载均衡，同时保留模型对 “高亲和度专家” 的偏好（与主任务兼容）。

### 四、与传统 MoE 的对比

| 设计维度 | 传统 MoE（如 GShard） | DeepSeek 优化 |
| --- | --- | --- |
| 专家分布限制 | 无设备级限制（专家可分散在任意设备） | 限制最多 M 个设备，减少跨设备通信 |
| 负载均衡方式 | 依赖辅助损失（auxiliary loss） | 通过 Bias $b_i$ 动态调整选中概率 |
| 路由复杂度 | 全局 Topk（所有专家参与竞争） | 两级路由（设备级预筛选 + 专家级 Topk） |
| 通信成本 | 高（跨多设备） | 低（仅与 M 个设备交互） |
| 训练复杂度 | 高（辅助损失与主任务耦合） | 低（Bias 嵌入路由逻辑，无额外损失） |

### 五、实际影响：模型效率与稳定性提升

1.  **训练效率**： 设备级预筛选减少跨设备通信，结合 Bias 实现负载均衡，可让 GPU 利用率从传统 MoE 的 60% 提升至 90% 以上（避免设备闲置或过载）。
2.  **模型稳定性**： 路由崩塌风险降低，专家知识更分散，模型在长文本、多领域任务中表现更稳定（如 DeepSeek-V3 在代码、数学任务中的一致性提升）。
3.  **可扩展性**： 支持更大规模专家数量（如 $N\_r=4096$），因为设备级限制和负载均衡让 “专家爆炸” 后的通信与计算仍可控。

参考： 1. [Parallelisms — NVIDIA NeMo Framework User Guide](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/features/parallelisms.html)

## 参考

1.  DeepSeek-V3 Technical Report
2.  DeepSeek-V2 Technical Report
3.  DeepEP