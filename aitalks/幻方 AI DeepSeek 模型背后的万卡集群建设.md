# 幻方 AI DeepSeek 模型背后的万卡集群建设

**作者：** AI闲谈

---

一、背景

幻方 AI 团队发布了一系列 DeepSeek 大模型，比如 DeepSeek-V2、DeepSeek-Math、DeepSeek-Coder 等。在 DeepSeek V2 中提出的 MLA（Multi-head Latent Attention）也广受好评。此外，DeepSeek V2 在强大性能的情况下还将 API 定价降低到 GPT-4 的百分之一，被称为“价格屠夫”，也由此引发大模型 API 的价格战。

本文中我们介绍一下幻方 AI 训练 DeepSeek 系列模型使用的大规模 GPU 集群以及相应的各种优化手段。

对应的论文为：[2408.14158] Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning

需要说明的是，本文中介绍的一些内容在我们之前的文章中已经详细介绍过，这里不再赘述，具体可以参考：

- [LLaMA 3 背后的大规模 GPU 集群 RoCE 网络建设](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247487544&idx=1&sn=1e500b3d9becc6ec19fc2912834beef7&chksm=c364d77df4135e6b2a2d500e013ea4f9b12eec9b2dcf50f834d9e703a7eed66e96e97e4bab95&scene=21#wechat_redirect)
- [Imbue-70B 的 AI Infra：从0到1搭建和运维4088 H100集群的最佳实践](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247487245&idx=1&sn=e71e3713fb39f0b9e0d308b058b43ce0&chksm=c364c848f413415e4f12b128a39b6cd618a1f29b78180e8f26aec6da649539a8ab91c591ce19&scene=21#wechat_redirect)
- [万卡 GPU 集群互联：硬件配置和网络设计](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247486775&idx=1&sn=abf7af24181cf5189e113fb161cc8d30&chksm=c364ca72f4134364f4e3fa4a971f767c2b07e6c2cae38c2a4ae28071fd330abaea68c36542c4&scene=21#wechat_redirect)
- [阿里 HPN：针对大规模 LLM 训练的万卡集群](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247487170&idx=1&sn=f07d6847526d1f317b361d04c9d0e72c&chksm=c364c987f4134091a5a86ec85112c6ec1e48fe645a1e7d8392e3695d1c16c72f41256c36eb13&scene=21#wechat_redirect)
- [HPN 7.0：阿里云新一代万卡集群网络架构](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247487094&idx=1&sn=f0a94bff3b3cc6e88cb95c8f82551e0c&chksm=c364c933f413402521586d8de7b9d274ea78e187d9222e645b450b6520bef32ffb5744424c69&scene=21#wechat_redirect)
- [万卡 GPU 集群实战：探索 LLM 预训练的挑战](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247486852&idx=1&sn=9f9dc1df99ab6aafb28e091f4532b89e&chksm=c364cac1f41343d7b10d9d234d1c7f3371d996afda01cb94d294a38cba4f1a14fe4594992aa2&scene=21#wechat_redirect)
- [剖析大规模 GPU 集群：针对 LLM 场景的挑战和优化](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247487054&idx=1&sn=fd540ee08fc40211d51856a146d22ac8&chksm=c364c90bf413401dc34fb9944f511a2960d4c532ea9bd8e4f88c696a5a7a6c58e549c73a8e27&scene=21#wechat_redirect)
- [阿里 C4：通信驱动加速大规模并行训练效率](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247487014&idx=1&sn=c49df9bd2de03acfae39bf4dce1c84b6&chksm=c364c963f4134075edee235c744c68c3f411ac7cdd1b9847de9333169292ff375a56c7d8ebd0&scene=21#wechat_redirect)
- [大规模分布式 AI 模型训练系列——数据并行](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247487775&idx=1&sn=52981f832c8ad7c9b111e37c0e788c3a&chksm=c364d65af4135f4cc999fd39659936f42bedc7faebeb2e2a674d5feb064bf50b68a6d412b89b&scene=21#wechat_redirect)
- [大规模分布式 AI 模型训练系列——张量并行](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247487815&idx=1&sn=69601e66f3f8413b5afbd8149b989ea7&chksm=c364d602f4135f1495f0c5e52bf911b26b528bd85f2ad1d2a97d93a358592676223bb9950ee1&scene=21#wechat_redirect)
- [大规模分布式 AI 模型训练系列——流水线并行](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247487851&idx=1&sn=7e18c1e0196193157081c4954c97c1af&chksm=c364d62ef4135f386b1f93bc1cd530116cd36a9002ab2f568373a978c6573f2b57661b2e21f3&scene=21#wechat_redirect)
- [大规模分布式 AI 模型训练系列——专家并行](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247487895&idx=1&sn=e2133a3052722c7c4e1d18f3053a6600&chksm=c364d6d2f4135fc49e3d380b0201678cadd3b741ce056baed4b7555d009c5a1dd36df1fea99d&scene=21#wechat_redirect)
- [](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247487895&idx=1&sn=e2133a3052722c7c4e1d18f3053a6600&chksm=c364d6d2f4135fc49e3d380b0201678cadd3b741ce056baed4b7555d009c5a1dd36df1fea99d&scene=21#wechat_redirect) [大规模分布式 AI 模型训练系列——序列并行](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247487939&idx=1&sn=1fe262d9316c6c09e9a267b683cd0b89&chksm=c364d686f4135f902e20e5d3b861f808d86ef77df158ccb3779573e469e3726b4c067fc54401&scene=21#wechat_redirect)

## 二、摘要

深度学习 （DL） 和大型语言模型 （LLM） 的快速发展对计算能力和带宽的需求呈指数增长。此外，更快的计算芯片和互联的成本也往往很高，这大大增加了高性能计算（HPC）的构建成本。为了应对这些挑战，作者提出了 Fire-Flyer AI-HPC 架构、软硬件协同设计框架及其最佳实践。对于深度学习训练，作者部署了配备 10000 个 PCIe A100 GPU 的 Fire-Flyer2，实现了接近 DGX-A100 的性能，同时将成本降低一半，能耗降低 40%。作者还专门设计了 HFReduce 来加速 AllReduce 通信，并采用许多措施来保证计算-存储网络无阻塞。其软件栈包括 HaiScale、3FS 和 HAI-Platform，作者通过重叠计算和通信实现了更好的可扩展性。

本文中涉及的关键技术点为：

- Network Co-Design：集成了计算-存储网络的两层 Fat-Tree 网络。
- HFReduce：为了适配器 PCIe 架构的集合通信库。
- HaiScale：基于 PCIe 架构优化的分布式并行方案。
- 3FS Distributed File System：解决 AI 任务下大数据的 I/O 瓶颈问题。
- HAI Platform：提供任务调度，容错等能力，以便增加利用率，降低成本。

PS：

- 本文中提到的 10000 卡 A100 集群最开始应该不是为了大规模 LLM 训练搭建，可能没有太大的网络通信需求；而随着大模型的发展，向这个方向转换时为了解决网络问题进而提供了一系列的解决方案，比如增加 NVLink Bridge。实际上针对大规模 LLM 推理场景，采用 PCIe GPU + NVLink Bridge 也是个不错的方案。
- 本文中的各种实验都是针对 PCIe 架构展开，也并没有提供业内比较常见的 MFU 指标，虽然其相比 Baseline 确实提升很多，但依然没有一个明确的对比。比如当前在 DGX A100 上的大规模训练通常能达到 50%-60% 的 MFU。

## 三、Fire-Flyer 2：网络架构

### 3.1 PCIe A100 GPU 架构

在 NVIDIA 官方 DGX 方案中，通常会采用 SXM GPU，有 NVLink 和 NVSwitch 实现高速互联，而且通常也会为每个 GPU 配备一个高速 IB 网卡（A100 通常是 200 Gbps）。而本文中作者采用的是 A100 PCIe GPU，无法使用 NVLink 和 NVSwitch 高速互联。此外 PCIe A100 和 SXM A100 在性能上也会略有差异，如下图 Table 2 所示。当然，PCIe GPU 服务器的成本和功耗也会更低一些。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgaSXiadEicTRsYxMtjoaxjOc2l3IgRoW9Iib4ZrutxPTIZ7icYwGHDlwNNjsndC9CxkZETVpQWH3TYZA/640?wx_fmt=png&from=appmsg&randomid=xyjyjasp)

实际上 A100 的各个版本中（甚至 A800 系列），理论算力都是相同的，比如 FP16 Tensor Core 算力都是 312 TFLOPS。作者上图中 A100 PCIe 是 A100 SXM 的 83% 应该是实测性能：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgaSXiadEicTRsYxMtjoaxjOcaNiblA1GgQ7C6aVXpFCQF4FcVhJ99WLiaGic6dLWFE8ypGe2cmwD4jUmg/640?wx_fmt=png&from=appmsg&randomid=ozf6budv)

成本低的另一个原因是服务器中只配备一个 200Gbps 的 Mellanox CX6 IB 网卡，并且直连到 CPU，没有经过 PCIe Switch，类似于下图红框 NIC 和绿框 NIC 的区别。当然，这里其实还会引入一个问题，不同 NUMA（CPU）下的 GPU 通信，或者 CPU1 下的 GPU 要通过 NIC 通信则都需要通过 UPI，这也额外增加了一些开销。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgaSXiadEicTRsYxMtjoaxjOcD9QD8P08TujOzEMGQl9PpGeaeLwduOMaFFGX4XaSL5ZVS8cOs2rZDA/640?wx_fmt=png&from=appmsg&randomid=88rfvq89)

上面提到，作者采用的 PCIe A100，没有使用 NVLink + NVSwitch 实现全互联。为了缓解 GPU 间数据交互的瓶颈，作者采用折衷的方案，每两个 GPU 通过 NVLink Bridge 实现高速互联，如下图所示，8 个 GPU 共分为 4 组，每组 2 个 GPU 通过 NVLink Bridge 连接。（PS：需要说明的是，作者早期的服务器没有 NVLink Bridge，而是后期为了适应 LLM 的需求新增加的）

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgaSXiadEicTRsYxMtjoaxjOc060VN95b1eBpyXSm6icDCfyVoP2UjlcfH6h3C9HsqOIOZ13GBtX2OicQ/640?wx_fmt=png&from=appmsg&randomid=x0bqvwwt)

### 3.2 网络拓扑

如下图所示为本文作者提出的两层 Fat-Tree 网络拓扑：

- 共包含两个 Zone。两个 Zone 的 Leaf Switch 直接通过 2 个 40-Port 的 Switch 互联（我们这里称作 Zone Switch），而不用经过 Zone 内的 Spine Switch。也就是 2 个 40-Port 的 Switch 共连接了 80 个 Leaf Switch。
- 每个 Zone 大概包含：
- 20 个 Spine Switch 和 40 个 Leaf Switch，Spine 和 Leaf 之间 Full Mesh 连接。
- 800 个 Node（包含 GPU Node 和 Storage Node，还有一些管理 Node）。
- 每个 Leaf Switch 40 个 Port：
- 20 个 Port 连接 Spine Switch。
- 1 个 Port 连接中间的 Zone Switch。
- 15 或 16 个 Port 连接 GPU Node，也就是每个 Zone 有 [40*15=600, 40*16=640] 个 GPU Node。（PS：论文中只说总共大约 1250 GPU Node，每个 Zone 大约 600 GPU Node，因此这里只能推测）
- 2 或 4 个 Port 连接 Storage Node。（PS：论文中提到两个 Zone 总共大约 200 个 Storage Node，但又介绍每个 Zone 800 个 Node。后文还提到包含 180 个 Storage Node，平均来看每个 Leaf Switch 会连接 2-3 个 Storage Node，Storage Node 包含 2 个 200 Gbps 的 NIC，不确定是否会将一个 Storage Node 连接到不同的 Leaf Switch）

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgaSXiadEicTRsYxMtjoaxjOcTV1KKAMwNkF6ibUBIIXTs2LA3daYg7qic3Fx84JSkm75ok5pZlZZvf3A/640?wx_fmt=png&from=appmsg&randomid=glhfw17h)

### 3.3 成本

作者对比了本文的方案与其他方案需要的 Switch 数量以及成本，具体如下图 Table 3 所示：

- 本文：122 个 Switch：(40+20)*2+2。
- PCIe 架构 + 3 层 Fat-Tree：每个 Node 1 个 NIC，则共需要 1600/20=80 Leaf Switch，80 Spine Switch 和 40 Core Switch，共 200 Switch。
- DGX-A100 GPU + 3 层 Fat-Tree：每个 Node 包含 8 个 GPU，有 8 个后向网络 NIC，因此 10000 个 GPU(NIC) 至少需要 10000/(40/2)=500 个 40-Port 的 Leaf Switch，500 个 40-Port 的 Spine Switch 和 320 个 Core Switch（PS：考虑 Full Mesh，这里不是 250），所以总共需要 1320 个 Switch。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgaSXiadEicTRsYxMtjoaxjOclCosvFOyadOibUjnQ0bgFXwfPzarLkyF71ghNuLt3Fn5sVg2CEgCNHQ/640?wx_fmt=png&from=appmsg&randomid=8dt4ajy2)

从上也可以看出，作者方案可以以 11600/23000=50.4% 的成本获得 83% 的 GPU性能。

### 3.4 下一代网络拓扑

作者也在准备构建下一代的 PCIe 架构集群来支持 MoE LLM 的训练，其包含大量的 All2All 通信，因此下一代架构中 GPU 和 NIC 会采用 1:1 配比，也就是每个 GPU 都有一个对应的 NIC，也考虑采用多平面网络。此外，会使用 RoCE 替代 IB Switch 以降低成本。使用 128 Port 的 400 Gbps RoCE Switch，4 平面的 2 层 Fat-Tree 网络可以支持 32,768 个 GPU。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgaSXiadEicTRsYxMtjoaxjOc0PRLXGZ8A0tfVgib5h5OaA8wKN8M9v6aw8icroUawgaGvrLYKD3ibD03A/640?wx_fmt=png&from=appmsg&randomid=nzof8rwt)

## 四、HFReduce：软硬协同网络设计

### 4.1 HFReduce 算法

在大规模分布式训练中，AllReduce 是一种非常常见的集合通信操作，比如不同 Data Parallelism 之间的梯度聚合操作。而 NCCL 通常是针对节点内有 NVLink 高速互联或者都通过 NIC 方式通信的范式进行优化的。针对本文这种网络拓扑不一定能发挥最优的性能。如下图 Figure 6 所示为作者优化之后的 HFReduce 概览，其包含几步：

- 第一步：节点内 Reduce 操作。
- 第二步：节点间在 CPU 上进行 Reduce 操作。
- 第三步：将 CPU 上 Reduce 后的数据传输会 GPU。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgaSXiadEicTRsYxMtjoaxjOcr1sfnwkqzEmkVw2m8AXiaOCe7zth886dGEgjpOxQxIABLKm1HichUb6Q/640?wx_fmt=png&from=appmsg&randomid=42ejwwpv)

节点内的 Reduce 操作算法如下图 Algorithm 1 所示：

- 将数据分成多个 Chunk 分别处理，这样可以将 IO 和 Compute 充分 Overlap。
- 每个 Chunk 的数据都通过异步的方式传输到 CPU 内存，拷贝操作也可以使用 GPUDirect 来拷贝小数据（可以参考 NVIDIA 的 GitHub - NVIDIA/gdrcopy: A fast GPU memory copy library based on NVIDIA GPUDirect RDMA technology），或者使用 cudaMemcpyAsync 来拷贝大数据。
- 已经拷贝到 CPU 内存上的 Chunk 可以执行 Reduce 操作，最终的结果也都是在 CPU 内存中。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgaSXiadEicTRsYxMtjoaxjOcg0pudFYrPxaGHAolc2ibQ3X6g5GgWMKkU1PkmicaHlHEMf19cMibpjE0Q/640?wx_fmt=png&from=appmsg&randomid=neaxtycb)

节点间的 Reduce 操作算法如下图 Algorithm 2 所示：

- 使用 Double Binary Tree Algorithm 算法实现节点间的 AllReduce 操作，节点间传输通过 RDMA 实现。
- 最后将计算完的数据通过 PCIe 传输到 GPU 显存中。此处的 Host to Device 操作也可以通过 GPUDirect 操作来同时写到同一个 NUMA 下的 4 个 GPU，而减少对 Host Memory 的读取（利用 CPU Cache）。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgaSXiadEicTRsYxMtjoaxjOcq5J2icVov9NmWp7tGA6ANnP1ia8PvPd4uzestTyoAFs6BVWQ2trFOrSQ/640?wx_fmt=png&from=appmsg&randomid=o4rdpe47)

### 4.2 HFReduce 对比 NCCL

针对本文的网络拓扑，作者提出的方案相比 NCCL 有 2 个优势：

- 减少了 PCIe 带宽开销：假设有 n 个 GPU 参与通信，在 NCCL 的 Ring 拓扑中每个数据单元需要 2n-1 次传输，对 PCIe 通信要求比较高。而 HFReduce 中，每个数据单元只需一次 D2H 和一次 H2D，这对于本文这种 PCIe 受限场景更加友好。
- 没有 GPU Kernel 开销：HFReduce 使用 GPU 的 Copy Engine(CE) 来执行异步的数据传输，而 NCCL 的 AllReduce 操作是使用 GPU Kernel 来完成。

如下图（a） 所示，本文的方案在执行 186MiB 数据的 AllReduce 时相比 NCCL获得了更高的带宽。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgaSXiadEicTRsYxMtjoaxjOcCw9zRiahVgXmZUvWkzAqyfBAYWXjRKcqJ1y7H6kUQxmhJR6aGtkSdLQ/640?wx_fmt=png&from=appmsg&randomid=epnvi3tl)

### 4.3 HFReduce with NVLink

我们前面提到过，作者在每两个 GPU 上添加了 NVLink Bridge，可以达到 600 GB/s 的高速通信带宽。而上述标准 HFReduce 并没有利用上 NVLink，因此作者也进一步探索了带有 NVLink 的 HFReduce。具体来说，在数据传输到 CPU Memory 之前，先在 2 个 GPU 上执行 Reduce；然后在结果返回时再将结果切分到对应的 2 个 GPU。

作者进一步测试了相应的通信带宽，如下图（b）所示，基本可以达到上述（a）中不带 NVLink 的 2x。其中蓝色为跨 Zone 的情况，因为一个 Leaf Switch 下有 15 或16个 Node，也就是 128 GPU，因此也只考虑超过 128 GPU 的情况：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgaSXiadEicTRsYxMtjoaxjOclljb9hBvboyx7BOYicIAQwFMrT7tvPxdk5YKUXHtNxruvssf0aJ7GAg/640?wx_fmt=png&from=appmsg&randomid=2u2gyqb5)

### 4.4 深入分析 HFReduce

实现中的关键技术决策：

- GPUDirect：使用 GPUDirect 加速 D2H 中的小数据拷贝，同时使用 GPUDirect 减少 3 倍的 H2D 开销。
- 节点内规约：使用 SIMD 指令完成 CPU 上的规约操作，支持 FP32、FP16、BF16 和 FP8。
- NUMA 感知：D2H 的目标内存会分配到 2 个 NUMA 对应的内存，以实现最大带宽。CPU Reduce 和网络传输的数据内存绑定在 IB NIC 对应的 NUMA，以尽量减少通过 UPI。
- 节点间规约：使用 Double Binary Tree 实现 AllReduce，避免额外的开销。

克服 EPYC Rome CPU 的限制：作者找 AMD 和 NVIDIA 的工程师帮忙定位了 PCIe 架构下通信的次优问题。最终发现 EPYC Rome CPU 不支持 chained write 功能，这个功能可以大幅提升 GPU 和 IB 之间的 PCIe P2P 带宽。作者测试发现，Rome CPU 上 IB NIC 和 GPU 的极限带宽在 9GiB/s，这也就可以解释上述 NCCL 的 AllReduce 带宽不超过 4GB/s。而 HFReduce 通过在 CPU 上进行 Reduce，在 CPU 和 IB 之间传输数据来规避这一问题。

HFReduce 的瓶颈：作者统计了一个 Node 上的所有内存操作：

- D2H 需要 8 次写操作（8 个 GPU）。
- 节点内 Reduce 涉及 8 次读操作和 1 次写操作。
- 节点间 Reduce 涉及 IB send 2 次读操作，IB receive 2 次写操作，以及 1 次 add 操作。
- H2D 利用 GPUDirect 执行 2 次读操作（8 次降低到 2 次）。

整体来说，上述内存操作相比 GPU 上的数据大小涉及 24x 的放大。一个 16 Channel 的 DDR4-3200MHz 内存，理论最大内存带宽为 320GB/s，对应理论最大 HFReduce 带宽为 320/24=13.3GB/s，而作者实测只有 8GB/s。

上述问题的主要原因是 EPYC CPU 的另一个限制，本文中作者的 GPU5 和 GPU6 直接通过相同的 PCIe Host Bridge 连接到 CPU。而 AMD EPYC Rome 和 Milan CPU 中 PCIe Host Bridge 的最大带宽为 37.5GB/s，即使 PCIe 4.0x16 从 GPU 到 CPU 可以实现 27GB/s。但是当 2 个 GPU 同时传输数据时将受到上述 37GB/s 的限制，也就是说平均最大只能达到 19GB/s。如果考虑双向传输，带宽瓶颈会更加明显。而作者加装的 NVLink Bridge （GPU5 和 GPU6 通过 NVLink Bridge 互联）可以提供一种有效的方案来缓解这个问题。此外，即使 AMD EPYC Genoa 也同样面对这个问题。

## 五、HaiScale：针对 DL 训练优化

### 5.1 HaiScale DDP

Pytorch DDP 会使用 NCCL 用于梯度聚合时的 AllReduce 操作，而本文中，作者使用 HFReduce 替换 NCCL。如下图（a）所示，训练 VGG 模型时，基于 HFReduce 的时延几乎是 Pytorch DDP（NCCL）的一半。同时，从 32 GPU 扩展到 512 GPU 时可以获得 88% 的线性加速。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgaSXiadEicTRsYxMtjoaxjOcvKuuKumdAbl1FhnRuicAIUfJRFCED7TpdQL7E0RuMrMEWkGh4GzracQ/640?wx_fmt=png&from=appmsg&randomid=icf9yk8k)

### 5.2 LLM 训练优化

针对 LLM 训练，作者同样优化了 DP、PP、TP 和 EP。

将 NVLink Bridge 连接的 2 个 GPU 用于 TP，实现高速互联。（PS：通常使用 NVLink + NVSwitch 的方案可以更好的是指 8 GPU 的 TP）

针对 PCIe 架构优化 PP。一台机器只有 1 个 NIC，使用 PP 时可能存在瓶颈，为此，作者在调度时将不同的 DP Rank 调度到同一个 Node 上，这样可以交错 DP 和 PP。如下图 Figure 9（a）所示，训练 LLaMA 13B 时，GPU 数从 32 扩展到 512，每一个 Step 的 Latency 从 64.118s 减少到 9.717s，获得了理论加速 91% 的加速效果。如下图 Figure 9（b）所示，DeepSeek-MoE 16B 训练时同样获得了理论加速的 92.92%。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgaSXiadEicTRsYxMtjoaxjOcrRAEnibZ8ibmJiaLGhqo5HEQq4HrQc2hMdQhRHYhcsf0oiaK8CT3bRDpUw/640?wx_fmt=png&from=appmsg&randomid=u0agqsd1)

HaiScale FSDP：此外，作者也对 FSDP 进行了适配和优化，如下图（b）所示，从 16 GPU 到 128 GPU，HaiScale 可以获得 95% 的加速。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgaSXiadEicTRsYxMtjoaxjOcEdDP6icOrnRLibQIX3xgBrTvHAfG50iaTjIsWFZFFx1rHZEeOluG2cViaw/640?wx_fmt=png&from=appmsg&randomid=vru34blc)

## 六、联合优化

### 6.1 计算-存储网络拥塞最小

如前所述，作者的网络方案中计算和存储在一个网络中，相较而言，之前的方案中往往是计算网络是高速后向网络，而存储网络是前向网络。因此，为了实现最大带宽，必须隔离不同类型的流量，避免相互干扰并造成网络拥塞。具体来说，作者实施了以下几个措施。

不同流量区分：在典型的训练任务中，有 4 种不同类型的流量：HFReduce 通信，NCCL 通信，3FS 存储流量和其他流量。作者利用 IB 的 Service Level（SL）技术，在节点之间建立连接时为其分配不同的 SL 值，并将 SL 映射到 IB 物理队列虚拟通道（VL），使用虚拟通道可以确保不同通道中的流量不会相互干扰。最终，通过配置它们的比例实现流量隔离，从而防止 Head-of-Line（HOL）阻塞和不同的流量冲突引起的网络阻塞。

拓扑调整和路由优化：在高吞吐存储场景中，存在许多 incast 通信模式，导致拥塞。针对这种情况，作者采用静态路由策略，将存储流量均匀分散在不同 Leaf -> Spine 连接，并将各种节点（存储、计算、管理）均匀分配到 Leaf -> Spine 连接。

NCCL 优化：调整了 NCCL 拓扑，以便调整同一个 Node 内的 IB NIC 和 GPU 的路由。可以减少 CPU chiplet 互联导致的 PCIe 拥塞。此外，通过使用 PCIe Relaxed Ording 进一步减少拥塞并增加带宽。

3FS 网络调优：3FS 实现了一个请求到发送的控制机制来缓解拥塞。

### 6.2 3FS 高吞吐分布式存储

如下图 Table IV 为本文的 Storage Node 配置，可以看出，其包含 1 个 CPU，2 个 200 Gbps NIC 和 16 个 15.36TB 的 SSD。

- 总共 2880 NVMe SSD，可以提供 20 PiB 的存储（有1个额外的存储副本）。
- 总共可以提供 180*2*200 Gbps = 72 Gbps = 9 TB/s 的理论带宽，实测可以达到 8 TB/s。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgaSXiadEicTRsYxMtjoaxjOc4M0LbCpfTUFaIN8nbejVic5cA8Dwy0Qe8OSNtbYoCf1Aj9SnVtm9l1Q/640?wx_fmt=png&from=appmsg&randomid=3wvxdibh)

3FS 系统包含 4 个角色：Cluster Manager、Meta Service、Storage Service 和 Client。其中 Storage Service 会部署在每个 Storage Node 上，每个 Storage Service 都能提供等分的带宽。根据这个设计，每个 Client 都可以访问每个 Storage Service。峰值负载时，作者在 Client 观察到 Incast 拥塞，为了缓解这个拥塞，作者在 Storage Service 和 Client 之间实现了一种请求发送控制机制（request-to-send），这种机制会增加端到端 IO 延迟，但又是实现可持续高吞吐的必要手段。

除此之外，还基于 3FS 实现了 3FS-KV，是 DeepSeek LLM Inference 中实现分布式 Context Caching 的关键所在。

### 6.3 HAI Platform

作者很早就开源了其对应的分布式训练平台，具体可以参考源码（GitHub - HFAiLab/hai-platform: 一种任务级GPU算力分时调度的高性能深度学习训练平台）和文档（欢迎来到HAI Platform 官方文档）。这里不再介绍。

## 七、稳定性和鲁棒性

### 7.1 Checkpoint 管理

在超大规模训练中，各种异常是在所难免的，为了减少异常导致的计算浪费，通常都会采用 Checkpointing 机制，定期保存 Checkpoint。本文中 Checkpoint 的保存同样依赖上述的 3FS，每个 Node 可以提供 10 GiB 的带宽，所以通常可以在几秒时间完成 Checkpoint 的保存。在作者的训练过程中，通常是每 5 分钟保存一次，也就是每次异常最多浪费 5 分钟的训练。

### 7.2 验证

增强设备稳定性最好的手段就是在发生异常之前检测到异常。因此作者开发了一系列的验证工具来识别是否存在硬件故障，然后平台可以自动进行一些运维工作。比如从集群中屏蔽异常机器，不允许调度。验证主要包括下述部分：

- 经常检测硬件，比如连接速度，状态。
- CPU 压测及内存带宽压测。
- GPU Memory 测试。
- GPU 运行 GEMM 测试。
- 节点内 AllReduce 测试。
- 存储带宽压测。

### 7.2 硬件故障

最常见的硬件问题包含两种：GPU Xid Error 和网络抖动。

如下图 Table V 所示，作者展示了常见的 Xid Error 和对应的原因：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgaSXiadEicTRsYxMtjoaxjOctfCkltcTCuYCBTyxiciagJUqlKWbrexyQxeO5R5OmVDV5VkiatAmibp8FQ/640?wx_fmt=png&from=appmsg&randomid=lttkbzou)

如下图 Table VI 所示，作者也展示了不同 Xid Error 的数量和比例，可以看出，NVLink Error 占比 42.57%，这可能和作者使用的 NVLink Bridge 有关。而 Xid 31 和 Xid 43 的软件错误总共超过了 50%，这种情况大部分是程序问题，如果排除程序问题那也基本可以确定是硬件故障。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgaSXiadEicTRsYxMtjoaxjOc9zpicXKvpKicTCicPvxBTUic10BetQFlMAJ4Hvkr0znNfbS4vEwY2a0W7Q/640?wx_fmt=png&from=appmsg&randomid=mqalclju)

如下图 Figure 11 所示，作者同样频繁受到网络抖动的影响：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgaSXiadEicTRsYxMtjoaxjOcQ1uJq4zEHkOKIMh9yib1sIpdZQNhCsvvawjICPKJdH2Q6ibtd2iaEhYmA/640?wx_fmt=png&from=appmsg&randomid=iav710ma)

## 八、参考链接

1. https://www.arxiv.org/abs/2408.14158
2. https://github.com/NVIDIA/gdrcopy
3. https://github.com/HFAiLab/hai-platform
4. https://hfailab.github.io/hai-platform/

