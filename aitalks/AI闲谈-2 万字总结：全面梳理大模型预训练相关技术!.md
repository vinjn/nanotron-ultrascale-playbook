# 2 万字总结：全面梳理大模型预训练相关技术

**作者：** AI闲谈

---

**## 一、引言

本文主要聚焦于大语言模型预训练相关阶段的技术和行业最新进展，其中包括常见的分布式策略、模型结构、常见的优化手段等。考虑到篇幅原因，暂不包含后训练、多模态等领域。

相关工作可以参考我们之前的文章：

- [综述：DeepSeek Infra/V1/MoE/V2/V3/R1 & 开源关键技术](https://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247489358&idx=1&sn=bda66cd5ffc40d6dd653d0bc076c7c09&scene=21#wechat_redirect)
- [LLaMA 3 技术报告解读：全面梳理 LLM 相关技术栈](https://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247487405&idx=1&sn=647217f38d505bbe15619217f17d20fb&scene=21#wechat_redirect)
- [大规模分布式 AI 模型训练系列——数据并行](https://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247487775&idx=1&sn=52981f832c8ad7c9b111e37c0e788c3a&scene=21#wechat_redirect)
- [大规模分布式 AI 模型训练系列——张量并行](https://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247487815&idx=1&sn=69601e66f3f8413b5afbd8149b989ea7&scene=21#wechat_redirect)
- [大规模分布式 AI 模型训练系列——流水线并行](https://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247487851&idx=1&sn=7e18c1e0196193157081c4954c97c1af&scene=21#wechat_redirect)
- [大规模分布式 AI 模型训练系列——专家并行](https://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247487895&idx=1&sn=e2133a3052722c7c4e1d18f3053a6600&scene=21#wechat_redirect)
- [大规模分布式 AI 模型训练系列——序列并行](https://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247487939&idx=1&sn=1fe262d9316c6c09e9a267b683cd0b89&scene=21#wechat_redirect)
- [大模型训练中的负载均衡：挑战与解决方案解析](https://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247489701&idx=1&sn=852a81b1228ee420a4371c7f57d1ebd4&scene=21#wechat_redirect)
- [大规模 GPU 集群运维实践：假装万卡 GPU 集群经验](https://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247489505&idx=1&sn=2c3fa700c352f7c4ce509192a769ec0e&scene=21#wechat_redirect)

## 二、模型结构

### 2.1 概述

当前 LLM 基本上都是 Decoder-Only 的 Transformer 模型，只不过都会进行一些修改。比如对 Attention 的修改衍生出来 Softmax Attention 系列和 Linear Attention 系列。而对 FFN 的修改衍生出了 Dense 模型和 MoE 模型。这个章节我们对这些模型结构的修改进行简单的总结。

### 2.2 Attention

#### 2.2.1 MHA、MQA、GQA、MLA

对于 Softmax Attention 系列，目前主要的是 MHA、MQA、GQA 和 MLA，主要区别如下图 Figure 3 所示：

- MHA：Transformer 模型的标准实现，每个 Attention Head 都有独立的 Q、K、V。
- 早期模型用的比较多，现在很少使用，至少是比较大的模型都没有使用。
- Inference 阶段的问题比较多，最主要的问题是 KV Cache 占比是几种方案里最大的，并且在 Continuous Batching 阶段 Attention 核心计算无法只用 Tensor Core，存在一定的局限性。
- MQA：所有 Attention Head 共享相同的 K 和 V（[2003.04641] MQA: Answering the Question via Robotic Manipulation）。
- 计算量和 KV Cache 都是最小的，也是对 Inference 阶段最友好的。
- 对模型效果影响较大，所以很少使用。
- GQA：Attention Head 进行分组，一组 Attention Head 共享相同的 K 和 V（[2305.13245] GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints）。
- 是 MHA 和 GQA 的折衷方案，KV Cache 大小处于 MHA 和 GQA 之间，计算效率也处于两者之间。考虑到 Tensor Core 要求矩阵乘法的 Shape 是 8 的整数倍，因此每组有 8 个 Head 的整数倍是最优的。如果每组中不到 8 个 Head，计算时可以 Padding，也能用上 Tensor Core，只是存在部分冗余计算。
- 是当前模型中最常见的的 Attention 方式。
- MLA：DeepSeek 2024 年 5 月在 DeepSeek V2（DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model）中提出，并在 DeepSeek V3 （[2412.19437] DeepSeek-V3 Technical Report）中沿用。核心思路是每个 Head 都有独立的 K 和 V，但它们可以投影和反投影到同样的、共享的 Latent KV。
- KV Cache 和 MQA 相当，明显少于 MHA 和 GQA，但效果还不错，和 MHA 相当。但是也会额外的增加一些计算量。DeepSeek 也开源了部分针对 MLA Inference 的代码实现（FlashMLA: Efficient MLA decoding kernels）。
- 当前主要是 DeepSeek V2、V3 以及最新开源的Kimi K2 模型（Kimi-K2 - a moonshotai Collection）中使用，其他模型并没有跟进。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64GptTSOaRo2I8O0lnzYVNicxw0qg9Y7f8rlWUJI61TkzWRL3AF0voImg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)

如下图 Table 1 可以看出，MLA 的 KV Cache 需求虽然依然大于 MQA，但明显优于 MHA 和 GQA，同时效果更好：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd649fYQKiazV4vL5Eu2j6m353egtqGfhiaaAKx0ZciaBPuCsKx25ibhrqJvMg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

如下图 Table 8 和 Table 9 所示，DeepSeek 团队在 DeepSeek V2 的技术报告中进行过一些消融实验，MLA 的效果优于 GQA，甚至优于 MHA：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64wWica9Mvc2JrYgwFG4p1dPX82s4zS9XSqFJzh4CicqQ6ibiamrsS6ribEXA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

#### **#### 2.2.2 MHA -> GQA/MLA**

模型在预训练阶段已经确定好是 MHA，还是 GQA、MLA。考虑到 GQA 和 MLA 在 Inference 阶段的优势，也就衍生了一系列将 MHA 转为 GQA、MLA 或者将 GQA 转为 MLA 的方案（PS：这些方案和预训练没有太大关系，更多是对 Inference 的帮助，不过这里也简单介绍）。

如下图 Table 1 所示，在 [2412.20677] Align Attention Heads Before Merging Them: An Effective Way for Converting MHA to GQA 中作者提出了将 MHA 转换为 GQA 的方案。其在 LLaMA2 7B 上做实验（共 32 个 Attention Head ）：

- 在 GQA-16 上能取得与 MHA 相当甚至更好的效果，但是 KV Cache 压缩比更高的 GQA-8（节约 75% KV Cache）和 GQA-4（节约 87.5% KV Cache）精度下降较大。
- LLaMA2 模型预训练数据比较少，模型训练不够充分，有效性有待商榷。
- 当前的 LLM 基本至少会默认采用 GQA，因此相应场景也就更少。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64ickTfCh96P5zcuDVzF4LDXI5HuZe2YJBRkIJTOjnKdUQib0g1gk1brSA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

如下图 Table 1 所示，在 [2502.14837] Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs 中，作者进一步探索了 MHA/GQA 到 MLA 的转换（PS：其实 GQA 可以看做 MHA 的特例，因此如果支持 MHA 转 MLA，就很容易支持 GQA 转 MLA）。

- 在模型比较小时损失比较大，在 LLaMA2 7B 上损失还可以接受。
- 上述评测还存在两方面问题：
- LLaMA2 模型没有充分训练，Baseline 比较低。
- LLaMA2 里面的评估任务中 OBQA 提升比较明显拉升了整体得分，在更多任务上的有效性有待商榷。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64wLzUkwiavceTrMqVNLfyA070b8P0H0mfCKqAG4CzsKWSIdHfffHKqBA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

如下图 Table 1 所示，在 [2502.07864] TransMLA: Multi-Head Latent Attention Is All You Need 中，作者同样提出了 MHA 到 MLA 的转换方案 TransMLA：

- 相比上述的 MHA2MLA，在无需训练的方式中提升较明显。而在加入一定训练数据后表现相当。
- 和 MHA2MLA 有同样问题，评测任务中考 OBQA 拉高得分，另外模型是 LLaMA2MLA。
- 如果与直接将 GQA 转为 MLA 的消融实验会更有说服力。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64icSXmiaRdaX8QNK1gBRFKgBRt1tE0qfBGkwoI4iclvLDVMEKHzh5dz0wQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)

#### **#### 2.2.3 Linear Attention**

由于 Softmax Attention 的二次方特性，当序列比较长时，Attention 的开销急剧增加。针对这种场景，Linear Attention 和 Sparse Attention 是两种常见的流派。在 Linear Attention 中比较常见的是 Mamba 和 RWKV 系列，不过在常见的开源模型中并非主流，业内质疑的声音也比较多。

如下图 Table 3 所示，在 RWKV-7[2503.14456] RWKV-7 "Goose" with Expressive Dynamic State Evolution 中，作者进行了相应对比实验：

- 针对小规模模型，基于 Linear Attention 的 RWKV7 模型确实能获得与基于 Softmax Attention 相当的性能。
- 主要实验聚焦在 3B 及以下模型，缺少 7B 及更大规模模型的实验。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd647YYoJa0SYddBXickhPby13sKDcLdrNgGic2gvE5H5HtcaQoCDMsjsm6g/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)

#### 2.2.4 Hybrid Linear Attention

相较于全部采用 Linear Attention 的方式，也有不少工作采用 Softmax Attention 和 Linear Attention 混合的方案。其相比全部 Linear Attention 更加保守，能够同时保留 Softmax Attention 的高精度以及 Linear Attention 在长序列的高效率。

这类工作中规模最大，影响力最大的是 MiniMax 的开源大模型 MiniMax-01 系列模型（[2501.08313] MiniMax-01: Scaling Foundation Models with Lightning Attention）。如下图 Figure 3 所示，其最大的 MiniMax-01 W456A46 模型包含 80 个 Transformer Block，其中每 7 个 Linear Attention 接一个 Softmax Attention（也就是下图中的 M=7）。并且 Softmax Attention 采用了 GQA，而 Linear Attention 也采用了定制化的高性能实现 Lighting Attention。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64wevbiaRpxRpjagrvJH8c4vnD82pNQhBd6kNRXjwvvtmO56gbmZfGRRw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=7)

如下图 Figure 8 所示，作者也对比了不同机制的训练吞吐。可以看出，随着序列长度增加：

- Softmax Attention 的效率逐渐降低，并且序列越长，下降比例越快。不过对于预训练常见的 8K 序列长度，基本还可以接受（PS：其实对于长序列也可以考虑采用 Sample Packing Mask 来优化，只不过是需要考虑负载均衡的问题）。
- Linear Attention 的 Lightning、HGRN2、Mamba2 几乎都能维持性能不降。（PS：也可以看出，对于 LLM 预训练常见的 4K、8K 训练长度，Linear Attention 没有特别明显的优势）
- Hybrid-Lightning 虽然也会出现性能下降的问题，但是明显好于 Softmax Attention。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd645DneJoyZTGCKthiasjZgic5EJXk08d6znKTtibwSXKkfZyGBumib1YwEdA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=8)

腾讯同样也提出了混合结构的 Hunyuan-TurboS W560A56 模型（[2505.15431] Hunyuan-TurboS: Advancing Large Language Models through Mamba-Transformer Synergy and Adaptive Chain-of-Thought），其中 Softmax Attention 与 Mamba2 Layer 的比例大约为 1:8（7 vs 57）。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd646Ra0uwaSPOJmosNraVs1G8AyyqrmO5wWI08fNfWp2Fs0y88HTFvwWw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=9)

如下图 Table 2 所示，其效果也还不错，与 DeepSeek-V3、Qwen3-235B-A22B 相当：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64WhWQgVjwhz6QlkTAuyn4ZlCiav02tgECibVEm4Aicgz7MSYiaicnEiaiaXmKQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=10)

除了 MiniMax 和 Hunyuan 外，NVIDIA 也发表过基于 Mamba 的混合模型 Mamba-2-Hybrid 8B（[2406.07887] An Empirical Study of Mamba-based Language Models），其混合结构如下图 Table 6 所示，共 28 个 Attention Layer，其中：

- 24 个是 Mamba2 的 Linear Attention
- 4 个是 Softmax Attention，并且采用了 GQA。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd644F8Bibt5dPbjSdqsjRN5Ax6sXmDXcfFl3N8RUEbAQic1siaBWic08jEaVA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=11)

如下图 Table 7 所示，在一些常见的任务上确实可以获得还不错的效果：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64jeqCfbVMcOeo2M2LPicjzkGldJI0ScZbw1NUlZ9dzIDqgJo7LndJcibA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=12)

#### **#### 2.2.5 Sparse Attention**

与 Linear Attention 类似，Sparse Attention 也是为了降低长序列的计算开销，只是方向不同。Sparse Attention 的主要动机是：Attention Score 是高度稀疏化的，只关注权重比较高的 Score 可以大幅降低计算复杂度并维持相当的精度。

Sparse Attention 在长序列 Inference 场景使用非常多，而预训练场景序列长度比较小，比较少使用。这里面比较早的工作是 Mistral 7B（[2310.06825] Mistral 7B）中使用的 Sliding Window Attention，是 Sparse Attention 的一种特例。如下图 Figure 1 所示，其相当于每个 Token 只关注附近的 Token。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64ic1uyOx6I9wI8nKTtmIJE5JnchSRHS2qOQvA1wM6ZU0ITWN5yaxF0pw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=13)

在这个领域，今年比较火的有两个工作（PS：都是针对长序列场景），一个是 Moonshot AI 的 [2502.13189] MoBA: Mixture of Block Attention for Long-Context LLMs，另外一个是 DeepSeek 的 Hardware-Aligned and Natively Trainable Sparse Attention。它们在某些方面都促进了一些共识：

- 长序列场景（Long Prefill 或 Long Decoding），Attention 是高度稀疏化的，也是高度动态化的。
- 固定 Pattern 的稀疏化方式往往很难保持精度，可学习 Sparse Pattern 是通用化且高精度的有效方案。
- Token 粒度的稀疏化很难充分发挥 GPU 算力，Block 粒度稀疏化是精度和性能（稀疏度、计算量）的良好平衡，基于此的高效 Block Sparse Attention 也成为标配。
- 当前常见的 LLM 通常会采用 GQA，也要充分结合 GQA 的特性来设计稀疏化方案，不然可能会影响整体的稀疏化程度。
- 在进行 Block 选择时并不需要使用 Block 内所有的 KV Cache，选择一个代表性的“聚类中心”即可，比如取 Avg 或者 Max。
- 不要随意永久性丢弃 Token，由于 LLM 的自回归特性，很难推测在后续的生成中是不是一定不需要某个 Token。这也就是为什么在 NSA 和 MOBA 中并不会节约 KV Cache 的存储空间。

如下图 Figure 1 所示为 Moonshot AI MoBA 的主要原理：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64xb5GfOltBNQPktbQvq6h5ibaDyzf3Pkj1yBdVadgKPmvu3UOXY2fPxw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=14)

如下图 Figure 2 所示是 DeepSeek NSA 的主要原理：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64RJ6pXghrfeSu8WLrdHicndZGKbH3zib6Z5k5n7kdXAkRhwFWVH2BpDXg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=15)

### 2.3 MoE

#### 2.3.1 粗粒度 MoE

2023 年的大语言模型还以 Dense 模型为主，2024 年初 Mistral AI 发布 Mixtral 8x7B MoE 模型（[2401.04088] Mixtral of Experts），引发了 MoE LLM 的热潮。不过早期的 MoE 还是比较粗粒度的专家，比如 Mixtral 8x7B 只有 8 个专家，后续的 Mixtral 8x22B 也是只有 8 个专家。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd649ncJtCVgWCCzRobqQSkLdX3V3LSdRsdzVNqico6ZKSD0PXQOYkgB0DQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=16)

早期的 MoE 模型训练都比较保守，往往采用先训练 Dense 模型，然后通过 Upcycling 的方式扩展到 MoE 模型，比如上述的 Mixtral 8x7B 是由 Mistral 7B Upcycling 而来。在昆仑万维的 Skywork-MoE（[2406.06563] Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models）中也对相应方案有所探讨，并将 Skywork 13B Dense 模型扩展为 Skywork 146B 的 MoE 模型（16 个专家）。

#### 2.3.2 细粒度 + 共享 MoE

DeepSeek 团队在 DeepSeek MoE 的技术报告（[2401.06066] DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models）中提出了细粒度专家+共享专家方案，并且在后续模型中一直延续。如下图 Figure 2 所示，DeepSeek MoE 模型主要有两点改进：

- 细粒度专家（Routed Expert）：常见的 MoE 模型中通常是 8 或 16 个专家，而这里会将一个大专家切分为 M 个小专家。比如原来从 16 个专家中选择 Top 2 大概有 120 种可能；而同样计算量的 64 个专家（M=4）中选择 8 个，对应了 4,426,165,368 种可能。
- 共享专家（Shared Expert）：额外增加了 1 个或多个共享专家，用于捕获通用知识，每个 Token 都会经过这些共享专家。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64dazF3JvHvuFibibqVOAcxNNNXwyU5UzciakPIzjKaUWkUhv75Lr7zBrRQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=17)

3 个模型的具体配置如下所示，需要说明的是，3 个模型中都未使用 GQA，而是使用的 MHA（PS：这个应该是 23 年的工作，正好是 MHA 往 GQA 过渡的阶段）：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd6479YvqFqic9MicJibQ5E3cNqib4OJ0QRVCcqLk8bmO1va5iaaUqRc6InkbAg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=18)

在 DeepSeek MoE 阶段（24 年 1 月），细粒度 MoE 还没被广泛接受（当模型规模不大时，细粒度 MoE 对训练性能影响较大）。直到 DeepSeek V2（DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model）甚至是后续的 DeepSeek V3（[2412.19437] DeepSeek-V3 Technical Report），这种细粒度专家和共享专家的方式的方式才被广泛接受并得到大规模使用。

#### 2.3.3 Dense + MoE

除了 Softmax Attention 与 Linear Attention 的混合架构外，也有一些 Dense 模型和 MoE 模型的混合架构。这种方式最早出现在 Google 经典的 Switch Transformer模型中（[2101.03961] Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity），其同样是 MoE 相关工作中非常经典的 Paper。如下图 Figure 2 所示，单看架构图很容易误解为是一个纯粹的 MoE 模型（每一个 Transformer Layer 都包含 MoE Block），一些非官方的代码实现中也是如此。然而，实际上该模型是一个混合架构模型。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64vcxySOXrWmwia4J0tUR7h6oFVjP7zfcQTZGPrKuWQnh4DwGOOhCdkPw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=19)

如下图 Table 9 所示，其中提到 Expert Freq 为 1/2，表明 MoE Transformer Layer 和 Dense Transformer Layer 各占 1/2：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64sD0flVviaQNO6Sz03B8fEV6GSO20HkyBvbibHzscCSrR3yWnnLTn2LEQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=20)

Google 在此后的 GLaM 模型（[2112.06905] GLaM: Efficient Scaling of Language Models with Mixture-of-Experts）和 ST-MoE 模型（[2202.08906] ST-MoE: Designing Stable and Transferable Sparse Expert Models）中都沿袭了这种方式。不过这些都是在 ChatGPT 之前的工作。

在后续的 LLM 中比较少出现这种情况。最常见的是 DeepSeek 系列 MoE 模型以及沿袭 DeepSeek V3 的 Kimi K2 模型：

- DeepSeek MoE 包含 28 层，第 1 层是 Dense Layer，后续 27 层为 MoE Layer。
- DeepSeek V2 包含 60 层，第 1 层是 Dense Layer，后续 59 层为 MoE Layer。
- DeepSeek V3 包含 61 层，前 3 层是 Dense Layer，后续 58 层为 MoE Layer。
- DeepSeek K2 包含 61 层，第 1 层是 Dense Layer，后续 60 层为 MoE Layer。

DeepSeek 在 DeepSeek MoE 的 Paper 中提到，采用这种方式主要是第 1 层的负载均衡状态收敛很慢，因此将第 1 层保留为 Dense 层。Moonshot 在实现 K2 模型时同样发现第 1 层的 MoE 的 Router 很难做到负载均衡，但不同的是第 2 层并没有发现此现象，因此相比 DeepSeek V3，只是第 1 层使用 Dense 层，第 2 和 第 3 层都使用 MoE 层。

其实后续的很多其他工作也采用了类似的方案：

- 小红书 dots.llm1：62 层，第 1 层 Dense，后 61 层 MoE。
- 百度 ERNIE 4.5：54 层，前 3 层 Dense，后 51 层 MoE。

### 2.4 其他改进

#### 2.4.1 MTP

DeepSeek V3 模型同样采用了 DeepSeek V2 的 MLA 以及细粒度专家+共享专家的 MoE 结构。除了模型增大之外，模型层面最主要的变化是引入 MTP（Multi-Token Prediction），这个思路在 Inference 的投机采样中经常使用，只不过这里也可以帮助提升模型训练的效果。具体来说：

- 其中 Main Model 就是标准的 Next Token Prediction。
- MTP Module 1 用于预测下下一个 Token，MTP Module 2 用于预测下下下一个 Token（与 LLM 推理中常见的多头投机采样思路一致）。
- MTP Module 中的输入都包含两个部分，一个是上一个 Module 的 Output Head 的输入，以及上一个输入 Token，并且其中的 Embedding Layer 和 Output Head 都是共享自 Main Model，只有新增的 RMSNorm + Linear Projection 和一个 Transformer Block。由于这里有两个输入分别经过 RMSNorm 后 Concat 到一起，因此需要一个额外的 Linear Projection 进行降维，保持维度一致。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64fIl0h0ClKaBkicvIHebibRxwkfcxfQ90ubDRyCicZbP79Mcd5wAhG9RPw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=21)

MTP 策略主要用于提升 Main Model 的性能，因此在推理阶段，可以直接舍弃 MTP Module，Main Model 仍能独立且正常运行。此外，还可将这些 MTP Module 用于投机解码，以进一步降低生成延迟。

## 三、分布式并行策略

### 3.1 概述

LLM 模型规模很大，单 GPU 甚至单节点往往无法放下，即使可以放下也可能因为显存空间有限而限制各种优化策略的实施。为了解决这些问题并获得最大吞吐，通常会采用各种混合分布式并行策略来优化，常见的有 DP（Data Parallelism）、TP（Tensor Parallelism）、PP（Pipeline Parallelism）、EP（Expert Parallelism）和 SP（Sequence Parallelism）。除此之外，也有许多优化方案试图改进上述并行策略，以便获得更优吞吐。

### 3.2 DP

#### 3.2.1 概述

DP 是最常用的并行策略，因为它与其他并行策略正交，实现简单并且通信量相对不是很大，很容易扩展训练的规模。但是 DP 也存在一个比较明显的问题：在每个 DP 组内都有完整的模型、优化器状态和梯度副本，导致内存开销比较大。

#### 3.2.2 DeepSpeed ZeRO

为了解决内存开销大的问题，微软提出了 ZeRO 相关工作（[1910.02054] ZeRO: Memory Optimizations Toward Training Trillion Parameter Models）。如下图 Figure 1 所示，根据不同的程度充分将优化器状态（os）、梯度（g）和模型参数（p）切分到所有的设备中，也就是不同的 DP 组中会存储不同的优化器状态、梯度和参数切片。

- ZeRO-1（Pos）：将优化器状态切分到所有设备，每个设备还有全量的模型参数和梯度。优先切分优化器状态是因为其占用内存更多，并且与 Forward 和 Backward 的反向传播无关，只影响模型权重参数更新阶段。（PS：由于 ZeRO-1 中每个设备只需要对应部分的平均梯度，而不像 DP 那样需要梯度的 AllGather，因此总的通信量不变。也就是说，在极大降低显存开销的情况下并不会增加通信量，这也是为什么常见的并行方案中基本都会默认采用 ZeRO-1 或者叫 ZeRO-DP）
- ZeRO-2（Pos+g）：在 ZeRO-1 的基础上进一步切分梯度，切分梯度也不影响 Forward 过程。
- ZeRO-3（Pos+g+p）：在 ZeRO-2 的基础上进一步切分模型参数，会影响 Forward 阶段，需要 AllGather 所有参数才能计算，会引入更多通信。采用 ZeRO-3 几乎可以将内存需求降低到 1/N，其中 N 表示设备数，这里是 64。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64PftlYRclIVyge5SCiaoTBSDcUFZ2kRphOdERZNtmyWqOXN7Vun8HWCA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=22)

#### 3.2.3 PyTorch FSDP

与 DeepSpeed 的 ZeRO 优化方案类似，Meta 也提供了 PyTorch 原生支持的 FSDP V1（[2304.11277] PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel）和 FSDP V2（[2411.00284] SimpleFSDP: Simpler Fully Sharded Data Parallel with torch.compile） 方案。早期的 Megatron-LM 框架不支持 FSDP，限制了 FSDP 的发展，最近半年 NVIDIA 在 Megatron-Core 里实现了相应的能力并进行了一系列优化，也能获得很不错的吞吐。随着后续集群 Scale-Up 域的扩展（比如 NVL72），FSDP 也许会有更大的空间。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64RCxgFC1jibAwB5HVZyNlv4xyzBIckeoicj4YyEzJ7OMpYzoAMn3sejtA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=23)

### 3.3 TP

很多时候模型在单一 GPU 无法放下，此时通常会采用 TP 切分，对于 Transformer 模型而言，最常见的是 NVIDIA 在 [1909.08053] Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism 提出的切分方式。

如下图 （a）所示，MLP 层的两个 FC 采用先列切（A，Column Parallelism），然后行切（B，Row Parallelism）的方案，这样两个 FC 之间不用通信：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd640dknuokfWZFL2Dflc3oSzv8khkK8DF7xibTa0NKGuVTLVj08QAXxaPA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=24)

如下图（b）所示，由于每个 Head 的 Attention，Softmax 都是独立的，因此可以采用按照 Head 的方式切分（等价于 Column Parallelism），然后对之后的 FC 采用行切分（B，Row Parallelism），这样 Self-Attention 中间也不用通信：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64jZnW3rz8yiaZ7DLyVLGPFM98XKF01n4Bh8iba2GVtAny25BYztcXEYGw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=25)

然而，由于每个 Transformer Layer 的 Forward 和 Backward 都需要 2 次 AllReduce 操作，并且通信量大，为了避免 TP 通信成为瓶颈，通常会将 TP 切分到一个节点内，因为节点内的 8 个 GPU 可以充分利用 NVLink + NVSwitch 的高带宽（PS：这也是为什么 TP 通常不会大于 8）。

如下图 Figure 8 所示为 Megatron-LM 中一个 DP + TP 的混合分布式并行方案，总共采用 64 台 8 GPU 机器，共 512 GPU。

- 每台机器的 8 个 GPU 组成一个 Model Parallelism Group（TP），共 64 个 TP Group；每个 TP Group 内的 GPU 包含不同的模型参数，并且使用相同的训练数据。
- 所有设备的同号 GPU（比如 GPU 1，9，...，505）组成一个 Data Parallelism Group（DP），共 8 个 DP Group；每个 DP Group 内的 GPU 都有相同的模型参数，但是使用不同的训练数据。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64qMegLWjQz5ycJezo3Ilc7ZqtTibKywFsibaExciaDWxj524aegM35snNw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=26)

当然，TP 也存在明显的问题：TP 会对 Tensor 进行切分，从而可能降低矩阵计算的算术强度。对于比较大的 Dense LLM，由于 TP 通常不会很大，基本还能接受；但是对于比较小的模型，或者细粒度的 MoE 模型，其矩阵乘法的 Shape 本身比较小，TP 切分后对算术强度的影响比较大，会导致吞吐的明显下降，无法充分发挥 GPU 的性能，因此在细粒度 MoE 模型的专家部分比较少采用 TP 并行。

### 3.3 PP

#### 3.3.1 概述

PP 是另一种常见的模型并行策略，其同样是将模型切分成不同的部分，只不过与 TP 的切分方式不同，其主要是将模型的不同层切分到不同的设备上。

如下图 Figure 3 所示为使用 4 个设备进行 PP 训练的执行过程。其每一行代表一个设备，蓝色表示 Forward，绿色表示 Backward，Forward 和 Backward 中的数字指的是 Mini Batch 的 ID。由于是按层切分，并且同一时间只有 1 个 Mini Batch，每个设备都必须等待之前的设备执行完才能执行对应的 Stage，导致存在大量的 Bubble。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64wLxRCNHWmrOzp7FKwCLU9tawSNu9n5mZicYlpfsutltj2xLspibjDUJw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=27)

#### **#### 3.3.2 1F1B**

实际上当设备 1 执行完 Mini Batch 1 的 Forward 之后便可以开始 Mini Batch 2 的 Forward，以此类推。在调度的过程中，系统中的每个设备都会有两个选择：

- 对某个 Mini Batch 执行 Forward，进而可以将 Mini Batch 传递到下一个设备。
- 对另一个 Mini Batch 执行 Backward，进而确保学习向前推进。

如果始终优先考虑 Forward，则会导致阻塞 Backward，模型也就无法学习和更新，因为只有 Backward 后才能执行权重更新；同样，如果只考虑 Backward 优先调度，则会导致计算资源闲置，无法充分发挥算力。

为了避免上述问题，1F1B （1次 Forward，1次 Backward，[1806.03377] PipeDream: Fast and Efficient Pipeline Parallel DNN Training）调度机制应运而生。如下图 Figure 8 所示，4 个设备，分成 4 个 Stage。在起始阶段允许执行多个 Mini Batch 的 Forward，稳定后就保持 Forward 和 Backward 的交替执行，这样可以保证 GPU 在稳定状态下没有空闲，并且始终继续学习。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64bcicPicfgMiaOnsIc6OzGVW8tgUO24KX87kpelBPg1OOzemGcUaqQT59w/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=28)

上述的 1F1B 过程并不需要 Forward 和 Backward 一样长，实际上，Backward 总是大于 Forward（大约 2 倍），此时 1F1B 依然是有效的调度机制。

PP 相比 TP 来说，只用在相邻的 PP Stage 间进行 P2P 通信即可，其通信次数、通信量相对较少，因此比较适合跨节点间通信；除此之外，从上图也可以推测出，增加梯度累加的次数，也就是 Mini Batch 的数量，可以较大程度降低 Bubble 率，梯度累加次数越大，Bubble 率越小。

#### 3.3.3 Interleaved 1F1B

采用 PP 最需要关注的问题就是 Bubble 率，需要尽可能的利用流水线机制让所有设备处于工作状态，提升整体吞吐。

NVIDIA 在 [2104.04473] Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM 中提出的 Interleaved-1F1B 方案正是为了降低 Bubble 率。

如下图 Figure 4 所示为基于 1F1B 提出的 Interleaved 1F1B 调度方案。

- 上图为 1F1B：假设模型有 16 层，每个 Device 有 4 层。
- 对应 K=4 个 Device，Micro Batch 个数为 M=8，也就是每 M=8 个 Micro Batch 进行一次同步梯度更新。
- 模型被分为 4 个 Stage，Device 1 包含 Layer (0,1,2,3)，Device 2 包含 Layer (4,5,6,7)，Device 3 包含 Layer(8,9,10,11)，Device 4 包含 Layer(12,13,14,15)。
- 下图为Interleaved 1F1B：
- 与标准 1F1B 的主要不同是层的切分方式。
- 模型被分为 8 个 Stage，Device 1 包含 Layer (0,1,8,9)，Device 2 包含 Layer (2,3,10,11)，Device 3 包含 Layer(4,5,12,13)，Device 4 包含 Layer(6,7,14,15)。可以看出，相当于将模型切分为 8 个 Stage，但是交替放在 4 个 Device 上，下图中深色代表前 4 个 Stage（Layer 0-7），浅色代表后 4 个 Stage（Layer 8-15）。以此就可以实现更细力度的调度，减少 Bubble。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64ibpN0M2H4iclI2yM2p5ggqcXBkondq9EnHEictywkPP8WgX3u1fJLp9dQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=29)

#### 3.3.4 ZeroBubble

Sea AI-Lab 团队在 [2401.10241] Zero Bubble Pipeline Parallelism 中进一步提出了 Zero Bubble 方案，可以进一步降低 Bubble 率。

如下图 Figure 1 所示，ZeroBubble 中将 Backward 分成两个部分，一部分计算输入的梯度，一部分计算权重的梯度。这里计算输入的梯度有明确的依赖关系，也是链式法则不断传递的基础；而计算权重的梯度却没有明确的依赖，甚至可以滞后很多。此外，三个红色部分计算量相当，这也就是为什么之前 1F1B 或者 Interleaved-1F1B 中 Backward 的长度为 Forward 的 2 倍。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd649oK4MHpbEKmgsOfr9pgkbNjBRocDics5ZFJXIbyKf0CwAz5hv13oQnw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=30)

可以看出，ZeroBubble 的方式为降低 Bubble 率提供了更多的可能，然而也有一定局限性。首先，当梯度累积次数比较多时，Bubble 率本身不大，提升的空间也就比较有限；此外，通常优化方案中会将上述两个梯度的计算放在一个 Kernel 里，ZeroBubble 会将其变成两个 Kernel，有可能导致效率的降低。

#### 3.3.5 非均匀 PP

由于 LLM 模型中除了 Transformer Block 外还有一些其他组件，比如还有开始的 Word Embedding，结尾的 LM Head 以及 Loss 计算，对于 DeepSeek 模型还有 MTP Layer，如果按照 Transformer Block 平均切分会存在计算的负载不均衡；此外，由于 1F1B 的机制问题，PP Stage 越靠前的部分就会占用越多的 GPU 显存，导致显存的不均衡。

对于计算负载不均的问题，思路也比较简单，既然首尾 Stage 的计算负载更大，那么让首尾的层数少一些即可。比如：

- 智谱在 GLM-130B 模型（[2210.02414] GLM-130B: An Open Bilingual Pre-trained Model）时就提出将 72 层 Transformer Block 变成 70 层，PP 为 8，中间各 9 层，起始各 8 层，这样可以降低首尾 Stage 的压力。
- Meta 在 LLaMA3 405B 模型（[2407.21783] The Llama 3 Herd of Models）中同样采用了类似的方案，共包含 126 层，同样是首尾 Stage 少 1 层。
- 昆仑万维在 Skywork-MoE 也类似，24 层，由 [6, 6, 6, 6] 切分变为 [5, 5, 5, 5, 4]。

对于显存开销不均的问题，通常会使用重计算和 Offloading 机制来缓解，不过也需要注意重计算或者 Offloading 的粒度，通常不会对整个 Transformer Layer 进行，而是针对个别计算，以便尽可能降低重计算等带来的额外开销。

#### 3.3.6 DeepSeek DualPipe

对于 DeepSeek V3 而言，跨节点 EP 引入的通信开销导致计算与通信比约为 1:1，效率很低。为了应对这一挑战，作者设计了一种创新的流水线并行算法 DualPipe。

DualPipe 的核心思想是：将一对独立的 Forward 与 Backward Chunk 内的计算与通信进行 Overlap。特别地，对于 Backward Chunk，借鉴 ZeroBubble，将 Attention 与 MLP 的 Backward 分为两部分：Backward for Input 及 Backward for Weight。

如下图 Figure 4 所示，针对一对 Forward 与 Backward Chunk，重新排列这些组件，并手动调整 GPU SM 在通信与计算间的分配比例。在此 Overlap 策略下，能够确保 All2All 和 PP 通信在执行过程中完全隐藏，其中：

- 橙色表示 Forward
- 绿色表示 Backward for Input
- 蓝色表示 Backward for Weight
- 紫色表示 PP 通信
- 红色表示 Barrier 同步

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64lExm2ialZdDxiaAS35sfX7NoNeUnpZpCnibxiauTSquZ0Z8SgXcyI6QouA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=31)

完整的 DualPipe 调度如下图 Figure 5 所示，其采用双向 PP 调度，同时从流水线两端输入 Micro Batch，使得大部分通信得以完全 Overlap（PS：8PP，双向 20 Micro Batch，反方向 10-19 的 10 个 Micro Batch 并没有列出来，因此我们用红色 10-19 补充了部分 Micro Batch）。这种 Overlap 还确保了随着模型进一步扩展，只要保持恒定的计算与通信比，仍可在跨节点部署细粒度专家的同时，实现近乎零的 All2All 通信开销。

PS：正常来说是无法实现双向 PP 调度的，主要是因为 Forward 执行顺序是从前往后，比如从 Layer 0,1,2,...,14,15，而 Backward 执行顺序是从后往前，比如 Layer 15,14,...,2,1,0。而常见 PP 中的 Layer 只会在某一个 PP Stage，比如 8 PP，那么：

- Stage 0 上有 Layer 0 和 1 的权重
- Stage 1 上有 Layer 2 和 3 权重
- Stage 7 上有 Layer 14 和 15 的权重
- Forward 的顺序也只能从 Stage 0 到 Stage 7，不能从 Stage 7 到 Stage 0。

而 DeepSeek V3 的双向 PP 调度中，还是 8 PP 为例：

- Stage 0 上有 Layer 0, 1 以及 Layer 14, 15 的权重
- Stage 1 上有 Layer 2, 3 以及 Layer 12, 13 的权重
- Stage 7 上有 Layer 14, 15 以及 Layer 0, 1 的权重
- 相当于有 2 份相同的模型副本，Forward 的顺序可以从 Stage 0 到 7，也可以从 Stage 7 到 0。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64Re4ibliaxxmXJhxeQITtEwtmfR9OprU4gYLTKUCawiatFDBgIia6UwYBrA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=32)

#### 3.3.7 NVIDIA Merged FWD-BWD

DeepSeek 的 DualPipe 会导致静态显存翻倍，此外仍然存在较高的 Bubble 率。针对上述问题，NVIDIA 也提出利用奇&偶 Micro-Batch 实现 Overlap 的方案，已在 Megatron-LM 中集成，如下图所示：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64iagWGibXNhibVkcAhywJIzrbspiaAjy7aar0XlhiahzaGn4sUnic74a43LHA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=33)

此外，华为在 [2505.04519] Pangu Ultra MoE: How to Train Your Big MoE on Ascend NPUs 中提出了和 NVIDIA 类似的方案，核心思路也是利用 Micro Batch 间的独立性，以 Forward 计算掩盖 Backward 通信（反之亦然）。

### 3.4 EP

#### 3.4.1 概述

对于 Dense LLM 而言，采用 DP（Zero-1） + TP + PP 基本都能获得不错的吞吐。而对于 MoE 模型，尤其是细粒度 MoE，TP 已经不再合适，一般都会引入 EP 策略。这里通常会涉及几个方面的问题：

- 在 MoE 之前引入 All2All Dispatch，在 MoE 之后引入 All2All Combine 操作，通信开销比较大。
- Dispatch 和 Combine 处存在比较多小的 Kernel，效率比较低。
- EP 中不同设备可能存在负载不均的问题。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64agcPaqQdN2DN7icclAJ6qLv3MnSibPgYpibiarQTRaJx9CWgISho0jfWKw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=34)

#### 3.4.2 DeepEP

MoE 模型 EP 的主要挑战就是高性能的 All2All 实现。DeepSeek V3 中，为了确保 DualPipe 具有足够的计算性能，定制了高效的跨节点 All2All 通信 Kernel（包括 Dispatching 和 Combining），以节省专用于通信的 SM 数量。Kernel 的实现也与 MoE Gating 算法及集群的网络拓扑共同设计。

- 为了有效利用 IB 和 NVLink 的不同带宽，将每个 Token 限制为最多被发送到 4 个节点，从而减少 IB 流量。
- 对于每个 Token，在做出路由决策时，首先通过 IB 传输到其目标节点上具有相同节点内索引的 GPU。一旦到达目标节点，将努力确保它通过 NVLink 立即转发到承载目标专家的特定 GPU，而不会被随后到达的 Token 阻塞。（PS：比如说，节点 A 上 GPU 0 的 Token 要发送到节点 B 上的 GPU 3，则对应的路径为：节点 A GPU 0 -> 节点 B GPU 0 -> 节点 B GPU 3。这样做是因为高性能 GPU 训练集群往往会采用轨道优化，同号 GPU 在一个 Leaf Switch 下，因此可以利用高速的 NVLink 来代替从 Leaf Switch 到 Spine Switch 的流量，从而降低 IB 通信时延，并且减少 Leaf Switch 和 Spine Switch 之间的流量，这也是 NVIDIA PXN 的核心思路）

DeepSeek 随后也开源了相应实现 DeepEP（GitHub - deepseek-ai/DeepEP: DeepEP: an efficient expert-parallel communication library），是专为 MoE 和 EP（Expert Parallelism, EP）设计的通信库。提供了一系列优化的通信 Kernel，实现了以下能力：

- 高度优化的 All2All 通信，适合 MoE 模型 2 个主要过程：
- Dispatch：将 Token 发送给专家。
- Combine：从专家接收处理过的 Token 过程。
- 同时支持不同的通信类型：
- 节点内（intra-node）：可以使用 NVLink + NVSwitch 通信。
- 节点间（inter-node）：可以使用 RDMA 通信。
- 针对不同场景的 Kernel：
- 常规（高吞吐） Kernel（Normal Kernel）：针对 Training 和 Inference Prefill。节点内 NVLink + 节点间 RDMA 通信。
- 低时延 Kernel（Low-Latency Kernel）：针对 Inference Decoding。使用纯 RDMA 通信来最小化时延。
- 原生支持 FP8，减少数据传输需求，相比 FP16 通信量减半。
- 灵活的 GPU 资源（SM）控制，支持计算和通信的 Overlap。

除此之外，如果模型不是非常大，将 EP 的 All2All 放在一个节点内，使用 NVLink + NVSwitch 的高带宽通信也是常见的优化方案。

#### 3.4.3 专家负载均衡

专家负载均衡是 MoE 模型另外一个非常常见的问题。其不仅影响预训练阶段，还会影响 Inference 阶段的性能。针对这个问题，通常会采用负载均衡损失来解决，由于 DeepSeek 系列模型的负载均衡比较典型，这里以 DeepSeek 系列为主。

DeepSeek MoE 中采用了 2 种负载均衡损失：

- 专家级负载损失（Expert-Level Balance Loss）：让各个专家的负载平均，尽量避免都路由到少数专家。
- 设备级负载损失（Device-Level Balance Loss）：强制让各个专家负载平均可能影响效果。由于每个设备（GPU）包含多个专家，因此让不同设备上的计算量尽量均衡同样可以一定程度避免负载不均的问题。

DeepSeek V2 中采用了 3 种辅助负载均衡损失：

- 专家级负载损失（Expert-Level Balance Loss）：同 DeepSeek MoE。
- 设备级负载损失（Device-Level Balance Loss）：同 DeepSeek MoE。
- 通信负载损失（Communication Balance Loss）：设备限制路由可以保证每个设备发送的通信量是有界的，但如果某个设备接收的 Token 比其他设备多，实际的通信效率也会受到影响。为此，引入通信负载损失，正是为了缓解不同设备接收 Token 不均衡的问题。保证设备之间均衡交换信息，促进高效通信。

DeepSeek V2Token Dropping 策略：负载均衡损失可以促进均衡，但是无法严格保证。为了进一步减少负载不均导致的计算资源浪费，DeepSeek V2 中额外引入了设备级 Token 丢弃策略（Device Level Token Dropping Strategy）。首先计算每个设备的平均计算预算，也就意味着每个设备的容量因子等同于 1.0，然后在每个设备上丢弃亲和力得分最低的 Token，直到达到计算预算需求；除此之外，确保约 10% 的训练序列中的 Token 永远不会被丢弃。根据效率需求，可以在 Inference 过程中灵活配置是否丢弃 Token，并始终保证 Training 和 Inference 的一致性。

DeepSeek V3 中进一步对专家负载均衡策略进行了调整：

- 无需辅助损失的负载均衡策略（Auxiliary-Loss-Free Load Balancing Strategy）：来自 DeepSeek 2024 年的论文（[2408.15664] Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts），具体来说，其通过动态更新每个专家的偏置（b）来维持专家的负载均衡，而不会引入额外的干扰梯度。
- 补充的序列级辅助损失（Complementary Sequence-Wise Auxiliary Loss）：尽管 DeepSeek-V3 主要依赖于无辅助损失的策略来实现负载平衡，但为了防止在任何单一序列中出现极端的不平衡，作者采用一种补充的序列级均衡损失。这种序列级均衡损失的目的是鼓励每个序列中的专家负载更加均衡，避免负载集中在少数专家上，从而提高模型的效率和公平性。
- 节点限制路由（Node-Limited Routing）：与 DeepSeek-V2 所采用的设备限制路由类似，DeepSeek-V3 同样使用了一种约束路由机制以控制训练过程中的通信开销。简而言之，确保每个 Token 最多被发送至 M 个节点，这些节点的选择依据是分布在各节点上的专家中，其亲和度得分最高的 Kr/M 项之和。在此约束下，MoE 训练框架几乎能够实现计算与通信的完全重叠。

DeepSeek V3No Token-Dropping：得益于高效的负载均衡策略，DeepSeek-V3 在整个训练过程中保持了良好的负载平衡。因此，DeepSeek-V3 在训练期间未丢弃任何 Token。此外，作者还实施了特定的部署策略以确保推理过程中的负载均衡，故 DeepSeek-V3 在推理阶段同样不会丢弃 Token。

#### 3.4.4 MoE 计算优化 —— 设备内

当前常见粗粒度 MoE 模型的专家数通常是 8-32，而细粒度 MoE 的专家数通常可以达到 64-256，在 Kimi K2 中更是高达 384。在 Inference 阶段经常会采用超大 EP 的方式以尽可能提升计算强度；而在 Training 阶段，不像 Inference 的 Decoding 有那么明显的 Memory Bound 问题，因此通常 EP 不会特别大。此时，一个设备上会存在多个专家，相应的计算问题就是经典的 Grouped GEMM 计算问题，由于每个专家的 Token 数可能不同，相应矩阵计算的 Shape 也可能不等。

Grouped GEMM 有几种常见的解法，在不同 Shape 下它们的性能也会差距很大（如下图 Figure 1 是非常早的一个图，一定程度上可以说明这个问题）：

- For 循环串行调度：效率最低，但是当 Shape 非常大时可能也不会特别差。
- Multi-Stream 调度：多个 Stream 同时执行，在 Shape 不是特别大时可以更充分的利用 GPU 资源，但是 Kernel Launch 开销无法避免。
- Batched GEMM：对于同 Shape 的多个 GEMM 计算，cuBLAS 提供了 BatchedGEMM 的 API（cublasgemmgroupedbatchedex），可以直接使用，通常能获得很不错的性能。
- Grouped GEMM：对于不同 Shape 的多个 GEMM 计算，CUTLASS 和 DeepSeek 开源的 DeepEP（DeepGEMM: clean and efficient FP8 GEMM kernels with fine-grained scaling）也提供了相应方案。这种方式通常能获得很不错的性能。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64ZtJicdwbVMrXNtl9amfHrjYRgERN8ich0n7PABdpficWejYwSLO5vlrWA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=35)

如下图 Table 1 所示，小红书团队在 [2506.05767] dots.llm1 Technical Report 中也提到了相关优化方案，通过优化 GroupedGEMM 获得相应算子 Forward 14%，Backward 8% 的提升：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64mSgvN4067lgHxhB6epQR8VlxcUrmgDfBrj5OGWYhT4nRyU68H2tVWA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=36)

#### 3.4.5 MoE 负载优化 —— 设备间

MoE 负载均衡损失可以尽可能的降低负载不均的问题，但是依然无法严格保证。此外，Grouped GEMM 可以优化设别内的计算效率，但是无法缓解节点间负载不均的问题。

针对上述问题，华为在 Pangu Ultra MoE（[2505.04519] Pangu Ultra MoE: How to Train Your Big MoE on Ascend NPUs）中提出了动态重排的方案。如下图 Figure 11 所示，如果某个 Device 上被路由到的 Token 数都比较少（Device 0），则相较于路由到比较多 Token 的 Device 会出现计算的 Bubble（Device 1）。通过 Planner 和 Executor 的协同合作，动态调整 Device 上 Expert 的排布，可以让不同 Device 上的负载尽可能均衡，从而提升整体的利用率。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64rts1fNmFEGSwq0iaSko83xvia9uuVhTEZxWSl0WYwxXub3hUm80DfOaw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=37)

华为在 Pangu Pro MoE（[2505.21411] Pangu Pro MoE: Mixture of Grouped Experts for Efficient Sparsity）中还采用了专家分组的方式，在专家选择时，让每个组都选择固定数量的专家，这样可以保证每组的负载是均衡的，但会降低专家的可组合数，也就是降低专家组合的空间。（DeepSeek 中也有类似的方案）

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64wtu4sicUKKibq8ny4tibhFFJrB3poyLdkNOVjnnicS8AMSes2hp6JHgfzw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=38)

其实 DeepSeek V3 在 Inference 阶段也会采用类似但稍有不同的负载均衡方案，比如针对共享专家、高负载专家采用冗余部署，并动态重排的方案来尽可能的实现负载的均衡。

#### 3.4.5 Permute & UnPermute 算子优化

在 MoE 模型中另外一个容易影响吞吐的地方就是 MoE 相关的 Permute 和 Unpermute 操作，此处会存在很多较小的 Kernel，如果不进行优化也会一定程度上影响性能。NVIDIA 在 GitHub - NVIDIA/TransformerEngine 中也提供了相应 Kernel 融合的优化，具体可以搜索 moe_permute 和 moe_unpermute 相关实现。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64EESwAzyXyF0B0a0iaXIQM9MLnDS0vnHqGf6GYEwXkZfib4dKiao88iaQqQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=39)

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64FbWDmEbSRNapNgiaibhtPLxDDvZE7riadvsIute2lo1hlicSDKCm5OjWjA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=40)

### 3.5 CP & SP

在 Transformer 模型中，自注意力机制的内存需求、计算量与序列长度成二次方关系，导致序列比较长时可能存在明显瓶颈。为此，也有一系列工作尝试解决这个问题，比如：

- [2105.13120] Sequence Parallelism: Long Sequence Training from System Perspective：核心是将输入序列分为不同的 Chunk，每个 Chunk 都使用独立的设备处理，而 Attention 部分采用 Ring Attention 的实现。
- [2205.05198] Reducing Activation Recomputation in Large Transformer Models：NVIDIA 在 Megatron 中也提供了相应实现，主要思路是在 TP 的基础上，将原来未切分的部分激活进一步切分，从而降低内存开销。
- [2309.14509] DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models：微软进一步解决了序列并行中通信量比较大的问题，同样是按序列切分，不过在 Attention 计算时通过 All2All 将同一个 Head 的序列汇聚在一起。这里也有个约束条件，Head 数需要是序列并行度的整数倍。
- [2310.03294] DISTFLASHATTN: Distributed Memory-efficient Attention for Long-context LLMs Training：进一步解决了长序列时 Ring-Attention 切分方式导致的负载不均衡问题。整体思路还是对调度进行重排，减少 Bubble。
- [2405.07719] USP: A Unified Sequence Parallelism Approach for Long Context Generative AI：结合了上述 Ulysses 和 Ring-Attention 的优势。

在 LLM 的预训练场景中通常序列长度不会超过 8K，因此没有太大必要使用序列并行，在序列长度扩展或视频场景中比较需要，这里不再展开。

## 四、常见优化方案

### 4.1 FP8 训练

#### 4.1.1 概述

NVIDIA GPU 从 Hopper/Ada Lovelace 开始支持 FP8 计算，其 FP8 算力通常是 FP16 算力的两倍，而常见的训练还是以 BF16 为主，利用 FP8 实现训练加速也是一个值得探索的方向。此外，NVIDIA 从 Blackwell GPU 开始也进一步支持了 MX Format，可以原生支持细粒度的 FP8 甚至 FP4 量化，为 FP8 训练，FP4 推理提供了更多可能。

微软在 23 年就提出了使用 FP8 加速训练的工作（[2310.18313] FP8-LM: Training FP8 Large Language Models）。NVIDIA 也在 Transformer Engine（Using FP8 with Transformer Engine）和 Megatron-LM 中提供了相应支持。

然而，LLM 预训练的代价很高，对 FP8 训练是否能真的获得和 BF16 相当的性能也不得而知，因此，早期业内一直没有真正的使用 FP8 做预训练。

为了解决上述问题，零一万物在 零一万物面向万卡集群的 AI Infra 建设 中提到了一个 Trick 的方法。如下图所示，每隔一段时间就会 Load FP8 的 Checkpoint 并使用 BF16 进行训练，验证 Loss 是否和 FP8 训练的 Loss 一致（PS：Loss 对齐就真的表示下游任务也能对齐吗？）。如果出现不一致的情况，就会使用 BF16 的训练代替 FP8，并在一段时间后继续使用 FP8 训练。最终获得了 1.3x 的吞吐提升，不过并没有说明这个提升是纯粹的 FP8 相比 BF16 还是也包含了 BF16 的校验预算。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64k0b2mVicD1j7SWVL2ZXlAHVSFNtlpZ6PMjXycdV3QYH9rgibFOZYfeSQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=41)

#### **#### 4.1.2 DeepSeek FP8**

DeepSeek V3 中另一个比较大的优化是 FP8 混合精度训练，也是业内首个宣称使用 FP8 进行端到端预训练并进行了大量优化的工作。在 DeepSeek V3 的训练中，大多数计算密集型操作以 FP8 执行，而少数关键操作则保留原始数据格式，以平衡训练效率与数值稳定性。整体框架如下图 Figure 6 所示，与线性算子相关的三个 GEMM 操作，包括 Forward（Fprop）、激活 Backward（Dgrad）和权重 Backward（Wgrad），接受 FP8 Tensor 作为输入，并输出 BF16 或 FP32 格式的结果，理论上使计算速度较原 BF16 方法提升一倍。此外，FP8 Wgrad GEMM 允许激活值以 FP8 存储，供 Backward 使用，从而显著降低内存消耗。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64MNktErO1MpaJUP0uAf21qb3YGFBDR7jLibav28sJjoNLsktb0RwCQrA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=42)

尽管 FP8 格式具有效率优势，但某些算子对低精度计算比较敏感，仍需更高精度。同时，一些低成本算子也可采用更高精度，对整体训练性能的影响较小。因此，对以下组件保持原始精度（如 BF16 或 FP32）：Embedding Module、输出 Head、MoE 门控模块、归一化算子及 Attention 算子。为进一步保证数值稳定性，将主权重、权重梯度和优化器状态以更高精度存储。

DeepSeek V3 中还引入了一系列策略来提升 FP8 训练的准确率，包括：

- 细粒度量化。
- 提升累加精度。
- 全部使用 E4M3 以获得更高精度。
- 在线量化。

除此之外，DeepSeek V3 还通过将缓存的激活值和优化器状态压缩为低精度格式，进一步减少内存消耗和通信开销。

### 4.2 CUDA Graph

CUDA Graph 首次出现在 CUDA 10 中，是 NVIDIA 在 CUDA 编程模型中引入的一种工作提交模型。允许将一系列 GPU 操作（如 Kernel、内存拷贝、Event 记录等）按依赖关系组织成一个 Graph，并与其执行分离开来。换言之，Graph 描述了整个任务流程的静态依赖关系，一旦定义完成，就可以多次重复执行，而无需每次都重新 Launch Kernel 和设置依赖。

在传统的执行模型中，每次 Kernel Launch 都需要 CPU 配合执行一系列准备和调度操作，这对每个 Kernel 都是额外开销。当 Kernel 执行时间很短时，这些 Launch 开销可能成为整个任务的主要瓶颈（PS：这也是 Kernel Fusion 的一个好处）。CUDA Graph 通过提前定义工作流、预先实例化 Graph，将这些开销挪至 Graph 的准备阶段，大幅降低了每次执行时的 CPU 负担。一旦定义完成，就可以多次重复执行，而无需每次都重新 Launch Kernel 和设置依赖。

如下图所示，Launch Graph 的时间远小于 A、B、C、D、E 这 5 个 Kernel 总的 Launch 时间。也就是在多个小 Kernel 按顺序执行的场景中，用单次 Graph Launch 替代多次小 Kernel Launch，可以显著减少 GPU 闲置等待时间，提高整体吞吐率。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64LVaP4rDJFHa1dXbaWoWMrKia50lr0iaeFuiaib3ZZlgZUJkvt8DmhTKwMA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=43)

实践中，当 Kernel 执行时间较短（微秒级）时，Graph 可显著减少调度开销并提升性能。此外，CUDA Graph 将完整的计算流程呈现给驱动程序，使得驱动能够针对整个流程进行优化（如更高效的线程块调度），这是逐次提交无法轻易做到的。

PyTorch 和 Megatron-LM 都提供了对 CUDA Graph 的支持，在一些小规模细粒度的 MoE 模型训练中会有比较明显的帮助。

### 4.3 细粒度 Overlap

DeepSeek V3 中的通信、计算细粒度 Overlap 也是其软硬协同设计的一个重要组成部分。然而 DeepSeek 方案是针对特定硬件环境、特定模型的高度定制化，在其他场景并不一定最优；此外，随着模型、硬件的变化，分别定制化的成本比较高，因此也就衍生出一系列更通用的方案。

其实在 DeepSeek V3 之前已经有一些相关工作，但是没有被广泛使用，在 DeepSeek V3 之后，字节跳动也出现了一些相关方案，值得关注。

24 年北大提出过 Centauri 框架（[ASPLOS 24.04] Centauri: Enabling Efficient Scheduling for Communication-Computation Overlap in Large Model Training via Communication Partitioning），其构建了一个由三个固有抽象维度组成的切分空间：原语替换、拓扑感知组切分及工作负载切分。这些维度共同构成了一个全面的优化空间，用于高效 Overlap。为确定通信与计算的高效 Overlap，将混合并行训练中的调度任务分解为 OP、Layer 和模型三个层次。如下图 Figure 3 所示，通过通信切分和层次调度实现通信和计算的细粒度 Overlap。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64vNZnzPQZiboYWibt65UDxUk36ibPFroab2tnyqbiaoAQ88SIic3d9rW4nvQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=44)

字节也相应提出了 Flux（[2406.06858] FLUX: Fast Software-based Communication Overlap On GPUs Through Kernel Fusion），旨在通过依赖计算隐藏 GPU 间的通信时延。Flux 将通信和计算操作分解为更细粒度的操作，并进一步融合成更大的 Kernel，从而在不损害 Kernel 效率的前提下有效隐藏通信。

25 年，字节又在 Flux 的基础上提出 TileLink（[2503.20313] TileLink: Generating Efficient Compute-Communication Overlapping Kernels using Tile-Centric Primitives），旨在高效编译并生成计算-通信 Overlap 执行的 Kernel，并且聚焦于层内并行。TileLink 由前端（Frontend）和后端（Backend）构成：

- 在前端，系统通过以 Tile 为中心的原语将通信与计算的设计空间解耦并建立关联。
- 在后端，将这些原语转换为底层指令，整合通信与计算组件以实现 Overlap 执行。

在 TileLink 之后，字节很快又发布了 Triton-distributed（Programming Overlapping Kernels on Distributed AI Systems with the Triton Compiler），其作为对 Triton 编译器的扩展方案，以更好的支持分布式 AI 工作负载的 Overlap 优化。其首先将符合 OpenSHMEM（NVSHMEM）标准的通信原语集成到编译器中，使得能够通过更高层次的 Python 编程模型使用这些原语。其次，还阐述了如何借助编译器实现计算、内存访问与通信的复杂联合优化，并重点介绍了如何利用 Overlap 技术隐藏时延。

### 4.4 Attention Mask

在 Attention 模块中的 Attention Mask 是实现序列中 Token 之间信息融合的关键模块，也是实现各种业务场景的关键所在。

- 在 LLM 预训练中，由于 Attention 计算占比不高，并且基本都是 Causal Mask，因此这块相应的关注比较少，基本上用 FlashAttention 就能获得非常高的性能。
- 在 LLM 后训练或者多模态等复杂场景，当序列比较长时，Attention 的占比变大，Attention 相应的优化问题也需要关注。

如下图所示就是一系列常见的 Attention Mask，比如：

- （1）Causal：Decoder 的标准 Mask。
- （2）Sliding Window：滑动窗口 Attention 常用的 Mask。
- （3）Causal Document Mask：SFT 等后训练常用的 Sample Packing Mask。
- （9）Prefix LM Cache Mask：一些多模态场景会用的 Mask。
- （12）Random Eviction Mask：有点类似 Tree Attention Mask，在投机采样，推荐等场景比较常见。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64Uxcgn5JrP8SzUErIHPxYlfzyD8qE0xANuicvV10XoibyHMrBicuzOTc6Q/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=45)

Attention 部分最经典的优化就是 FlashAttention 系列工作，包括 V1、V2、V3，随后也有其他相关优化，比如：

- PyTorch 官方的 FlexAttention 优化：FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention。
- 百度的 FlashMask 优化：[2410.01359] FlashMask: Efficient and Rich Mask Extension of FlashAttention
- FlashInfer 的 Sparse Attention：flashinfer.sparse

## 五、业内常见模型

### 5.1 Meta LLaMA

Meta LLaMA 3.1 405B（[2407.21783] The Llama 3 Herd of Models）是开源的最大的 Dense 模型：

- 总参数量 405B。
- 采用 GQA，比例为 128/8。
- 共 126 层。
- 词表 128,000。
- 15T Token。

如下图 Figure 5 所示为 405B 模型对应的分布式排布，其 MFU 在 38%-43% 之间（未使用 FP8）：

- TP 始终是 8，保持在一个节点内，充分利用 NVLink + NVSwitch。
- 在 8K 序列长度时不用 CP，在 128K 时使用 16 CP。
- 始终使用 16 PP。
- 根据预算 GPU 数调整相应的 DP 数和 Batch Size。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64xqINGmLapzfLyeic8kGwLAic2o48HM576bJphVcZCYpZAEucibicePpvOw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=46)

主要提到的优化就是对 VPP 的一些改进，以便解决内存和计算的不均衡，比如首尾 Stage 都少一层，主动释放一些不需要的 Tensor 等：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64LjtWwFncmbPSibibxgR1pp4VZcD3OLgjfb9IjdnY49Kiben7dWICice3lw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=47)

### 5.2 DeepSeek

如下图所示，今年上半年我们曾总结过 DeepSeek 相关工作中的关键技术点，这里不再赘述，详细内容可以参考： [综述：DeepSeek Infra/V1/MoE/V2/V3/R1 & 开源关键技术](https://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247489358&idx=1&sn=bda66cd5ffc40d6dd653d0bc076c7c09&scene=21#wechat_redirect)。其中 DeepSeek V3 预训练 MFU 预估在 38.2% 左右。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64YV1xzoibmWQjT5tVVZwRrHe2xFJSEpJbN5JfriaVV5CrIzdeic5IU9ibiag/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=48)

### 5.3 阿里 Qwen3 MoE

阿里之前也开源了其最新的 Qwen3 系列模型（[2505.09388] Qwen3 Technical Report），其中最大的模型为 Qwen3 W235A22 模型，具体配置如下：

- 总参数量 235B，激活参数 22B。
- 采用 GQA，比例为 64/4。
- 128 个路由专家，每个 Token 激活 8 个专家。
- 共 94 层。
- 词表 151,669。
- 预训练序列长度 4K。
- 预训练 36T Token，119 种语言和方言。

Qwen3 虽然提供了技术报告，但并没有提供使用的 GPU 资源量、MFU 以及相关的训练优化手段。

### 5.4 腾讯 Hunyuan-TurboS

我们前面提到过，腾讯的 Hunyuan-TurboS W560A56 模型（[2505.15431] Hunyuan-TurboS: Advancing Large Language Models through Mamba-Transformer Synergy and Adaptive Chain-of-Thought）采用了 Softmax Attention 与 Mamba2 混合的架构：

- 总参数量 560B，激活参数 56B。
- 采用 GQA，比例为 64/8。
- 预训练 16T Token。
- 1 个共享专家，32 个路由专家，每个 Token 激活 2 个路由专家。

不过并没有提供预训练 Infra 优化相关部分，没有介绍使用的资源量、MFU 及相应的优化方案。

### 5.5 小红书 dots.llm1

#### 5.5.1 模型配置

小红书于 2025.06.06 发布并开源自研大模型 dots.llm1 W142A14，并提供相应技术报告（[2506.05767] dots.llm1 Technical Report）。对应的模型配置为：

- 总参数量 142B，激活参数 14B。
- 采用 MHA，32 Head。（PS：在大模型里依然采用 MHA 是比较少见的）
- 第 1 层是 Dense 层。
- 2 个共享专家，128 个路由专家，每个 Token 激活 6 个共享专家。
- 共 62 层。
- 预训练 11.2T Token。
- 预训练序列长度 8K。

#### 5.5.2 预训练优化

论文中针对预训练主要提了两个优化手段，在前面已经介绍过：

- 和 NVIDIA 合作的类似 DualPipe 的 Interleaved-1F1B 优化。
- Grouped GEMM 优化。

根据公式可以大概推出 DeepSeek V2 预训练的 MFU：

MFU = (Token 数 * Ctoken) / (训练 GPU 小时数 * GPU FLOPs * 3600)

如下图 Table 4 所示 Qwen2.5 72B MFU 大约为 35.69%，dots.llm1 W142A14 MFU 大约为 18.15%：

Qwen2.5 72B：(1T*72B*6) / (340K*989T*3600) = 35.69%

dots.llm1 W142A14：(1T*14B*6) / (130K*989T*3600) = 18.15%

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64ueWiaOSVZMYOvGdLXgiaZEOpeiaAWj3OPH7IehXnuJULdzogoQJFQ7F5Q/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=49)

### 5.6 百度 ERNIE 4.5

#### 5.6.1 模型配置

百度于 2025.06.28 也开源了自研的 ERNIE 4.5 系列模型，并提供了详细的技术报告（ERNIE 4.5 Technical Report）。其中最主要的 LLM 为 ERNIE-4.5-300B-A47B：

- 总参数量 300B，激活参数 47B。
- 采用 GQA，比例为 64/8。
- 前 3 层是 Dense 层。
- 64 个路由专家，每个 Token 激活 8 个专家。
- 共 54 层。
- 预训练序列长度 4K。

#### 5.6.2 预训练优化

W300A47 LLM 预训练的分布式策略为 21DP（ZeRO-1）、8EP（Attention 8TP）、12PP，共 2016 H800 GPU，Global Batch Size 15120，GPU 之间使用 RoCE NIC，MFU 达到 47%。此外，ERNIE 也采用了一系列的优化措施。

机内 EP：从上可以看出，其训练使用 8EP，可以将 EP 的 All2All 全部放在机内，进而降低跨机 All2All 的压力。

All2All 内存优化：如下图 Figure 9a，传统 MoE 实现在第二次 All2All 后应用 Gating 概率乘法算子。该方法需保留第二次 All2All 的输出 Tensor 以供 Backward，造成显著内存压力。如下图 Figure 9b，作者提出将 Gating 概率乘法算子重新放于专家计算模块内部。这一架构改进使得第二次 All2All 输出 Tensor 在使用后可立即释放。尽管通过概率置换和额外轻量级 All2All 操作引入了微小开销，但该优化显著降低了峰值内存使用量，并消除了 Backward 过程中的大量重计算。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64uxiczX5OLuBeUrBdYL6FD9wRh48SqzbXm3eHWZ1eicvVq1UmLy6tYHmQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=50)

VPP 优化：至于 PP 方案，当梯度累积比较少时，比如小于 PP-Degree 时，采用 1F1B；当梯度累积比较大时，采用 VPP（Interleaved 1F1B）。考虑到最后一个 PP Stage 还包含损失函数相关计算，也会占用较高内存，因此对其进行优化，一旦最后一个 PP Stage 的 Forward 完成，立即启动 Backward 计算并释放损失函数的激活内存，这样最后一个 PP Stage 最多只保留单个 VPP 阶段的激活内存。如下图 Figure 11 中的红框所示。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64tuLicTGztoKibdlIrKs51OPgTW0cAEx1qWeGdy4xsbZliaia5WTuY2sDicA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=51)

FP8 混合精度训练：采用和 DeepSeek V3 类似的量化策略，使用 E4M3 FP8 数据格式及在线量化方法，对权重实施 Block-wise 量化，对激活实施 Tile-wise 量化。如下图 Figure 12 所示为 ERNIE 4.5 的 FP8 混合精度训练策略。

- FP8 训练中的细粒度内存优化：FP8 训练可以降低内存开销，因此可以降低重计算以提升吞吐。MoE 中主要激活内存消耗来自 Up-gate Linear、Down-gate Linear、Down Linear、SwiGLU 以及 Gate 概率乘法的输入激活。
- 对于 Up-gate Linear，保留其 FP8 输入激活值 XFP8 而非 BF16 张量 XBF16 用于 Backward。在 Backward 中，需要利用转置后 XBF16 的 FP8 量化版本来计算权重梯度。因此，在权重梯度计算阶段，需要对 XFP8 执行反量化-转置-再量化操作。该策略可降低 Up-gate Linear 的内存占用，并使得第一个 All2All 操作使用 FP8 精度执行以节约通信开销。这是内存占用、通信成本与计算精度之间的权衡方案，实验表明该方法能保持与基线实现相同的收敛速率。
- 对于 Down Linear，有两种优化方案：（1）保留 Up-gate Linear 的 BF16 输出张量；（2）利用上述 XFP8 张量重新计算 Up-gate Linear 以生成其 BF16 输出张量。这两种方法均需要对 SwiGLU 激活函数和 Gate 概率乘法算子执行轻量级重计算，从而节约这两个算子输入张量的内存占用。
- FP8 算子融合优化：通过算子融合降低数据移动开销并提升计算强度，具体包括：(1) Forward 中 Permutation 操作与 FP8 量化的融合；(2) Forward 与 Backward 中 SwiGLU、Gate 概率乘法及 FP8 量化的三重融合。
- FP8 通信优化与 Overlap：Forward 阶段采用 FP8 精度执行第一个 All2All 以降低 BF16 通信成本；Backward 阶段将第二次 All2All 与 Up-gate Linear 权重梯度计算进行 Overlap。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd6410vjsOBvib3TuZhkeROCxib2nKnABSniaJEXpY3ljAJ89r042oHhibV3yQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=52)

**重计算优化：为了制定最优的重计算策略，对模型中的每个算子进行精细化分析，系统评估了内存占用与计算时间的权衡。选择性的对性价比最高的算子（能以最小运行时代价换取显著内存节省的算子）实施算子级重计算，最终设计出能最大化训练效率的最优重计算方案。**

Paddle 框架原生容错系统：为了实现快速识别硬件故障等异常，并做到快速恢复，设计了框架原生的容错系统。如下图 Figure 13 所示，包括以下几个关键组件：

- TraceHang：通过并行度信息和通信记录的协同分析，实现对 Hang 源头的自动化诊断。TraceHang 可以精准定位 Hang 的根本原因，从而加速问题解决并最大限度减少停机时间。
- Online SDC Scanner：静默数据损坏（Silent Data Corruption，SDC，这种情况 ECC、CRC 可能捕获不到）因其隐蔽性特征对模型收敛构成重大威胁。利用 PP 中的 Bubble，采用固定输入参数执行计算与通信操作，并将结果与基准真值进行实时比对验证，识别出多个存在 SDC 的节点单元。
- Parallelized Warmup：PP Warmup 阶段存在数据依赖性，因而初始化导致的性能退化被放大 P 倍（PP Stage 数量）。因此，采用了跨 PP Stage 的并行同步 Warmup 方案，将首个训练 Step 的延迟降低到 1/P。
- Zero Cost Checkpoint (ZCC)：支持在每个训练步骤保存检查点，且不会对训练吞吐量产生任何开销，从而确保训练中断时不会丢失任何进度。首先是训练中异步保存，其次是将异常节点 Checkpoint 通过 RDMA P2P 操作快速传输到健康节点，并与初始化 Overlap；如果异常节点内存不可访问，则从持久化 Checkpoint 恢复。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64F7uFUgibUPtVxR7jibYpm1jHact5IC9b9Y9lbgjfo35Vn2wlib1PnuU4g/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=53)

### **### 5.7 Kimi K2**

#### 5.7.1 模型配置

Kimi K2 是 Moonshot AI 在 2025.07.11 新发布的模型（Kimi K2: Open Agentic Intelligence），其采用了和 DeepSeek V3 类似的模型，也获得了很不错的效果。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64E7icgURLk82OIkm9SK4ria81Ed8gzOwvbGQltlKLFfFoultTuvcswSmg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=54)

**如下图（https://x.com/rasbt/status/1944056316424577525/photo/1）所示为 Kimi K2 与 DeepSeek V3 模型的主要区别：**![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64B9XAC9BlF2hFSTYeo1icujqav9gjRSmX6ic2h7COd74qMcicwTW8JQCLQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=55)

除此之外还有些需要关注的地方：

- Kimi K2 采用了自研的 MuonClip Optimizer。
- DeepSeek V3 增加了 Expert 分组以便降低通信开销，同时强制负载均衡；Kimi K2 中去除了这一个限制，以便增加更多的专家组合空间。
- Kimi K2 参数量更大，但是激活更少，主要是因为 MoE 部分共享专家增多，但是激活专家并没有增加；同时 Attention Head 减半，因此总的激活参数反而降低。

2025.07.22 Moonshot 发布了 Kimi K2 的技术报告（https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf），模型配置与上述一致：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64IEMA4iaJ8NnQ2E5kXViceJtdzl90JfcK0SRGgWuslo7V9aBibvU47LjiaA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=56)

#### 5.7.2 预训练优化

Kimi K2 在 H800 GPU 集群训练，每个节点包含 8 个 H800 GPU，2T 内存，采用 NVLink + NVSwitch 高速互联，并且包含 8 个 400 Gbps 的 RoCE 网卡。不过没有提供 MFU 相关信息。

采用动态资源训练，并保证资源数量是 32 个节点（256 H800）的整数倍，其 ZeRO-1 DP + 16PP + 16EP 的策略，训练时直接扩展 DP 组即可。训练时以 BF16 格式存储模型参数，并采用 FP32 梯度累加，约需要 6TB 显存，分散在 256 个 GPU 的模型并行组中。而对于优化器状态，节点较多时分散在所有节点，节点较少时，则 Offload 到 CPU 内存。

通信和计算 Overlap：通过增加 Warmup Micro Batch 数量，实现 EP 中 All2All 与 Interleaved 1F1B 的计算 Overlap。不过 Interleaved 1F1B 把 PP 切的更碎，会引入更多的通信开销，为了降低这一成本，同样解耦了权重梯度重计算，使其能与 PP 并行通信 Overlap。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgSA05QPEibIUwicGbY4uMd64vHr2TI02XIlvZ0jlYKlFhoCCFVDJEiaoVqLXqNwFqbjkKwstMqNXfJg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=57)

**最小化 EP 并行：由于 Attention Head 减半，Attention 计算时间变少，为了更好的实现计算和通信的 Overlap，需要最小化 EP 耗时，因此采用了 16EP 的最小化 EP 并行策略，这也放宽了负载均衡约束。**

细粒度重计算：对 LayerNorm、SwiGLU 和 MLA Up-proj、MoE Down-proj，已最小化计算开销并最大化显存节约。

FP8 存储激活：对于 MoE Up-proj 和 SwiGLU 等不敏感激活采用 FP8-E4M3 存储（ 1 x 128 Tile 的 FP32 Scale）。潜在性能下降风险，未采用 FP8 计算。

激活 Offload：如上图 Figure 7 所示，采用了激活 Offload 的机制。

## 六、参考链接

1. MQA：[2003.04641] MQA: Answering the Question via Robotic Manipulation
2. GQA：[2305.13245] GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
3. FlashMLA：FlashMLA: Efficient MLA decoding kernels
4. MHA to GQA：[2412.20677] Align Attention Heads Before Merging Them: An Effective Way for Converting MHA to GQA
5. GQA to MLA：[2502.14837] Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs
6. TransMLA：[2502.07864] TransMLA: Multi-Head Latent Attention Is All You Need
7. RWKV-7：[2503.14456] RWKV-7 "Goose" with Expressive Dynamic State Evolution
8. MiniMax-01：[2501.08313] MiniMax-01: Scaling Foundation Models with Lightning Attention
9. Hunyuan-TurboS：[2505.15431] Hunyuan-TurboS: Advancing Large Language Models through Mamba-Transformer Synergy and Adaptive Chain-of-Thought
10. Mamba-2-Hybrid 8B：[2406.07887] An Empirical Study of Mamba-based Language Models
11. Mistral 7B：[2310.06825] Mistral 7B
12. MOBA：[2502.13189] MoBA: Mixture of Block Attention for Long-Context LLMs
13. NSA：Hardware-Aligned and Natively Trainable Sparse Attention
14. Mixtral 8x7B：[2401.04088] Mixtral of Experts
15. Skywork-MoE：[2406.06563] Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models
16. Switch Transformer：[2101.03961] Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
17. GlaM：[2112.06905] GLaM: Efficient Scaling of Language Models with Mixture-of-Experts
18. ST-MoE：[2202.08906] ST-MoE: Designing Stable and Transferable Sparse Expert Models
19. DeepSeek MoE：[2401.06066] DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models
20. DeepSeek V2：[2405.04434] DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
21. DeepSeek V3：[2412.19437] DeepSeek-V3 Technical Report
22. ZeRO：[1910.02054] ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
23. FSDP1：[2304.11277] PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel
24. FSDP2：[2411.00284] SimpleFSDP: Simpler Fully Sharded Data Parallel with torch.compile
25. Megatron-LM：[1909.08053] Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism
26. PipeDream：[1806.03377] PipeDream: Fast and Efficient Pipeline Parallel DNN Training
27. Interleaved 1F1B：[2104.04473] Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM
28. ZeRO Bubble：[2401.10241] Zero Bubble Pipeline Parallelism
29. GLM 130B：[2210.02414] GLM-130B: An Open Bilingual Pre-trained Model
30. LLaMA 3：[2407.21783] The Llama 3 Herd of Models
31. Pangu Ultra MoE：[2505.04519] Pangu Ultra MoE: How to Train Your Big MoE on Ascend NPUs
32. DeepEP：GitHub - deepseek-ai/DeepEP: DeepEP: an efficient expert-parallel communication library
33. DeepGEMM：DeepGEMM: clean and efficient FP8 GEMM kernels with fine-grained scaling
34. Pangu Pro MoE：[2505.21411] Pangu Pro MoE: Mixture of Grouped Experts for Efficient Sparsity
35. 小红书 dots.llm1：[2506.05767] dots.llm1 Technical Report
36. DeepSpeed Ulysses：[2309.14509] DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models
37. DistAttention：[2310.03294] DISTFLASHATTN: Distributed Memory-efficient Attention for Long-context LLMs Training
38. USP：[2405.07719] USP: A Unified Sequence Parallelism Approach for Long Context Generative AI
39. FP8-LM：[2310.18313] FP8-LM: Training FP8 Large Language Models
40. Centauri：Centauri: Enabling Efficient Scheduling for Communication-Computation Overlap in Large Model Training via Communication Partitioning
41. FLUX：[2406.06858] FLUX: Fast Software-based Communication Overlap On GPUs Through Kernel Fusion
42. TileLink：[2503.20313] TileLink: Generating Efficient Compute-Communication Overlapping Kernels using Tile-Centric Primitives
43. Triton-Distributed：Programming Overlapping Kernels on Distributed AI Systems with the Triton Compiler
44. FlexAttention：https://pytorch.org/blog/flexattention/
45. FlashMask：[2410.01359] FlashMask: Efficient and Rich Mask Extension of FlashAttention
46. Qwen3：[2505.09388] Qwen3 Technical Report
47. ERNIE 4.5：ERNIE 4.5 Technical Report
48. Kimi K2：Kimi K2: Open Agentic Intelligence**

