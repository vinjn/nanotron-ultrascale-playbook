# FP8 预训练真的成熟了吗：一些实践和分析

**作者：** AI闲谈

---

一、背景

之前我们已经总结过 FP8 的发展历程，以及其在大规模语言模型（LLM）训练和推理中的应用。如今，FP8 推理几乎已经成为行业共识，许多 LLM 推理框架已经支持 FP8 推理，且多家公司也开源了相应的 FP8 模型。例如，Meta 在最近发布的 LLaMA-3 技术报告中就提到了 FP8 推理的应用。

FP8 推理相比 INT8 推理的最大优势在于其后训练量化（PTQ，Post Training Quantization）能够获得几乎无损的精度，同时显著提升推理速度。例如，相比 FP16，在 NVIDIA H20 上可以实现 2 倍的加速效果，而在 H100 上也可以获得超过 1.5 倍的加速。

与此同时，还比较少看到使用 FP8 进行 LLM 训练的工作，目前看到的有如下几个：

- 微软发布 FP8-LM 论文 [2310.18313] FP8-LM: Training FP8 Large Language Models，并开源相应的代码 MS-AMP。
- 零一万物在 零一万物面向万卡集群的 AI Infra 建设 中提到了 FP8 预训练。
- NVIDIA 和 Mistral AI 联合发布 Mistral-NeMo-12B 模型，提到了使用 FP8 量化感知训练，但没有介绍更多细节。
- NVIDIA 的 Megatron-LM 也早已通过 Transformer-Engine 库支持了 FP8 训练。

然而，也有一些工作对 FP8 训练持怀疑态度，因此我们决定自己做一些实验，以进一步验证 FP8 训练。此外，有关 FP8 训练可能的问题，也可以参考 NVIDIA 的官方文档： [探索 FP8 训练中 Debug 思路与技巧](https://mp.weixin.qq.com/s?__biz=MzU2NzkyMzUxMw==&mid=2247546168&idx=2&sn=aff0017c5e94f85c316718869ae9670d&scene=21#wechat_redirect)。

FP8 相关详细介绍可以参考我们之前的文章：

- [万字综述：全面梳理 FP8 训练和推理技术](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247487327&idx=1&sn=784f76c54183fd46dd7300ab7b28cfcb&chksm=c364c81af413410cd1a38f816d7591ce4b0ce38314809a0695d5d9a4b544e8cfbbe16a967cd1&scene=21#wechat_redirect)
- [万字综述：全面梳理 FP8 训练和推理技术 -- 附录](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247487359&idx=1&sn=f0eebf28ac98ecc571c6a129ce7df83b&chksm=c364c83af413412c480a106c8e97068b50e3de81ba5d047e56a705e320a64b995ac43df5f825&scene=21#wechat_redirect)

## 二、To FP8 and Back Again

### 2.1. 摘要

在 [2405.18710] To FP8 and Back Again: Quantifying the Effects of Reducing Precision on LLM Training Stability 中，作者指出，之前使用 FP16 训练的稳定性不如 BF16，而 FP8 的 Bit 数更少，可能导致更多稳定性问题。因此，作者认为降低精度的训练方案必须具有与更高精度的训练方案相似的训练稳定性和超参敏感性，才能具有成本效益。同时，作者发现目前可用的 FP8 训练方法不够稳健，无法将它们用作当前方案的替代品。

PS：当然，作者也强调了使用 FP8 进行 LLM 推理是完全没问题的。

### 2.2. 实验

#### 2.2.1 FP8 训练实验

作者使用微软开源的 https://github.com/Azure/MS-AMP.git（作者使用的是 v0.3.0，当前最新的为 v0.4.0）来进行 FP8 训练验证。如下图 Figure 5 所示，作者使用 8 个 H100 GPU 进行实验，其中 MS-AMP 仅使用 O1 优化，其在 GPT-2 124M 和 LLaMA 120M 上都有比较严重的收敛性问题，在 LLaMA 120M 上使用 FP8 训练甚至无法收敛

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg4Do4UO5yLyGibRmGkM2LtYXiblPtfLcQSMSRichQf60wZFXGefsTmsVmQmpOtwOeffMM4A55m2tia0Q/640?wx_fmt=png&from=appmsg&randomid=0pv2s9hv)

#### 2.2.2 降低 Bit 数实验

如下图 Figure 6 所示，使用 E8M3、E8M4 和 E8M5 来训练 TinyLLaMA 120M 模型，依然会出现 Loss 不收敛的问题：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg4Do4UO5yLyGibRmGkM2LtYVRM629z6b10IGNkvoB4s60rnib94iapTX9kEqCicZOoU4QGQRKX9iaibQzw/640?wx_fmt=png&from=appmsg&randomid=pj3dpboo)

如下图 Figure 7 所示，进一步使用 E8M3、E8M4、E8M5 和 E8M6 训练 LLaMA 7B，在 E8M5 和 E8M6 时才能保证相对的稳定性：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg4Do4UO5yLyGibRmGkM2LtYDKHjhSZOguZ0wEKPSOf4RFsoKdfiaB3tmBWB1VvyCAfwyTS9zFo3lRQ/640?wx_fmt=png&from=appmsg&randomid=sre35oju)

## 三、Megatron-LM FP8 训练验证

### 3.1. 摘要

对于上述论文中的实验我们持怀疑态度，与我们之前看到的各种结论不符，因此决定进行相应的复现。同时我们也在考虑一个问题：如果无法充分保证 FP8 训练的稳定性以及可比 BF16 训练的精度，怎么权衡是否要使用 FP8 训练。比如说，FP8 训练相比 BF16 训练可以加速 30%，但是 Loss 下降会慢一些，那么是否要使用 FP8 训练呢？除此之外，我们也进一步测试了 GPT3 系列模型在不同 Batch Size 和 Seq Length 下 FP8 相比 BF16 训练的加速比，以便为相关决策提供参考。

### 3.2. FP8 训练 Loss 对比

训练在 8*H100 机器进行，训练数据集采用 cerebras/SlimPajama-627B · Datasets at Hugging Face。使用 NVIDIA 的 GitHub - NVIDIA/Megatron-LM: Ongoing research training transformer models at scale 训练框架，具体示例可以参考其 examples 中的 gpt3。

如下图所示为一个 1B 模型使用 FP8 训练和 BF16 训练的 loss 对比，总共训练了 21K 个 Step，其 BF16 的 loss 基本上和 FP8 相当，并且收敛趋势完全一致。当然，我们也发现 FP8 的 loss 始终会比 BF16 高一点：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/zhVlwj96tTg4Do4UO5yLyGibRmGkM2LtYbDxQH8icRibXrAFQ8U6iax3jg6gwkayumYMCP18FfBp95Qk2GFpG8libmA/640?wx_fmt=jpeg&from=appmsg&randomid=mohj91oa)

如下图所示，我们的结论与 Benchmarking Large Language Models on NVIDIA H100 GPUs with CoreWeave (Part 1) | Databricks Blog 中的结论基本一致：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg4Do4UO5yLyGibRmGkM2LtYFRgtabs46ymqVGVFQ81OKTeVTxycHWcQN8XNyGonyDLibj3ctDzXC8A/640?wx_fmt=png&from=appmsg&randomid=ni19tyr8)

除了 1B 模型之外，我们还验证了 345M 模型和 13B 模型，结论基本与上述一致。

### 3.3. FP8 训练速度对比

在验证收敛性的同时我们也同步验证了加速比，其 13B 模型 FP8 相比 BF16 可以获得 30% 左右的加速，而 1B 模型可能只有 20% 左右，更小的模型加速比甚至小于 10%。（PS：不同的分布式策略都可能产生不同的结果，我们这里只是简单同配置下的验证）

如下图所示，Benchmarking Large Language Models on NVIDIA H100 GPUs with CoreWeave (Part 1) | Databricks Blog 中 1B，3B 和 7B 模型的 FP8 训练相比 BF16 的训练加速比也只有 1.2x-1.3x：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg4Do4UO5yLyGibRmGkM2LtYzFEAU6S5oQes5rylzibzjVXrCJMmxL30zXBaafhlliaUovqpI1c6SnPw/640?wx_fmt=png&from=appmsg&randomid=31abrh0j)

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg4Do4UO5yLyGibRmGkM2LtYqWcF0cUIic5XHNG9HrWx7EoltJYTACJBVAUJn0PSxziaAkib2cia7iajwng/640?wx_fmt=png&from=appmsg&randomid=nntukand)

为了对比不同配置下的性能，我们使用 Transformer-Engine 构建了一个 1 层的 Transformer Block 进行速度对比，同样在 8*H100 上验证，采用 8TP，具体示例可以参考 Getting Started — Transformer Engine 1.8.0 documentation。

如下图所示为 GPT-3 系列模型在 Seq Length=1024 时的性能，其中：

- 红色：表示加速比小于 1，通常是模型比较小，Batch Size 比较小的情况。
- 蓝色：表示加速比大于 1 并且小于 1.3，通常是模型相当比较大或者 Batch Size 比较大。
- 绿色：表示加速比大于 1.3，同时是模型很大或者 Batch Size 很大。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg4Do4UO5yLyGibRmGkM2LtYAJxKxjjzYPcYboVt60HRU1cyd2QFeSgWAQBFnh6fJ8r6gD5TXK2s6Q/640?wx_fmt=png&from=appmsg&randomid=0lz9o8jn)

如下图所示为 Seq Length 为 2048 的情况：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg4Do4UO5yLyGibRmGkM2LtY8BHTWZrSZiaKJDZAeujqUpb0tfbPYKj2jhibDoSiaTPoUjcaa2bHZHfMw/640?wx_fmt=png&from=appmsg&randomid=hz6dd1o7)

如下图所示为 Seq Length 为 4096 的情况：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg4Do4UO5yLyGibRmGkM2LtYicmpDl0Ol6GQwe5pzhFUgManQEIe4WXYWadtOp1WsQEHIaJY19X9fCQ/640?wx_fmt=png&from=appmsg&randomid=8ox92bdb)

如下图所示为 Seq Length 为 8192 的情况：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg4Do4UO5yLyGibRmGkM2LtYFxrv60GElRaS77IUQY221qI5HRySSO1J8QKC7rsz1RNAcibzYZXpx0w/640?wx_fmt=png&from=appmsg&randomid=qvo7o7cl)

从上述结论可以看出，要想获得比较大的加速比，通常需要具有比较大的模型或者比较大的 Batch Size、Seq Length。当然，也并不是说 Seq Length 越大越好，可以看出，Seq Length 为 8K 是其加速比反而不如 4K。此外，也可以看出，大部分加速比不超过1.5x，甚至很多不超过 1.3x。（在实际使用中最好经过一些充分的分析和实验）

### 3.4. 零一万物的实践

LLM 预训练的代价很高，比如可能需要上千个 GPU 训练几个月的时间，30% 的加速比似乎有很大的吸引力。然而，其结果又像薛定谔的猫，除非同时训练一个 BF16 模型和 FP8 模型，才能确定 FP8 模型是否真的符合预期。

为了解决上述问题，零一万物在 零一万物面向万卡集群的 AI Infra 建设 中提到了一个 Trick 的方法。如下图所示，每隔一段时间就会 Load FP8 的 Checkpoint 并使用 BF16 进行训练，验证 Loss 是否和 FP8 训练的 Loss 一致。如果出现不一致的情况，就会使用 BF16 的训练代替 FP8，并在一段时间后继续使用 FP8 训练。最终作者获得了 1.3x 的吞吐提升，不过并没有说明这个提升是纯粹的 FP8 相比 BF16 还是也包含了 BF16 的校验预算。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg4Do4UO5yLyGibRmGkM2LtY6B0qtEG6ibDNJhCZ4GY7c3sIclXkJjUrHKTCjMu9CWjqJlt1RPaF1iaQ/640?wx_fmt=png&from=appmsg&randomid=jtrqjs9x)

## 四、参考链接

1. https://arxiv.org/abs/2310.18313
2. https://01-ai.github.io/
3. [https://mp.weixin.qq.com/s/ezdGxxmTRfEnzXmrVtwq7g](https://mp.weixin.qq.com/s?__biz=MzU2NzkyMzUxMw==&mid=2247546168&idx=2&sn=aff0017c5e94f85c316718869ae9670d&scene=21#wechat_redirect)
4. https://arxiv.org/abs/2405.18710
5. https://github.com/Azure/MS-AMP.git
6. https://huggingface.co/datasets/cerebras/SlimPajama-627B
7. https://github.com/NVIDIA/Megatron-LM
8. https://www.databricks.com/blog/coreweave-nvidia-h100-part-1
9. https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/quickstart.html

