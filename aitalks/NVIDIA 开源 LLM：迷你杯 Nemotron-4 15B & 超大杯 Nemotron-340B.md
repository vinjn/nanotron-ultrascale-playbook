# NVIDIA 开源 LLM：迷你杯 Nemotron-4 15B & 超大杯 Nemotron-340B

**作者：** AI闲谈

---

一、背景

NVIDIA 最近公布了其超大杯的 340B 模型 Nemotron-4 340B，实际上其在今年 2 月份就发布过 15B 版本。此次的 340B 模型使用的预训练数据还是一样的，主要是扩充了 Post Training 阶段，比如使用合成数据用于对齐。这里我们对其进行简单介绍。

如下图所示：在最新的 LMSys Chatbot Arena Leader中，Nemotron-4 340B 已经排到开源模型的 Top1：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjQxwBoB4uMEXvNibiciad7znkyfvluAPmjPFVlribSKhTC6I6d5j3WHiapKWibjn6IsXww1ztknl2XnIdg/640?wx_fmt=png&from=appmsg&randomid=w5pqw442)

Nemotron-4 15B 对应论文为：[2402.16819] Nemotron-4 15B Technical Report

Nemotron-4 340B 对应论文为：Nemotron-4 340B Technical Report

## 二、Nemotron-4 15B

### 2.1 摘要

NVIDIA 在 2024.02 发布了自研的 Nemotron-4 15B 模型，其在 8T 的预训练语料上训练而来，总共包含 15B 的参数量。其在英语、多语言和编码任务上表现出强大的性能：在 7 个下游评估任务中的 4 个里优于之前类似规模的开源模型，并在其余任务中实现与领先开源模型相当的性能。具体来说，Nemotron-4 15B 在所有类似规模的模型中表现出最好的多语言能力，甚至优于大 4 倍以上的 Palm-62B 模型和部分专门用于多语言任务的模型。

### 2.2 模型结构

其同样采用 Decoder-Only 的 Transformer 结构，具体配置如下图 Table 1 所示：

- 3.2B Embedding 参数（输入），12.5B 非 Embedding 参数，使用独立的 input、output Embedding，也就是 lm head 中有独立的 Embedding
- 使用 RoPE 位置编码
- 使用 SentencePiece BPE tokenizer，词表大小 256,000
- 在 MLP Layer 中采用 Squared ReLU 激活，没有 bias，Dropout rate 为 0
- 使用了 GQA

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjpKjQtveFS8DhVE5MAdW7oT2fGEeTsYT2dgAD5bDw0Oq03fyLSficDujNicchGVplZfffqoVV34jfA/640?wx_fmt=png&from=appmsg&randomid=yquxrxck)

### 2.3 数据

作者构建了 8T 的预训练预料，其中包括：

- 70% 英文自然语言数据
- 15% 多语言自然语言数据
- 15% 源代码数据

其英文预料包含多个来源，比如网络抓取，书籍，新闻等等，具体如下图 Figure 2 所示：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjpKjQtveFS8DhVE5MAdW7ojaq3Td5eMibPbgEzUveicynMQ7Pgb39SiaKK4VUmQW4WY24B39AGp9lAw/640?wx_fmt=png&from=appmsg&randomid=sdv829sl)

其源代码数据包含 43 种编程语言，其中最多的是 Markdown、JavaScript、Python、C、CPP、Java、Html 等，具体分布如下图 Figure 3 所示：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjpKjQtveFS8DhVE5MAdW7ocnlWmM1icmkJCWgYd42NvVPzm9mibI6XFd5SgTn8T6zmic6nTSyxvZz0Q/640?wx_fmt=png&from=appmsg&randomid=1usy38p4)

### 2.4 预训练

训练硬件：作者最多使用了 384 个 DGX H100 节点进行训练。每个 DGX H100 包含 8 个 80G SMX5 H100，使用 NVLink + NVSwitch 全互联；每个 DGX H100 还有 8 个 Mellanox 400 Gbps IB 网卡，实现跨节点互联。（PS：这基本是 NVIDIA Data Center 的标配，各项配置打满）

分布式策略：在节点内使用 8 TP（Tensor Parallelism），跨节点使用 96/192/288 DP（Data Parallelism）。DP 数有 3 种是因为使用了 Ramp-Up Schedule Warmup（PS：如果使用课程学习，序列长度逐渐增加，则一般不会使用 Ramp-Up），其 Batch Size 逐渐扩大，单 DP 的 batch size 是固定的，所以总的 Batch Size 与 DP 数成正比。其 MFU 只有 30%-34% 之间，总共训练了 13 天。（PS：NVIDIA 有钱任性，社区中大家普遍把 MFU 优化到 50%-60%，NVIDIA 依然还只有 30%-34%）

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjpKjQtveFS8DhVE5MAdW7op4JnvJFjbZGG6CXElFqPbOdpevVIIVqgZGRMaGKK7zFceJUcLYY9iaw/640?wx_fmt=png&from=appmsg&randomid=o6g6g3vh)

Continued Training：在 Pretraining 之后，作者参考 Google [2312.11805] Gemini: A Family of Highly Capable Multimodal Models 的方案，会额外进行一个继续训练。在 Gemini 中是将其与 SFT 和 RLHF 统称为 Post Training。具体来说，作者使用两个分布的数据继续训练，一个是从预训练语料中采样，当然会给高质量数据更高的权重；另一个是少量的 Benchmark 风格的对齐样本。训练中 Loss 函数保持不变，只说使用了少量数据，未具体介绍是多少，通过这种方式可以进一步提升模型质量。

### 2.5 评估结果

如下图 Table 3 所示，作者使用 Harness 评估，最终模型效果优于类似规模的 LLaMA-2 13B/34B 以及 Qwen 14B 等：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjpKjQtveFS8DhVE5MAdW7oJVNvreQfBmGNeWMe3GAR3xkm7EcTfKFg6lszVZ8mP2rxODrYqN5ibag/640?wx_fmt=png&from=appmsg&randomid=wz6bl5vt)

如下图 Table 5 所示，其数学和代码评估也取得不错的结果：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjpKjQtveFS8DhVE5MAdW7ofdCIeqW1nKX5Xaibfm9WsibQXlGFpibibXXW7BibgJFGp7rg9DVGdcLInrA/640?wx_fmt=png&from=appmsg&randomid=076ojxby)

如下图 Table 7 所示，其多语言能力优于之前专门的多语言模型：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjpKjQtveFS8DhVE5MAdW7ovlqoXicHEZNoGfogrdxV6NbXmVdUjeT08Q1qdia1NNLWdROQ0OVFOotA/640?wx_fmt=png&from=appmsg&randomid=vt19qkr2)

## 三、Nemotron-4 340B

### 3.1 摘要

NVIDIA 继续在 2024.06 发布了自研的 Nemotron-4 340B 系列模型，包括 Nemotron-4-340B-Base、Nemotron-4340B-Instruct 和 Nemotron-4-340B-Reward。该模型与当前的开源模型相比具有非常强的竞争力，并且使用 FP8 部署时可以在单节点 8* 80G H100 GPU 上部署。此外，作者在模型对齐阶段使用了超过 98% 的合成数据，展示了模型使用合成数据训练的有效性。

PS：340B 参数量，FP16 推理仅参数量就要占用 680 GB 显存，至少需要 9 个 80G A100 或 H100。使用 FP8 只占用 340GB 显存，可以在 8 * 80G H100 放下；A100 不支持 FP8，但使用 INT8 损失有可能比较大；可见，NVIDIA 还是想要大家使用 8*H100 进行 Nemotron-4 340B 模型推理。The more you buy, the more you save.

### 3.2 模型结构

其模型结构和 Nemotron-4 15B 几乎一致，只是改了部分超参数，使模型更大，具体如下图 Table 1 所示，其 Embedding 参数有 9.4B，非 Embedding 参数有 331.6B：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjpKjQtveFS8DhVE5MAdW7obGjw1otY1icYUC9ySJ1UCC6hyKWk8kibqu6Jsmz2Uhhfjxrx3VKjnEow/640?wx_fmt=png&from=appmsg&randomid=26ckwrme)

### 3.3 数据

其使用了和上述 Nemotron-4 15B 相同的 8T 预训练预料，这里不再具体介绍。

### 3.4 预训练

训练硬件：使用的硬件配置和 Nemotron-4 15B 一样，只不过是节点数最多到了 768 DGX H100。

分布式策略：在节点内使用 8 TP（Tensor Parallelism），由于模型更大，单个节点放不下，因此额外使用了 12 PP（Pipeline Parallelism），此外使用 16/32/64 DP（Data Parallelism）。DP 数有 3 种同样是因为使用了 Ramp-Up Schedule Warmup。其 MFU 只有 41%-42.4% 之间。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjpKjQtveFS8DhVE5MAdW7oxiclPpIeS7Xr2O25NcF6kRfnczmAPydvdNXHSvgWLUjTspvt0AJicWKw/640?wx_fmt=png&from=appmsg&randomid=smm7wuwj)

Continued Training：在 Pretraining 之后，同样进行了 Continued Training。同样使用了预训练语料加问答风格的对齐样本，这里介绍了共使用 1T Token。

PS：综合考虑模型大小、数据量、GPU 数、MFU 等因素，猜测 340B 模型训练天数大概是 15B 模型的 7 倍，大约 90 天。

### 3.5 对齐

#### 3.5.1 Reward 模型建模

为了开发强大的 Reward 模型，作者收集了 10K 的人类偏好数据，称作 HelpSteer2，具体可以参考 [2406.08673] HelpSteer2: Open-source dataset for training top-performing reward models。Reward 模型建立在 Nemotron-4 340B Base 模型之上，将最后的 Softmax（包括 lm head） 替换为一个线性投影层，将最后一层的 Hidden State 映射为 5 维向量，分别对应 5 个属性（Helpfulness，Correctness，Coherence，Complexity 和 Verbosity）。推理时，通过加权求和获得一个整体的奖励。

如下图 Table 3 所示，作者在 Reward Bench 上对比了不同模型的表现，其获得了最高的表现：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjpKjQtveFS8DhVE5MAdW7oHxgZzWMknzQAiaaWTbGaHK238XJUia4DgugOs75NO70QmaWOI23nibHFA/640?wx_fmt=png&from=appmsg&randomid=k5ht4vcj)

#### 3.5.2 对齐数据

作者发现，随着模型的持续提升，当前的对齐数据集已经存在瓶颈，因此作者采用合成数据来扩展对齐数据集，具体来说，其仅使用 20K 的人工标注数据（10K 用于 SFT，10K 用于奖励模型训练和偏好微调），然后合成了额外的 98% 数据用于 SFT 和偏好微调。

生成 Prompt：具体来说，作者采用 UltraChat 的方式，使用 Mixtral-8x7B-Instruct-v0.1 模型作为生成器，按照 open Q&A、writing，closed Q&A 以及 match&coding 几个任务单独生成 Prompt。作者构建了生成 Pipeline，分别按照单轮合成 Prompt，指令跟随 Prompt 以及两轮 Prompt 来生成。

生成对话数据：作者基于上述的 Prompt 调用指令模型来生成响应，然后使用 Nemotron-4 340B Reward 模型来评估质量，并进行相应过滤。

生成偏好数据：作者使用上述 Prompt 加上 ShareGPT，LMSYS，以及 GSM8K 和 MATH 训练集中的 Prompt 来生成偏好数据。

Groundtruth 作为评委：每个 Prompt 会包含多个响应，作者会使用 Groundtruth（GSM8K 和 MATH 中的标准答案，或者 Python 能计算出标准答案的）来挑选出哪些是正确答案，哪些是错误答案。

额外数据源：作者也使用了一些额外的数据集，比如 Topic following，Incapable task，STEM 数据集，以及基于文档的推理和 QA 数据，还有 Function calling 数据。

### 3.6 评估结果

如下图 Table 3 所示为其 Base 模型的评估结果，可以看出，在不少指标上都超过了当前 Top 开源模型 Qwen-2 72B、LLama-3 70B 以及 Mistral 8x22B：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjpKjQtveFS8DhVE5MAdW7oN1vAPJeEgNbjrJKmLMabtgt9JYXefk2vibuC7QgCgaGLGNQ2ecbzn8A/640?wx_fmt=png&from=appmsg&randomid=cnsuhcdv)

如下图 Table 5 所示，其 Instruct 模型也同样和当前的 Top 开源模型相当（PS：需要说明的是，常规的 MT-Bench 只使用早期 GPT-4 作为 judge，而下述指标中使用的是 GPT-4-Turbo，因此得分会和开源 Leaderboard 有所不同）：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjpKjQtveFS8DhVE5MAdW7oXBiaicG5F6fpuh6brkkCkPicslHoAI9AJ9g7BTeRQSjxBhVtasPBqjQ9Q/640?wx_fmt=png&from=appmsg&randomid=dp2uhl02)

如下图所示，作者进一步展示微调 Instruct 模型时中间阶段模型的评估指标，可以看出，通过多阶段微调，各种指标不断提升，尤其通过 DPO，IFEval Prompt-Strict-Acc 从 61.7 增长到 79.9，MT-Bench 从 7.90 增长到 8.22：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjpKjQtveFS8DhVE5MAdW7oRc4IoBEOM1JroTxFVd9pNn2ufibXNA0o4icEBM2EWSxFqoJSjSZHKbBg/640?wx_fmt=png&from=appmsg&randomid=ch7qb33i)

## 四、附录

我们介绍这篇文章的另一个原因是其在各种评估实验中都会具体介绍相应评估的配置是什么；如果是参考数据，也会提供具体的参考来源。相比很多文章，比如只说使用了 GPQA、MATH、ARC-C，具体的配置都不介绍清楚，甚至还有错误，Nemotron-4 340B 的数据就会更加可信，至少可复现性会更强一些：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjpKjQtveFS8DhVE5MAdW7o1EibFyEZBbwR0HibDT7iaMhuFCiauz0TicyuorF4Czzof1MIibUIFic50y9uQ/640?wx_fmt=png&from=appmsg&randomid=djmk8x5j)

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjpKjQtveFS8DhVE5MAdW7oyVc1XwRzUiaicKdvQegltuU89xmzdNArzZTWNG6LDUfrSNxqlhJVUCjA/640?wx_fmt=png&from=appmsg&randomid=zvfzwan4)

Claude 3 的 Paper The Claude 3 Model Family: Opus, Sonnet, Haiku 也是个很好的例子，如下图所示为 Claude 3 中的评估结果：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjpKjQtveFS8DhVE5MAdW7oQvQRRSyS5Tv1VjuGiavjFSGia3XxXpuMWrI6FiajRibY5m6btAAicOJ5Hibw/640?wx_fmt=png&from=appmsg&randomid=6vrbw6ig)

## 五、参考链接

1. https://arxiv.org/abs/2402.16819
2. https://d1qx31qr3h6wln.cloudfront.net/publications/Nemotron_4_340B_8T_0.pdf
3. https://arxiv.org/abs/2312.11805
4. https://arxiv.org/abs/2406.08673
5. https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf
6. https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard

