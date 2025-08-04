# 大规模分布式 AI 模型训练系列——张量并行

**作者：** AI闲谈

---

一、背景

之前的文章中我们详细介绍了大规模分布式训练中的数据并行（Data Parallelism）以及相关技术，比如梯度下降、Adam 优化器，以及集合通信中的 AllReduce 操作等。本文中我们继续介绍分布式训练中的张量并行（Tensor Parallelism，TP），包括 AlexNet、Google、Facebook、NVIDIA 以及 Colossal-AI 的一系列 Tensor Parallelism 方案。涉及 1D TP，2D 和 3D TP，也包含行切分和列切分等。

这里说的 Tensor Parallelism 和 Zero DP 以及 Pytorch FSDP 中的模型切分方式不一样，Zero DP 和 FSDP 中的模型切分在实际使用的时候还会将相应的参数 AllGather 到当前设备，使用全部的参数进行计算。而 Tensor Parallelism 中的参数都始终在当前设备，最终聚合的是结果（Activation）。当然，在后续的文章中我们也会介绍 Zero 和 FSDP 相关方案。

相关内容可以参考之前的文章：

- [大规模分布式 AI 模型训练系列——数据并行](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247487775&idx=1&sn=52981f832c8ad7c9b111e37c0e788c3a&chksm=c364d65af4135f4cc999fd39659936f42bedc7faebeb2e2a674d5feb064bf50b68a6d412b89b&scene=21#wechat_redirect)
- [MoE 系列论文解读：Gshard、FastMoE、Tutel、MegaBlocks 等](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247486329&idx=1&sn=32935ff35dc32bb04b4e222fb9b45405&chksm=c364cc3cf413452a2205dc10400e755378c3435b0a180f3d7ba74c15d235e07af709ad61dd10&scene=21#wechat_redirect)
- [万卡 GPU 集群实战：探索 LLM 预训练的挑战](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247486852&idx=1&sn=9f9dc1df99ab6aafb28e091f4532b89e&chksm=c364cac1f41343d7b10d9d234d1c7f3371d996afda01cb94d294a38cba4f1a14fe4594992aa2&scene=21#wechat_redirect)
- [万字综述：全面梳理 FP8 训练和推理技术](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247487327&idx=1&sn=784f76c54183fd46dd7300ab7b28cfcb&chksm=c364c81af413410cd1a38f816d7591ce4b0ce38314809a0695d5d9a4b544e8cfbbe16a967cd1&scene=21#wechat_redirect)

## 二、分布式矩阵乘

Tensor Parallelism 的核心就是分布式矩阵乘法，其在高性能计算领域已经有很长的历史，也已被广泛研究。在 Tensor Parallelism 中有两种常见的切分方式，Column Parallelism 和 Row Parallelism。如果从模型的的角度考虑，通常指的是 Y=XW 的形式，其中 X 为输入，W 为权重参数，Y 为输出。而具体是哪种切分方式也可以看 W 矩阵是在哪个维度切分的。（PS：这个小节的图片来自 Tensor Parallelism — PyTorch Lightning 2.4.0 documentation）

### 2.1 Column Parallelism

如下图所示为 Column Parallelism，其中的 Column 就是指权重参数 W 按照 Column 维度切分。每个 GPU 都包含一部分权重参数，并使用整个输入 X 计算，得到 Y 的一部分，最后通过 AllGather 操作可以获得全量结果。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWz9AFYibWuGjQGWyJCibapKWlIqyiaxpTiaiaWrOvMBXdYtE0ftTgRYUYlMQ/640?wx_fmt=png&from=appmsg&randomid=e8az9m04)

### 2.2 Row Parallelism

如下图所示为 Row Parallelism，其中的 Row 就是指权重参数 W 按照 Row 维度切分。每个 GPU 都包含一部分权重参数，并使用部分输入 X 计算，结果和 Y 的 Shape 相同，但结果不完整，最后通过 AllReduce 操作可以获得全量结果。因为 AllReduce 可以通过 ReduceScatter 和 AllGather 的方式实现，而 Column Parallelism 中的 AllGather 和 Row Parallelism 中 AllGather 通信量是一样的，因此，总体来说 Column Parallelism 的通信量更少：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWfzLlBkKld5EqHgBOgibxvnCflaVdBaGIfCt14DOdaTd0cDMaciayhXCQ/640?wx_fmt=png&from=appmsg&randomid=1aacvdf3)

### 2.3 Column Parallelism + Row Parallelism

在 Transformer 等模型中会存在连续两个矩阵乘法（Linear Layer）的情况，此时通常都会采用先 Column Parallelism，之后 Row Parallelism 的方式切分，可以在两个 Linear 之间减少一次通信操作。如下图所示，W 是第一个 Linear 权重，V 是第二个 Linear 权重。只用在最后进行一次 AllReduce 操作即可：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWpbEND0PoEvoSPncwxtn3AKrocLZp9MicEX84wvA9ZIHGTpjpia6sjVHQ/640?wx_fmt=png&from=appmsg&randomid=8l40p80y)

## 三、AlexNet

Tensor Parallelism 在 AI 模型中的应用最早可以追溯到著名的论文（2012: ImageNet Classification with Deep Convolutional Neural Networks），也就是 AlexNet，作者是 Alex Krizhevsky，Ilya Sutskever 和 Geoffrey E. Hinton。如下图所示为 AlexNet 模型的网络结构：其整个模型由 8 个可学习层组成，包括 5 个 Convolutional 层 和 3 个 Fully Connected 层；此外还有几个 Max Pooling 和 ReLU 层。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqW2k6XtGunniaRUsOlGsiaNxvLicIBJq35myRVvBwKAnFAWBdBuy4q8gbzg/640?wx_fmt=png&from=appmsg&randomid=8gmwwh40)

然而，当时流行的消费级 GPU 为 GTX 580，只有 3GB 显存，无法训练这么大的模型，因此作者采用 Tensor Parallelism 的方式，当然，那个时候还没叫 Tensor Parallelism。其切分也很简单，如下图所示，Conv 层按照卷积核（Kernel）切分，而 FC 层按照神经元（Neuron）切分。由于 Conv 层也可以等价于矩阵乘操作，所以 Conv 和 FC 的切分也都可以理解为矩阵乘中将 Weight 按行切分。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWrfr0X53H9Yegtzd4GrOyOzOP16CnfgoGztUIC23kaHNC9FntZpvR5g/640?wx_fmt=png&from=appmsg&randomid=uryppq9i)

2012 年贾扬清的 Caffe 还没有诞生，Alex Krizhevsky 自己实现了一个 cuda-convnet 库（Google Code：cuda-convnet），为了支持 AlexNet 的 Tensor Parallelism，也继续开发了 cuda-convnet2（Google Code：cuda-convnet2）。

PS：冷知识，NVIDIA 的 K20 GPU 是 2012 年发布的，只有 5GB 显存；P100 是 2016 年发布的，最大也只有 16GB 显存，也是在 P100 GPU 中首次引入 NVLink；NVIDIA 的集合通信库 NCCL 也是在 2016 年发布。作者本人也是在这一年用 CUDA 重构了熊厂的图像检索系统，用的还是 K1200 GPU。

## 四、Google DistBelief

和 AlexNet 同年，Google 团队也发表了 DistBelief 论文（ 2012：Large Scale Distributed Deep Networks）。论文主要讨论了如何使用大规模分布式计算集群（CPU）来训练具有数十亿参数的深度神经网络。论文的核心贡献是开发了 DistBelief 软件框架，该框架能够利用数千台机器来训练大型模型（1.7B，现在的 1.7B 已经是 Tiny Model 了 (⊙o⊙)…）。

如下图所示为 DistBelief 中的模型并行（Model Parallelism）方案，这里已经涉及 Tensor Parallelism 和 Pipeline Parallelism。下图中是一个 5 层的深度神经网络，被切分到了 4 个机器上。由于每一层不是 Fully Connected 的，而是部分连接，因此只有切分的边缘需要跨设备通信：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWaO9nwobmd6yYtylYU9I6LB7TibcAW40vFIlsF80SeLjjzVcTtt9ZqZw/640?wx_fmt=png&from=appmsg&randomid=jemiooxk)

此外，模型训练也不是采用的集合通信方式，而是使用了 Parameter Server 架构，如下图所示。也就是说模型权重的更新都是在专有的 Parameter Server 上进行，更新完后会 Broadcast 给相应的 Training Worker：![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWDrEsB27mWRs7NJhRXr7LqUIBxsyr5tbCB2ZmE9ZWWpgqibClTCSibYZw/640?wx_fmt=png&from=appmsg&randomid=d7x6ohbx)

如下图所示，作者在 2012 年已经训练了 1.7B 参数量的模型。（PS：可能是模型结构或者训练资源的制约，最终并没有在 ImageNet 上大规模训练，和 AlexNet 同期但并没有在效果上超越 2个 GPU 上训练的 AlexNet）

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWtpkgRMy8vuViaZP0eciarDbVCHR8WVfybwB26esbOfYIhXlJibbXQdEdg/640?wx_fmt=png&from=appmsg&randomid=ijgfb3ny)

## 五、Facebook TP + DP

Facebook AI 团队在 [1312.5853] Multi-GPU Training of ConvNets 中将数据并行（Data Parallelism）和 AlexNet 的 Tensor Parallelism 相结合。如下图 Figure 5 所示，GPU 1 和 GPU 2 共同包含了一份完整的模型副本，并通过 Tensor Parallelism 的方式切分模型；GPU 3 和 GPU 4 包含另外一份模型副本，两个模型部分使用不同的数据训练，并在每个 Iter 进行聚合同步：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqW9CIb2mBoZHeVhZ9l2Xd8YZQIib6N3AZur2gQ6BWyFslvluElT43sThA/640?wx_fmt=png&from=appmsg&randomid=j7hb0umb)

作者也对比了不同的分布式策略，最终使用 Tensor Parallelism + Data Parallelism 的方式获得最优的加速比：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWx3pYTK64hBpsgkMZLfvbT3PSVjJkzvILaQkfxl0uTbCIX3BWPvaEsQ/640?wx_fmt=png&from=appmsg&randomid=rev4ooqp)

## 六、Google TP + DP

Google 在 [1404.5997] One weird trick for parallelizing convolutional neural networks 中进一步对 TP + DP 进行了优化（PS：其实作者还是 Alex Krizhevsky，从 Google 离职后似乎不再进行 AI 研究相关工作；现在 AlexNet 的二作 Ilya Sutskever 更为人熟知，尤其是 OpenAI 火了之后）。如下图 Figure 1 所示，具体来说，作者发现其模型的大部分参数集中在最后的 Fully Connected 层，因此在前面的 Conv 层采用 DP，而在后面的 FC 层采用 TP。其中 DP 里每个 Work 使用不同的 Batch 数据，而 TP 中使用相同的 Batch 数据：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWOLM7rbnJtbwXYYFQWGvdElicZzxrIPuYmclkbGTOuQwC1ZXBwh4riaUg/640?wx_fmt=png&from=appmsg&randomid=c1vg12x3)

如下图 Table 1 所示，在 2GPU 和 4GPU 几乎都获得了线性加速：![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWoOKRE0lVnqCY6O946vULJ0aibnM1RYDUDfMwCDWJlgvm7WFV2Rc7N1w/640?wx_fmt=png&from=appmsg&randomid=8evkux9m)

## 七、UC Berkeley TP + DP

之前方案中讨论了 Model Parallelism 和 Tensor Parallelism，其主要都是指权重参数按 Column 的切分方式。而在 [1712.04432] Integrated Model, Batch and Domain Parallelism in Training Neural Networks 中（PS：本文中作者用的是 Y=WX 的方式，所以虽然 Weight 按照 Row 切分了，但是实际对应 Y=XA 中的 Column Parallelism），UC Berkeley 等作者进一步讨论了输入（Activation，X）相关的切分方式（作者称作 Domain Parallelism），且与 Data Parallelism 和 Column Parallelism 一起分析（比如分析通信量）、整合，并证明 Data Parallelism 或 Tensor Parallelism 单独使用都不是最优的方案。

当然，本文中作者的讨论也有一定的局限性，比如主要针对 AlexNet，并且假定所有设备之间都使用相同的拓扑连接。

如下图 Figure 5 所示，作者首先评估了 Data Parallelism + Column Parallelism 的方案。其相当于 TP=2，DP=3：

- 第一行：Forward 过程，采用的 Column Parallelism，所以计算完之后需要 AllGather 操作拿到完整数据。
- 第二行：Backward 权重梯度过程，相当于 Row Parallelism，所以计算完之后需要 AllReduce 操作。
- 第三行：Backward 链式法则（输入梯度）过程，相当于 Row Parallelism，所以计算完之后需要 AllReduce 操作。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWMtN3jPqtqWdv23u5XugJQOrH9xsiay0HMU8t0qFE6ZzBjm0f7kJbJuQ/640?wx_fmt=png&from=appmsg&randomid=9ucv7i0y)

如下图 Figure 3 就是其所述的 Domain Parallelism，因为其数据实际是按 NCHW 方式存储的，因此这里按照 Width 切分为 4 部分。这里实际就是对输入的切分：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWyXSffrDxkpWyvQm4ibibWWw2UtUo1US77kF7TIeJuXYrELk6b0yUL5Ag/640?wx_fmt=png&from=appmsg&randomid=8b9swpwx)

作者对 Domain Parallelism 的通信也进行了分析，对于 AlexNet，前积层都是 Conv 层，Activation 比较大，适合 Domain Parallelism（切分输入）；后几层是 FC 层，参数量比较大，适合切分 Weight。

## 八、Google Mesh-TensorFlow

Google 团队在 2017 年发表著名的 [1706.03762] Attention Is All You Need，此后模型规模不断扩大，Model Parallelism 已经成为不可或缺的分布式训练方案。然而，当时并没有一个高效的 Model Parallelism 的训练框架，因此 Google 的作者提出了 Mesh-TensorFlow（[1811.02084] Mesh-TensorFlow: Deep Learning for Supercomputers）框架，它是一个通用的分布式 Tensor 计算框架，用户可以在多维网格的任何维度上分割任何的张量，也就天然可以支持 Data Parallelism 以及 Tensor Parallelism（包括 Column Parallelism 和 Row Parallelism）。

此外，作者也在 512 个 TPU 上训练了 5B 参数量的 Transformer 模型（PS：本文的一作也是 Attention Is All You Need 的二作 Noam Shazeer）。

也是在这篇文章中作者对 Transformer 模型里这种 2 个 FC 层相邻的 Model Parallelism 切分方式进行了介绍。如下图所示，其中 w 和 v 是两个权重矩阵，x 为输入，y 为输出，中间激活为 Relu：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWUOCTibrNQXicXicxKyAeeON73rOVEiaz8B2PQHhRgaP9aibnC2tY9Q60Y5Q/640?wx_fmt=png&from=appmsg&randomid=lk3vn132)

针对上述 2 个 FC 相邻（比如 FFN，Attention）的计算，作者提出了可以第一个 FC 的 Weight（w）采用列切，第二个 FC 的 Weight（v）采用行切的方案。如下图所示，这样的好处是在第一个矩阵乘之后并不用通信，只用在最后的 y 这里 AllReduce 即可：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWibQZonswKXeFpAnmZpDzfsjxj7bbibd0ThhdAZN7icSJOOnMdugibxMKoA/640?wx_fmt=png&from=appmsg&randomid=mfowoa43)

其实作者也探讨了更复杂的 2D Parallelism 切分方式，如下图 Figure 5 所示。不过最终在 GitHub - tensorflow/tensor2tensor 中实现的还是上面提到的方式：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWpuxBNt684MXKYnGiaqicqvicy2WhAUxdsIViaSJF9RtuwEveicKoa4ufqWQ/640?wx_fmt=png&from=appmsg&randomid=viq08iai)

## 九、NVIDIA Megatron-LM

Google 在 Mesh-TensorFlow 中已经提供了比较丰富的 Tensor Parallelism 的支持，然而它是一种新的框架，用户如果想要使用会有比较大的改造成本。因此 NVIDIA 在 [1909.08053] Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism 中发布了 Megatron-LM 框架，只用对已有的 Pytorch Transformer 模型进行少量修改就可以实现，相应的更加简单、方便。

如下图 （a）所示，其和 Mesh-Tensorflow 类似，MLP 层的两个 FC 采用先列切（A，Column Parallelism），然后行切（B，Row Parallelism）的方案，这样两个 FC 之间不用通信：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWict7jJFpl1AVDAqVSJibQ74CgwzJiasgFFcOGu2xHXnLO0wZzMkGjLsAg/640?wx_fmt=png&from=appmsg&randomid=lkmlss0n)

如下图（b）所示，由于每个 Head 的 Attention，softmax 都是独立的，因此可以采用按照 Head 的方式切分（等价于 Column Parallelism），然后对之后的 FC 采用行切分（B，Row Parallelism），这样 Self-Attention 中间也不用通信：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWzRA6ICYSG2FElnTtVQycEibrLlkmeAIibW8bQWojr4iahz8B0msm2zcUQ/640?wx_fmt=png&from=appmsg&randomid=hzgt0xj2)

需要说明的是，作者在其中引入了 f 和 g 模块，其主要就是一个通信的抽象，比如如下图所示，f 的 Forward 就是一个 Identify 操作，Backward 是一个 AllReduce 操作；而 g 正好相反，Forward 是一个 AllReduce 操作，Backward 是一个 Identify 操作。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWsvePAjQj0x2e66YG2Y2v0sLapKqEgFrYI9HibyBKvZEZjLEudFF4iajQ/640?wx_fmt=png&from=appmsg&randomid=p2200y69)

如下图 Figure 4 所示，采用这种 Tensor Parallelism 的 Transformer 模型的每一层在 Forward 和 Backward 都各只有 2 次 AllReduce 操作：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWSZr4PFCOPE2fDwwckicib76tEvRX5ge8BiaPQuciaiaJibZ19yKQJibShmbjg/640?wx_fmt=png&from=appmsg&randomid=z9j0cja1)

## 十、Colossal-AI 2D Tensor Parallelism

在 [2104.05343] An Efficient 2D Method for Training Super-Large Deep Learning Models 中，作者进一步提出了 2D Tensor Parallelism 的方案 Optimus，它是一种高效的 2D Tensor Parallelism 并行范式，可以进一步促进无限大 LLM 的训练，Colossal-AI 作者也在 GitHub - hpcaitech/ColossalAI 中开源了相应代码。

作者提出 2D Tensor Parallelism 是因为观察到之前的方案（比如 Megatron-LM，1D）中虽然将模型的参数分配到了多个设备上，但是每个设备上 Forward 和 Backward 的 Activation 并没有被有效切分，还会占用相应的显存空间。此外，虽然 Mesh-Tensorflow 中进行了多维 Tensor Parallelism 的抽象，但是并没有很好的实现。因此，作者提出了 Optimus 方案，以最大化通信效率，最小化显存占用。

PS：这里有个约束，2D 的两个维度需要是相等的，也就是说 2D 的数量只能是 2x2=4， 4x4=16， 6x6=36， 8x8=64 这些。

如下图所示为 Megatron-LM 对 Transformer 模型的切分方案，其中绿色为激活，蓝色为权重。可以看出，权重都在各个设备上进行了 1 维切分，而 Input 和 Output 在每个设备上都是保存的整个副本，也就是存在存储上的冗余。作者将这种方式称为 1D Tensor Parallelism。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWKfbuhuicSIcowGg5wM1pelsKgc4x0YXt5IA0WGDO38HqHBx5MEBuruA/640?wx_fmt=png&from=appmsg&randomid=f67m8nk6)

对于 1D Tensor Parallelism，给定 P 个设备，其计算、内存和通信成本如下所示：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWIxF3TVsam9YOYykdpjPWLS1B4VcQIDC7Gq900Kzt7ZGMNrY4RBJZBg/640?wx_fmt=png&from=appmsg&randomid=idsbyurb)

如下图所示为本文的 2D Tensor Parallelism 切分方案，可以看出，不管是权重还是 Input、Output 都按照设备进行了切分，此时设备也是按照 2D Mesh （q，q）进行排布。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqW8g9eg58A6tnuVFnsichiayMvWKFkoUgrYCa7ia6dxfEZspnUVEk9ZufZw/640?wx_fmt=png&from=appmsg&randomid=p5s72wmo)

对于 2D Tensor Parallelism，给定 P = q x q 个设备，其计算、内存和通信成本如下所示：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWaa0abySBV5tdujwic98fJvofywbApnu0rF8TsU4K0dEcSCEvh9iaNjAQ/640?wx_fmt=png&from=appmsg&randomid=a4xuc51t)

如下图 Table 1 所示，作者也统计了 Megatron-LM 和 Optimus 的通信量和计算量，可以看出，两者计算量相同（分块计算并不会增加计算量），而通信量有所差距，当 p 比较大时 Optimus 的 2D 方式可以有一定优势。不过当前通常都会将一个 TP 分组放在一台机器内（通常不超过 8 个 GPU），此时 Optimus 在通信上并没有什么优势，不过在显存上会更有优势，当然也会增加实现和调度的复杂度。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWxEpLDyRZB8nibu3W0y2K5H5H6oZbt4ugiaxtrQmewKyeE16tf0DzU7Zw/640?wx_fmt=png&from=appmsg&randomid=3mw23qkv)

## 十一、Colossal-AI 3D Tensor Parallelism

在 [2105.14450] Maximizing Parallelism in Distributed Training for Huge Neural Networks 中，作者在 2D Tensor Parallelism 的基础上进一步提出了 3D Parallelism。通过实现完美的负载均衡，提出的方法提供比当时最先进的 1D 和 2D Tensor Parallelism 更小的内存和通信成本。在 64 个 V100 GPU 上的实验结果表明，提出的 3D 并行比 1D 和 2D 并行分别加速 2.32x 和 1.57x。

PS：这里同样有个约束，3D 的三个维度需要是相等的，也就是说 3D 的数量只能是 2x2x2=8， 4x4x4=64 这些，无法支持 4 个设备的 Tensor Parallelism。

如下图 Figure 2 所示为一个 3D 矩阵乘的示例，其中 A 和 B 的箭头表示 Broadcast 的方向，C 中的箭头是 Reduce 的方向：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWO0AVskBnibdf617ictwqGWCBkKoAWib42iaeXuhskptOCboOBPwhulDCSg/640?wx_fmt=png&from=appmsg&randomid=mhpcxg2n)

如下图 Figure 6 所示是 Transformer Layer 的 3D Tensor Parallelism 方案：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWZkIaLjUY6tAysWoFGygu3Giba7UFBAckfTdUSJNZgmMLia7ibooPD6zOQ/640?wx_fmt=png&from=appmsg&randomid=pxt9tvkc)

对于 3D Tensor Parallelism，给定 P = q x q x q 个设备，其计算、内存和通信成本如下所示：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqW5W3EHkoUFRDQAW0t5FJJ7f8gb8ibUyP42w8Ugua59cJXuplMdr5GF5A/640?wx_fmt=png&from=appmsg&randomid=4ac37ofk)

如下图 Table 2 所示，作者对 1D，2D 和本文提出的 3D Tensor Parallelism 的方案进行了验证，其中的 Average step time(s) 为 (Forward time + Backward time) / Batch Size。从中可以看出：

- 2D Tensor Parallelism 相比 1D 只在 64 GPU 时略微有优势；而在更少的 GPU 时，反而不如 1D Tensor Parallelism。
- 3D Tensor Parallelism 相比 1D 有比较大的提升。（PS：3D 更节约显存，可以使用更大的 Batch Size，如果是 Zero-DP 或者 FSDP 的方式，通过增加 Batch Size 是否可以达到同样的效果？）

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWicPx5S0dsVrOY6iajibM8clpJFKwMLcFKlwS5tjqpZ8ME57khDQibz1HTg/640?wx_fmt=png&from=appmsg&randomid=2om15kfi)

PS：当前除了 Zero-DP 和 FSDP 外，也有 Pipeline Parallelism，Sequence Parallelism，使用 2D 或者 3D Tensor Parallelism 的场景比较少；此外，GPU 算力越来越强，3D 方式会将矩阵计算切分得更加小，不一定能充分发挥 GPU 算力，还是需要综合考虑。

## 十二、Megatron-LM TP+DP

NVIDIA 在上述的 Megatron-LM 中也介绍了 Tensor Parallelism 和 DP Parallelism 混合的方案。当前大规模 GPU 集群广泛采用 NVIDIA GPU 构建，比如 H100、A100、V100，以及后续的 B200。通常每台机器有 8 个 GPU，每台机器内由 NVLink 高速互联，通信带宽更高。因此，在排布分布式并行策略时，通常将通信量比较大且有些时候无法有效 Overlap 的 Tensor Parallelism 放在一台机器内部，而不同机器之间采用 Data Parallelism 或 Pipeline Parallelism。

如下图 Figure 8 所示为 Megatron-LM 中一个 DP + TP 的混合分布式并行方案，总共采用 64 台 8 GPU 机器，共 512 GPU。

- 每台机器的 8 个 GPU 组成一个 Model Parallelism Group（TP），共 64 个 TP Group；每个 TP Group 内的 GPU 包含不同的模型参数，并且使用相同的训练数据。
- 所有设备的同号 GPU（比如 GPU 1，9，...，505）组成一个 Data Parallelism Group（DP），共 8 个 DP Group；每个 DP Group 内的 GPU 都有相同的模型参数，但是使用不同的训练数据。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWYkIAIlvmVXJugSk0HkvUSh2VL8YwWIicK2OP5raPicEFlBQbMpNoAGdA/640?wx_fmt=png&from=appmsg&randomid=t6hwakq5)

基于以上的分布式排布方案，反向传播和梯度更新的过程如下所示（当然，下述两个阶段可以一定程度上 Overlap，以提升训练速度）：

1. 各个 TP Group 独立的进行 Backward，Backward 涉及的 AllReduce 也只在 Group 内部，也就是单个机器内。每个 GPU 上也只有对应权重参数的梯度。
2. 各个 DP Group 独立地进行梯度 AllReduce 以及权重参数更新。

如下图所示，我们在之前的文章中介绍过，当前的大规模 GPU 训练集群的网络拓扑中，同一个 Group 里不同机器的同号 GPU 对应的 NIC 会连接到同一个 Leaf 交换机，比如所有机器的 2 号 NIC 都连接到了 2 号 Leaf 交换机。这也与上述的分布式排布方案相对应，所有 TP Group 的通信都在机器内，所有 DP Group 的通信都只用经过 Leaf 交换机，此外在交换机上执行规约计算也变成了一种可能。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWsiapRuAz3AkB328LFYXKgiacz2GQEcyIUGDq18GQLFe9tQcqtUgSn88A/640?wx_fmt=png&from=appmsg&randomid=sbr9miq1)

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWVeicmmSq39nn4u5kic195IBy0lxDc7jkeoHNnqGV2LtrGwpYPvYLiaY3A/640?wx_fmt=png&from=appmsg&randomid=iczyay2l)

## 十三、分布式 LLM 推理：DP+TP

在 LLM 的推理场景也会存在 DP + TP 的组合，比如为了支持更大的吞吐、考虑容灾等场景，通常会有多个 Instance；每个 Instance 都会有完整的模型参数，这些 Instance 可以独立、并行的处理各自的 Request，这种方式可以理解为 DP，只是因为推理只有 Forward，所以不涉及权重更新，各个 Instance 之间也就不必通信。而单个 GPU 的显存或者算力可能不足，每个 Instance 可能使用多个 GPU 来推理，在 LLM 场景最常见的就是采用 TP 的方案。

然而，此时使用 TP 也可能有其局限性。如下图所示（图片来自 [2312.03134] A Hardware Evaluation Framework for Large Language Model Inference），我们之前已经分析过，一个 Transformer Layer 中会有两次 AllReduce 通信，一次是 MHA 中的最后一个 Linear，一次是 FFN 中的最后一个 Linear。以 GPT-3 175B 模型为例，其共 96 个 Layer，也就是说一次 Forward 要有 192 次 AllReduce（忽略非 Transformer Layer 之外的通信）。每次的通信量与 Hidden Dim 和 Batch Size 成正比，模型确定后 Hidden Dim 确定，其通信量就与 Batch Size 成正比。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqW1S6JetgrLWAyzfKDlt0rO09W22UNaATjPezGziaQfFMVcXakwKjem8g/640?wx_fmt=png&from=appmsg&randomid=3lnd6c6u)

由于 LLM 推理通常会使用 Continuous Batching 的方式提升吞吐，随着 Batch Size 增加，MHA 和 FFN 的 Kernel 计算时延不会明显增加（序列不太长）；而 AllReduce 的通信量却线性增加，相应的通信时延增加的更加明显，以至于 AllReduce 通信可能成为瓶颈。GPU 间的通信时延与 GPU 之间的互联方式有关，节点内通常通过 PCIe 或 NVLink 互联，在 PCIe 互联方式下就需要密切关注 AllReduce 相应的性能问题，尤其还会涉及跨 PCIe Switch 或 跨 NUMA 通过 UPI 通信的场景。

如下图所示为 Batch size 等于 1 和 512 时 LLM 中几个主要 OP 的计算耗时，可以看出，将 Batch size 从 1 增加到 512，计算量增加 512 倍，但是其整体时间只增加到原来的 3 倍左右（图片来自 openppl-public · GitHub）：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTheThUvcHZSTIhUYM6xRQqWqeYGPEa70vrMy2Siaa6TgQTyvblMqZibE0YZ3VsqPd9vKCFPibl8RXpRQ/640?wx_fmt=png&from=appmsg&randomid=i9p9swbp)

除此之外，Continuous Batching 的方式会希望组合尽可能大的 Batch Size，也就意味着同一时间可能只有一个 Batch 的数据在计算，AllReduce 通信与计算无法充分 Overlap，出现算力的浪费。针对这个问题，也有一些优化方案，比如：

- Prefill 阶段和 Decoding 阶段的细粒度 Overlap，比如新到的 Request 在执行 Prefill 计算时，可以执行之前已经 Batching 的 Decoding 的通信；而 Prefill 在通信时，可以执行 Decoding 的计算。当然，实现这种细粒度的计算、通信 Overlap 的代价也比较高。
- Pipeline Parallelism、Sequence Parallelism 以及其组合方案，这里先不展开。

## 十四、参考链接

1. https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/tp.html
2. https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
3. https://code.google.com/archive/p/cuda-convnet/
4. https://code.google.com/archive/p/cuda-convnet2/
5. https://www.cs.toronto.edu/~ranzato/publications/DistBeliefNIPS2012_withAppendix.pdf
6. https://arxiv.org/abs/1312.5853
7. https://arxiv.org/abs/1404.5997
8. https://arxiv.org/abs/1712.04432
9. https://arxiv.org/abs/1706.03762
10. https://arxiv.org/abs/1811.02084
11. https://github.com/tensorflow/tensor2tensor
12. https://arxiv.org/abs/1909.08053
13. https://arxiv.org/abs/2104.05343
14. https://github.com/hpcaitech/ColossalAI
15. https://arxiv.org/abs/2105.14450
16. https://arxiv.org/abs/2312.03134

