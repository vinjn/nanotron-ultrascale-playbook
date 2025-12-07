# 全面解析 Google TPU 演进：从 TPUv1 到 TPUv7

**作者：** AI闲谈

---

**## 一、背景

之前的文章中详细介绍过 NVIDIA GPU 系列以及 AMD GPU 系列，本文借着 Google TPUv7 Ironwood 发布的契机，详细梳理一下 Google TPU 系列的发展历程以及关键指标。相应的参数对比如下表所示：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGib6q1pQhYjk4icssHE2VvRh6OZkiccfTHIzEVU6DLlRjQ8jpOGWznKJwA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)

NVIDIA GPU 和 AMD GPU 相关介绍可以参考笔者之前文章：

- [GTC 2025 |  GB300 系列 GPU 的最新演进：DGX B300 & GB300-NVL72](https://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247489531&idx=1&sn=fcfa0e0654ea51a4cbc6f4d82999ac70&scene=21#wechat_redirect)
- [万卡 GPU 集群互联：硬件配置和网络设计](https://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247486775&idx=1&sn=abf7af24181cf5189e113fb161cc8d30&scene=21#wechat_redirect)
- [NVIDIA 最新 GPU 解读：GB200、NVL72、SuperPod-576GPU](https://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247486291&idx=1&sn=9be7845ca2ce03a9b15cdc9848d70cef&scene=21#wechat_redirect)
- [GPU 关键指标汇总：算力、显存、通信](https://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247484942&idx=1&sn=2b69b610d4dacdc372036916d4c91325&scene=21#wechat_redirect)
- [NVIDIA B200/B300/GB200/GB300 集群互联](https://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247490782&idx=1&sn=273ed35cfe5b8f69fd0b80de26051af7&scene=21#wechat_redirect)
- [全面梳理 AMD CDNA 架构 GPU：MI325X 等 8 种 A/GPU 介绍](https://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247488222&idx=1&sn=282545e3e3c796edac8fe47b2918bfc7&scene=21#wechat_redirect)

## 二、引言

### 2.0 TPUv1

如下图 Figure 1 和 Figure 2 所示为 TPUv1 的架构。考虑到 TPUv1 比较特殊（比如这里还是用的 DDR3 DRAM），从 TPUv2 开始才逐渐统一，本文中不再具体介绍 TPUv1，详细内容可以参考 [1704.04760] In-Datacenter Performance Analysis of a Tensor Processing Unit [1]。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGzdpxlN0FWKFkciaib3TAwWyyicDn9cSG5qGx2xTjfXRDzMzQ6bSHfb0ibQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

### 2.1 TPU Chip 基础架构

如下图所示为 TPU Chip 的基础架构概览，其核心是：

- TensorCore：计算单元。包括最主要的 MXU 矩阵计算单元（PS：红框的 TPUv3 only 是表示之前 TensorCore 中有 1 个 MXU，TPUv3 中有 2 个，后续可能更多），此外还有 VPU 向量计算单元。
- HBM：高速存储单元。
- ICI：芯片之间高速互联单元。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGus0GVAK9UqT1FK7EeJsuj6tP1IM02nbRmj1oRib2PIBuZXfppXfyM3A/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

TPU 在演进中的主要变化集中在：

- MXU 规模以及计算精度。
- HBM 容量和带宽。
- Die 的数量和拓扑。
- ICI 互联的进化。

### 2.2 MXU

#### 2.2.1 概述

MXU（Matrix Multiply Unit）是 TPU 中最核心的硬件单元，用来执行大规模矩阵乘法，相当于 NVIDIA GPU 的 “Tensor Core”。在不同的代系中略有差异，叫法都相同。

MXU 采用脉冲阵列（Systolic Array）架构，其每个处理单元（Process Element，PE）执行小型计算（如乘积和累加）并将“结果（或者输入）” 传递给相邻 PE。

#### 2.2.2 Systolic Array - 示例

如下图所示为一个 A3x3 * B3x3 = C3x3 的矩阵乘法示例：

- 左上为初始状态，预期：
- PE00 为 A 的第 0 行与 B 的第 0 列的内积，PE00 = a00*b00 + a10*b10 + a20*b20。
- PE12 为 A 的第 1 行与 B 的第 2 列的内积，PE12 = a01*b02 + a11*b12 + a21*b22。
- 中上为第一次执行（1 个 PE 有效，计算量为 1*2）：
- a00 输入 PE00，b00 输入 PE00，PE00 完成计算 a00*b00。
- 右上为第二次执行（2 个 PE 有效，计算量为 (1+2)*2）：
- PE00：输入 a10 和 b10，完成计算 a00*b00 + a10*b10。
- PE01：输入 a00 和 b01，完成计算 a00*b01。
- PE10：输入 a01 和 b00，完成计算 a01*b00。
- 左下为第三次执行（6 个 PE 有效，计算量为 (1+2+3)*2）：
- PE00：输入 a20 和 b20，完成计算 a00*b00 + a10*b10 + a20*b20。
- PE01：输入 a10 和 b11，完成计算 a00*b01 + a10*b11。
- PE10：输入 a11 和 b10，完成计算 a01*b00 + a11*b10。
- PE02：输入 a00 和 b02，完成计算 a00*b02。
- PE11：输入 a01 和 b01，完成计算 a01*b01。
- PE20：输入 a02 和 b00，完成计算 a02*b00。
- 中下为第四次执行（7 个 PE 有效，计算量为 (2+3+2)*2）
- 中下为第五次执行（6 个 PE 有效，计算量为 (3+2+1)*2）

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGq8RNib8ia8yoZTmO2e6wVRI8nRkFq9lQFDXtjG9UGF7pd8Uxhye5FpPQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

#### 2.2.3 Systolic Array - Bubble

从上述的示例可以看成，对于一个 ANxN * BNxN = CNxN 的矩阵乘法， 需要经过 3*N-2 次迭代才能完成计算，并且每次迭代总是有计算的浪费，有效计算的比例为：

2*N*N*N / ((3N-2) * (2N*N)) = N/(3N-2)

可以看出，N 越大，有效计算的比例越小，浪费的算力越多，有效计算的极限是 1/3。

实际上，一个 NxN 的计算单元也可以计算 ANxK * BKxN = CNxN 的矩阵乘法。如下图所示，当 K 远大于 N 时，可以大幅提升有效计算的比例，只在首尾有些 Bubble 存在。如下图所示：

- PE00 = a00*b00 + a10*b10 + a20*b20 + a30*b30 + a40*b50 + a50*b50。
- PE12 = a01*b02 + a11*b12 + a21*b22 + a31*b32 + a41*b42 + a51*b52。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGAndBjN4ibgibX030GIID6P8Q9G3nu2zIOGOnuOv2v22ElnQmyFmnB0AA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

当 K 远大于 N 时，上述计算大概可以分为 5 个阶段，共 K + 2N - 2 个 Step：

- 阶段 1：N 个 Step，计算量从 [1] 增加到 [1, 2, 3, …, N]。
- 阶段 2：N - 2 个 Step，计算量从 [2, 3, …, N, N] 增加到 [N-1, N, …, N, N]，此阶段较少 PE 被浪费。
- 阶段 3：K - 2N + 2 个 Step，计算量始终为 [N, N, …, N, N]，此阶段所有 PE 都参与计算。
- 阶段 4：N - 2 个 Step，计算量从 [N, N, …, N, N-1] 降低到 [N, N, …, 3, 2]，此阶段较少 PE 被浪费。
- 阶段 5：N 个 Step，计算量从 [N, N-1, …, 2, 1] 降低到 [1]。

上述 5 个阶段可以理解为下图所示方式，每个 Step 利用的 PE 个数为虚线框中元素的个数：

- 第 1 和 K+2N-2 个 Step 中仅使用 1 个 PE。
- 从第 Step 2N-1 开始使用所有的 N*N 个 PE。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGp470BVx7yRaicfhRibA4D3R6iaNn7npPvwK7Z6T3QQz1fK2qibnmunyFRg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)

综上，可以得出，K 越大，阶段 3 越长，有效计算越多。当 K 远大于 N 时，浪费的 PE 算力几乎可以忽略。

#### 2.2.4 Systolic Array - 数据流

在上述的实例中，PE 中保留的是累积的结果 C，流转的是两个输入 A 和 B；实际上也可以驻留 A 或者 B，流转的是另一个输入和计算的中间结果。考虑到在 AI 模型场景中 A 和 B 分别对应权重和激活，C 对应输出，因此也常将按照驻留的内容将其分为 3 种典型的数据流模式：

- 输出驻留（OS）：如下图 Figure 5a 所示，也就是我们上面的示例，比较适合 GEMM、MLP 等。
- 权重驻留（WS）：如下图 Figure 5b 所示，权重 W 在计算前预先填充在 PE 中，并在 PE 里驻留。
- 输入驻留（IS）：如下图 Figure 5c 所示，预先将输入 I 填充到 PE 中，并在 PE 里驻留。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGeOzNZndm3jk7rvyakPFr1qdGKnM7ZxIuL8CjDFA2WtwSndG1icpKr0A/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)

#### 2.2.5 Systolic Array - 优劣势

Systolic Array 比较适合矩阵乘法、卷积运算，由于元素只用输入一次，之后都是在 Systolic Array 内部流转，因此对内存带宽的要求相对较小。此外，这种设计也更加简单、高效。

不过，由于 Systolic Array 的这种特性，其不太适合作为通用计算范式，因为往往很难适配 PE 的数据流或者编排要求。

### 2.3 Scalar Unit/Core Sequencer

Scalar Unit 是计算的起点，它从本地指令存储器（instruction memory）获取完整的 VLIW（Very Long Instruction Word） 束，在本地执行 Scalar 操作 Slot，随后将解码后的指令转发到 Vector 或 Matrix 计算单元执行后续操作。VLIW 指令束包含 322-bit，包括 2 个 Scalar Slot、4 个 Vector slot（两个用于 Vector 加载和存储）、2 个 Matrix Slot（push 和 pop 各一个）、一个混合 Slot（比如延迟指令），6 个立即数。

如下图 Figure 3a 所示（来自：The Design Process for Google's Training Chips: TPUv2 and TPUv3 | IEEE Journals & Magazine [2]）为 Scalar Unit 的结构示意图：

- 左上绿色部分为 Instruction Bundle Memory。
- 左中黄色部分为 Scalar Decode and Issue。
- 右侧为 Scalar 执行区。
- 右下设有通往内存系统（主要面向 HBM）的 DMA 端口，通过该端口向本地的 Scalar Memory SRAM（4K x 32-bit） 发起加载和存储请求。
- 数据随后流入 32 个 32-bit 的 Scalar 寄存器。
- 最终输送到右上方的双发射 ALU。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGu2zTGJhdKcfHQlNPAA4RO4f4PwHc1IugzEVbiaT5gIRZmENlG7J4XOw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=7)

### 2.4 Vector Unit

在 Scalar 执行完后，指令束及最多 3 个 Scalar 寄存器值将被转发到 Vector Unit。如上图 Figure 3b 展示了单个 Vector lane 的架构，整个 Vector 包含 128 个 Vector lane。

- 每个 Vector lane 额外配置了称为 sublane 的 8-way 执行维度。
- 每个 sublane 包括一个双发射 32-bit ALU，该 ALU 与 32 层深度的寄存器文件相连。

综合来看，该 Vector Unit 支持每个时钟周期对 8 组 128-wide 向量执行计算，sublane 能够提升 Vector 计算与 Matrix 计算的比例，对 Batch Normalization 运算很有意义。

各个 lane 的寄存器文件可对其对应的 Vector 存储器局部切片执行加载和存储操作，该存储器通过 DMA 系统连接（主要提供 HBM 的访问接口）。如下图 Figure 3b 展示了与 Matrix 单元的连接结构：

- push 指令 slot 将数据向量发送到 Matrix Unit。
- 结果 FIFO 队列负责接受 Matrix Unit 返回的结果向量，这些结果可通过 pop 指令 slot 存入 Vector 寄存器。

## 三、TPUv2

### 3.1 TPUv2 Chip

如下图所示，每个 TPUv2 Chip 包含 2 个 Tensor Core 和 16GB HBM：

- 16GB HBM：
- 两个 8GB Stack。
- 共 600GB/s HBM 带宽。（PS：也有介绍是 700GB/s ?）
- 2 个 Tensor Core:
- 时钟频率 700 MHz。
- 每个 Tensor Core 包含：
- 1 个 MXU，一次可以完成 128x128 个 16-bit 的乘加操作。
- 总的算力为：700M * (128*128 * 2) * 2 = 45.9 TFLOPs。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGbObbRDOtO4iaqF7ibkQlx28oBEuGRCnjT6y4cMIViajofTV9ticBdNjKXg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=8)

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGAm7oAuQHBWcaBfBwKUcP9DB90Q7vbZQkvFxqq7lh2fqXRrqWCnsh8A/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=9)

### 3.2 TPUv2 Board

如下图所示为一个 TPUv2 Board（来自：Under The Hood Of Google’s TPU2 Machine Learning Clusters [3]），其包含：

- A：4 个 TPUv2 芯片（上面是散热片，这一代还是风冷散热）。
- B：每个 TPUv2 对应 2 个 BlueLink 接口，每个接口带宽为 25GB/s。根据 IBM BlueLink 技术规范，每个传输方向 8 条 200Gb/s 信号通道（共 16 条通道），构成最低 25GB/s 的配置单元（称为"子链路"）。
- C：每个 Board 上两个，推测为 10Gb/s 的以太网接口，或者 100Gb/s 的 Intel OPA 接口（Omni-Path Architecture）。
- D：Board 电源连接器。
- E：可能是网络交换机。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGV5iaiasxPuf3kdia1zLEJ1Ar7lEEOWBTScoY6McW79xFxEXwNexP3t6AQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=10)

### 3.3 TPUv2 Supercomputer

TPUv2 提供 4 条定制化的 ICI（Inter-Core Interconnect）Link，每条 Link 的传输速率达到 62GB/s（496 Gb/s）。ICI 技术使得芯片间能够直接互联构建 Supercomputer。如下图 Figure 1 所示，对应一个 16x16 的 2D Torus 互联架构，任何一个 TPUv2 都和上、下、左、右 TPU 通过 ICI 互联。每个芯片的互联带宽为 4x62=248 GB/s。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGy1eA7deYN2wZmpZhaMicD5cDJZdrxGrQMK5ia6diaeR63YeEvN0vCD8og/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=11)

如下图所示为 TPUv2 Rack 配置，每个 Rack 都包含 32 个计算单元，其中：

- A 和 D 为 CPU Rack，每个 Rack 包含 32 个 CPU。
- B 和 C 为 GPU Rack，每个 Rack 包含 128 个 TPUv2。
- 4 个 Rack 最多 64 个 CPU，256 个 TPUv2，对应上述的 16x16 2D Torus。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGC61P6A8FlKbQbBBbBlgTcnG0GrHN9kiby7FTWDpY5fXy7jxPic2aibVGw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=12)

## 四、TPUv3

### 4.1 TPUv3 Chip

如下图所示，TPUv3 芯片与 TPUv2 芯片略有不同，变化是：

- 每个 TensorCore 里面额外增加了一个 MXU 单元：
- 也就是每个 Tensor Core 包含两个 128x128 的 BF16 MXU，是 TPUv2 的两倍。
- 考虑到时钟频率也提升到 940MHz，因此两个 TensorCore 对应的总算力为 940*(128*128*2*2)*2=123.2 TFLOPs。
- HBM 升级到 32GB，带宽升级到 900 GB/s。
- ICI 带宽提升到 82GB/s（656 Gb/s），4 Link 总带宽 328GB/s。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGSe0Nhs8URabNYqh7JXIuQqJRjsJtBOkshwibeMpCxWGkZelBlwSGb5g/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=13)

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGeRP4s83sNdOLRp7rDfzR3W3eicAUzbRXMaSVfdjR9ox99nZtBHFgeNg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=14)

详细参数对比如下图 Table 3 所示，其功耗只增加到 1.6x（280 -> 450），而算力增加到 2.67x（46 -> 123）。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGgJULJP0W2xQvYyK9S6fFvTgQ6lVLnnSUzCSBphJe6ib2pItDtYJ99ug/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=15)

### 4.2 TPUv3 Board

如下图所示为 TPUv3 的基板：

- 同样是 4 个 Chip。
- 从 TPUv3 开始采用液冷散热。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGfPECaKfpaR4nRMMSRU448rDCfBQVBaQeJYiaGdicSvsEKictfJhSq9d1g/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=16)

### 4.3 TPUv3 Supercomputer

如下图所示，TPUv3 Supercomputer 共 8 个 TPU Rack，每个 Rack 依然 128 个 TPU 芯片，共 1024 个 TPUv3 芯片。依然是 2D Torus 拓扑互联，也许是 32x32=1024。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGbvfct2qrh3osHMUa58oQhNeYFjevHp5ZvGqlHw4ubwK27SPnicO0WCA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=17)

## 五、TPUv4

### 5.1 TPUv4 Chip

如下图所示，TPUv4 芯片架构进一步进化：

- 每个 TensorCore 上包含：
- 4 个 128x128 MXU 单元。
- 1 个 VPU 单元，对应 128 lanes，每个 lane 对应 16 个 ALU。
- 16 MiB 的 VMem。
- 两个 TensorCore：
- 共享 128 MiB 的 Cmem。
- 时钟频率提升到 1050MHz，对应的总算力为 1050*(128*128*2*4)*2=275 TFLOPs。
- 依然是 32 GB HBM，不过带宽提升到 1200GB/s。
- ICI Link 从 4 个提升到 6 个，不过带宽下降，从每 Link 70GB/s 下降到 50GB/s（PS：之前的介绍中提到 TPUv3 的 ICI Link 带宽为 82GB/s？）。
- 其中 OCI 表示 On-Chip Interconnect。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UG4j2Hgmq7OcQaRLTKkfPNiajQwEpw8Q0UCeNtwMH6iaf2lDSsGRvkQwDg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=18)

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGklcMqHsuARWiad5YpsEN0VlZPAGASFJWjWfpnRza48VtCLmhCvxdYUQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=19)

详细参数对比如下图 Table 3 所示（来自：[2304.01433] TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings [4]）：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGoKDoFibrtdgQALeibXsohxmGzj125OquA0b9qBo501UCNNz2w0UicrEow/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=20)

### 5.2 TPUv4 Board

如下图所示为 TPUv4 Board，包含 4 个芯片，使用液冷散热。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGziaqBmhqXmaTLT8iawicLcrGhR1wuh7jdShwPZia3YO0icwMbPVrTPCPwdA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=21)

### 5.3 TPUv4 Supercomputer

#### 5.3.1 3D Torus

从 TPUv4 开始，每个 Rack 中的 64 个 TPUv4 会构成一个 4x4x4 的 3D Torus 拓扑：

- 立方体内部的 TPUv4 通过 6 个 ICI Link 分别连接上、下、左、右、前、后的 6 个 TPUv4。
- 6 个平面，每个平面都会空余 4x4=16 个 ICI Link。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGRAwia1Yk0LYXuSVs3LTQkwdB2x49S1Z4ORDsSdSCpt4G1adibmJUdA0g/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=22)

#### 5.3.2 Palomar OCS

如下图 Figure 5 和 Figure 6 展示了 Palomar OCS（Optical Circuit Switche）的工作原理及对应关键组件。

- 输入/输出的光信号通过 2D 光纤准直器（fiber collimator）阵列进入光学 core。
- 每个准直器阵列由 NxN 光纤阵列和 2D 透镜阵列组成，这里是 136x136。
- 如下图 Figure 5 中绿色线条所示，每个带内光信号依次穿过各准直器阵列的 Port 及两个 MEMS（Micro-Electro-Mechanical Systems） 反射镜。通过驱动反射镜偏转，可将信号切换到对应的输入/输出光纤准直器。
- 整个端到端光路具备带宽互易特性，支持与数据速率无关的双向通信。
- 该系统最终形成双向、无阻塞、全互联的 136x136 OCS（对应 136 个 Port，每个 Port 都有一个 input 和 一个 output，任何一个 Port 的 Input/Output 都可用和其他 Port Output/Input 通信 ）。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGqdY2VcXhKgANyLh3V6GoIAshT57oicOVCYSSwCUASppInMoH7emUH6w/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=23)

#### 5.3.3 Supercomputer

一个 OCS 对应 136 个 Port，其中 128 个用于连接 3D Torus TPUv3，8 个用于测试或备份。而 3D Torus 中的两个相对 Link 需要连接到一个 OCS，也就是一个 3D Torus 有 48 个 Pair Link。因此：

- 一个 3D Torus 最多连 6*16/2 = 48 个 OCI。
- 一个 OCI 最多连 128/2=64 个 3D Torus。
- 最多对应 64 个 4x4x4 的 3D Torus，也就是 4096 个 TPUv4 芯片。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGVpRuDYnDmTxjNTibX52ibxEuWYpEZl7fZI1jibVWHJzmibtmkJT5ja3DIQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=24)

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGUgzMsPLGowxMVhIfrkWpLt1n5hicrG9iaNJuv2HEPkv4TUb4EAkAgwqg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=25)

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGKhXibZyEXomBUQF6ABcWPBu7rf0psUqUZiacPWbR0B5yHfhsWFLTzXLw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=26)

PS：这种 OCS 的拓扑结构是实现容错的关键。直接互联的方式容易因为单个节点故障影响整个集群，因此规模不会特别大；而通过 OCS 可以动态调整网络拓扑，绕过故障单元，实现更加灵活的分组，提升系统可用性。对于芯片规模达数十万量级的超大规模系统，多个 Supercomputer 模块可通过 DCN（Data Center Network）实现互联。

### 5.4 TPUv4 Slice

#### 5.4.1 TPUv4 拓扑变体

实际任务通常不会使用整个 Supercomputer，因此 Google Cloud 也提供了更小粒度的多 TPU 切片（TPU v4 | Google Cloud Documentation [5]）。比如：

- 512 个 TPUv4 可以采用的切片方式为：4x4x32、4x8x16 或 8x8x8。
- 2048 个 TPUv4 可以采用的切片方式为：4x4x128、4x8x64、4x16x32 和 8x16x16。

不同的切片方式往往与分布式并行策略密切相关，往往将并行性维度映射到 TPU 切片维度时性能最佳。比如对于 8x16x16 的 TPU v4 切片，使用 8 路或 16 路模型并行（映射到其中一个物理 TPU 拓扑维度）的性能更高。

如下图是小于 64 个 TPUv4 切片时支持的切片方式，这里的 V4-64 表示 64 个 TensorCore，对应 32 个 TPUv4 芯片：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGicIfY4I27iaXFutvvhgEANBpCS9SgW8jWsTwAySS6T7jtqcflzjhAX1Q/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=27)

如下图是大于等于 64 个 TPUv4 切片时支持的切片方式：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGryqYibqO2ePrkgUaCya9Eib3dAiaXxdswlxT5HConmPwg01zuJib4Kc3qg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=28)

#### 5.4.2 Twisted Torus

相比如下图 Fig.1 和 Fig.2 这种标准的 2D Torus 和 3D Torus 拓扑，Twisted Torus（Twisted Torus Topologies for Enhanced Interconnection Networks [6]）可通过重构部分链路，以实现无需增加 Switch 硬件的情况下降低最坏情况的时延。在 TPUv4 的实现中，可以通过 OCS 的路由重编程实现。Google 在 TPUv4 集群实验，对于 All2All 通信，在 4x4x8 和 4x8xx8 切片配置中，Twisted Torus 相比标准的环状拓扑可以实现 1.63x 和 1.31x 的 All2All 吞吐提升。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UG3lfp5gubN2lx4WMxR3RC7pCgT9S67GKjGvyvAJ0muzKK3nqibicpZ69Q/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=29)

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UG5exzzFjPJ5iczHNrERic6ibgPcQGP0SIb02iaobcm5MNSWC5ZRhdtEg0eg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=30)

其他对比数据如下图所示：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UG93pOx3MFZn3x9D9S8YhljOBqjsU5yMkTKftvqKPIG6mOAWfxJ5ibFnA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=31)

## 六、TPUv5

### 6.1 TPUv5e

#### 6.1.1 TPUv5e Chip

如下图所示，TPUv5e Chip（TPU v5e | Google Cloud Documentation [7]）包含：

- 一个 TensorCore，每个 TensorCore：
- 4 个 MXU。
- 时钟频率 1.5GHz，对应的：
- BF16 算力为 1.5G * 4*(128*128*2) = 197 TFLOPs。
- INT8 算力为 393 TOPs。
- 16GB HBM，带宽为 819GB/s。
- 4 个 ICI Link，总带宽 4x50GB/s = 200GB/s（1600 Gb/s）。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGAJRRNiahFFRAFBnicZlFSeJxqEia76KSAHtY6hdjNaQkmAk7utCDjPSZg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=32)

#### 6.1.2 TPUv5e Supercomputer

TPUv5e Supercomputer 由 2D Torus 互联（只有 4 个 ICI Link），最大支持 16x16 = 256 个芯片。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGkPsGbFoVYAVWHPy4yBP4OlOnjBgIBia3uYiardydYwiabEcrBkzUvEqoQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=33)

上述 256 TPUv5e 芯片对应 4 个 TPU Rack，如下图所示。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGkRQjfNS6Tm3M8Z0ZLpB890grrlic3zHUdzIM4B3h5c8pLeRIWmVrgoQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=34)

支持如下的 2D 切片方式：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGowvFRsxZ3QZNaTuibo0NgsOvUB48fmm4tCfXzxAGpUh0gkxYtCVjEpw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=35)

### 6.2 TPUv5p

#### 6.2.1 TPUv5p Chip

TPUv5p Chip（TPU v5p | Google Cloud Documentation [8]）与 TPUv5e 略有不同，每个芯片中包含 2 个 TensorCore：

- 每个 TensorCore 同样包含 4 个 MXU、1 个 Scalar Unit 和 1 个 Vector Unit。
- 时钟频率为 1.75GHz，因此两个 TensorCore 对应：
- BF16 算力为 1.75G * 2*(4*(128*128*2)) = 459 TFLOPs。
- INT8 算力为 918 TOPs。
- 95GB HBM2e，带宽为 2765GB/s。
- 6 个 ICI Link，总带宽 6x100GB/s = 600GB/s（4800 Gb/s）。

#### 6.2.2 TPUv5e Supercomputer

TPUv5e 有 6 个 ICI Link，因此其 Supercomputer 可以构建 3D Torus 互联，包括 140 个 Rack，共 140*(4*4*4) = 8960 TPUv5e 芯片。不过，最大支持 16x16x24 = 6144 个芯片（96 个 4x4x4 立方体）。支持的切片方式如下图所示：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGyxrGzCm3B65z9XRsFpDgxmIwQKhlPcVySw0ib0p80ZEYCZ3CWWYnQvg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=36)

## 七、TPUv6e

### 7.1 TPUv6e Chip

TPUv6e Chip（TPU v6e | Google Cloud Documentation [9]）主要是对标 TPUv5e，每个芯片同样包含 1 个 TensorCore：

- TensorCore 同样包含 4 个 MXU、1 个 Scalar Unit 和 1 个 Vector Unit，只不过 MXU 变成 256x256 的 Systolic Array。
- 时钟频率为 1.75GHz，因此：
- BF16 算力为 1.75G * (4*(256*256*2)) = 918 TFLOPs。
- INT8 算力为 1836 TOPs。
- 32GB HBM，带宽为 1600GB/s。
- 4 个 ICI Link，总带宽 4x100GB/s = 400GB/s（3200 Gb/s）。

TPUv6e 与 TPUv5e 的详细对比如下图所示：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGHbDgqg0WVG1AGicXribMJwuUMA5M9xMWZ30SC9mAbegJhxOrS2u20K2A/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=37)

### 7.2 TPUv6e Supercompute

TPUv6e Supercomputer 同样由 2D Torus 互联（只有 4 个 ICI Link），最大支持 16x16 = 256 个芯片。对应的切片方式如下图所示：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGchyBCatPlAicKorfSYzuJ0nTxJbicXtAf0wtOPPgGQ58licwUQq9gC0Ig/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=38)

## 八、TPUv7

### 8.1 TPUv7 Ironwood Chip

如下图所示，Google 最近发布了最新的 TPUv7 芯片 Ironwood（Ironwood TPUs and new Axion-based VMs for your AI workloads [10]）：

- 第一个双 Compute Die TPU：
- 两个 Tensor Core，每个 Tensor Core 包括 2 个 MXU（依然是 256x256）。
- FP8 算力 4614 TFLOPs（TPU 首次支持 FP8）。
- BF16 算力 2307 TFLOPs。
- 8 个 HBM3e Stack，共 24*8=192 GB，带宽为 7.3TB/s。
- 6 个 ICI Link Stack：200GB/s * 6 = 1.2TB/s。
- 4 个 SparseCores：可用于 embedding lookup。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGVvCnI9XJJO2M2SfvfSWVTDoustAnF8dLicSicXqqL8znFSgyCrGl4NtA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=39)

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGf48ZicoN90zQ2QHkLLSrMH2H2WdGtTuJSqeAAKic9GOibsZ8j1rwxOtwQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=40)

PS：这里有个疑问，2307 TFLOPs 算力是如何得到的？按照 Google 的介绍，虽然 TPUv7 有 2 个 TensorCore，但是每个芯片上总的 MXU 数量与 TPUv6e 相同，都是 4 个；并且都是 256x256 的大小，那是如何实现将近 2317 / 918 ≈ 2.5x 的算力提？

- 如果是时钟频率提升，则 2307 TFLOPs/4/(256x256x2) = 4400MHz，不太可能。
- 也可能采用了类似 SIMD 的方式，一个时钟周期完成多次乘加操作，比如从 TPUv6e 的 1 次变成 TPUv7 的 2 次，则时钟频率为 2200MHz，相对合理。
- 还有一种可能是实际每个 TensorCore 有 4 个 MXU，总共 8 个，不过这个与 Google 官方数据不符。

### 8.2 TPUv7 Ironwood Tray

如下图所示为一个 Ironwood Tray（Board）：

- 同样包含 4 个 Ironwood 芯片。
- 每个 Ironwood 都通过 PCIe Gen5x16 与 Host 互联。
- 使用液冷散热。
- 18 个 OSFP 连接器，16 个位于图中的上部。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGW6JR2oibAvibCEyotVGzAqpZ4dvibEWZwXD4ciaAVZ9UBcq9DLZ6hbms3A/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=41)

### 8.3 TPUv7 Ironwood Rack

如下图所示，一个 TPUv7 Ironwood Rack 中包含：

- 16 个 TPU Tray，对应 64 个 Ironwood 芯片。
- 16 个 CPU Host Tray。
- 左侧为 DCN 连接，用于 Pod 间互联；右侧是 ICI Link，用于 Pod 内互联。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UG2q01RYpXRtkDbYy51duqOGiaBYcGTjtaoAbzHHSfFHmwAD1MO2KibUIA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=42)

### 8.4 TPUv7 Ironwood Supercomputer

Google 并没有介绍其 Supercomputer 的拓扑：

- 基于上述 Rack 推测，还是每个 Rack 构成一个 4x4x4 的 3D Torus。
- 则 Supercomputer 对应 9216/64=144 个 Rack。
- 每个 Rack 有 6x16=96 个 Port 连接到 OCS。
- 由于每个 Rack 相对的 Port 要连接一个 OCS，因此同样需要 48 个 OCS。
- 也就是 96*144=13824 个 Port 要连接到 48 个 OCS，每个 OCS 对应 13824/48=288 个 Port（PS：也许是如下图所示的 300x300 OCS，有 300 个 Port，剩余 12 个作为备份）。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UG3qrD3hEVvz8r0H6F7ia83o8NBe5LjdMiaDhuMK8HEIOfNVmak8zwPXqA/640?wx_fmt=other&from=appmsg&watermark=1#imgIndex=43)

上述方式对应的集群互联拓扑如下图所示：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGdiaaOZg0PThc2V8zXxOydFiaLQXo2qR7LGDOQh931FoUibDyZs6uUg5mQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=44)

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjzaibia3zNbSKXgoSYPEP3UGDvKP6icrUiaG5uuX8MLr8iau4gL8A5AvggCFnZQEaDx3bem5SjC5VsZfw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=45)

## 九、参考链接

1. https://arxiv.org/abs/1704.04760
2. https://ieeexplore.ieee.org/document/9351692
3. https://www.nextplatform.com/2017/05/22/hood-googles-tpu2-machine-learning-clusters/
4. https://arxiv.org/abs/2304.01433
5. https://docs.cloud.google.com/tpu/docs/v4
6. https://ieeexplore.ieee.org/document/5406510/
7. https://docs.cloud.google.com/tpu/docs/v5e
8. https://docs.cloud.google.com/tpu/docs/v5p
9. https://docs.cloud.google.com/tpu/docs/v6e
10. https://cloud.google.com/blog/products/compute/ironwood-tpus-and-new-axion-based-vms-for-your-ai-workloads**

