# TRT-LLM MultiShot：LLM 分布式推理 3x AllReduce 加速

**作者：** AI闲谈

---

一、背景

最近工作比较忙，看的 Paper 比较少，简单介绍两个对 LLM 分布式推理场景中 AllReduce 的优化，虽然其有各自的局限性，但也可以带来一定的启发。本文中我们首先会详细介绍分布式场景中非常常见的 AllReduce 操作以及其在 LLM 中的应用场景，然后会分别介绍 TensorRT-LLM 中基于 NVSwitch 的 MultiCatst 能力实现的优化和 Recogni 提出的基于量化压缩实现的 AllReduce 加速方案。

相关内容也可以参考我们之前的一些文章：
- [大规模分布式 AI 模型训练系列——数据并行](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247487775&idx=1&sn=52981f832c8ad7c9b111e37c0e788c3a&chksm=c364d65af4135f4cc999fd39659936f42bedc7faebeb2e2a674d5feb064bf50b68a6d412b89b&scene=21#wechat_redirect)
- [大规模分布式 AI 模型训练系列——张量并行](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247487815&idx=1&sn=69601e66f3f8413b5afbd8149b989ea7&chksm=c364d602f4135f1495f0c5e52bf911b26b528bd85f2ad1d2a97d93a358592676223bb9950ee1&scene=21#wechat_redirect)
- [](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247487815&idx=1&sn=69601e66f3f8413b5afbd8149b989ea7&chksm=c364d602f4135f1495f0c5e52bf911b26b528bd85f2ad1d2a97d93a358592676223bb9950ee1&scene=21#wechat_redirect) [全面解析 LLM 推理优化：技术、应用与挑战](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247486732&idx=1&sn=9887fdc9b6d1151aaf8c2b443d3c595d&chksm=c364ca49f413435f5f93e68195a38708cb195454272d044d7df368d78586958bef75f5075c8e&scene=21#wechat_redirect)

## 二、AllReduce

### 2.1 介绍

AllReduce 是集合通信中非常常见的分布式计算操作，主要用于多个设备（如多台服务器或多个 GPU）之间聚合数据的场景，可以包含 Sum、Min、Max 等操作。以 AllReduceSum 为例，假设有 K 个 设备，每个设备上有 N 个数据，则 AllReduce 后每个设备上的 out[i] = in0[i] + in1 [i] + … + in(k-1)[i]，也就是每个设备上的 i 位置都是所有设备 AllReduce 前 i 位置元素的和。可以参考 NCCL 的文档 Collective Operations — NCCL 2.22.3 documentation [1]。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg7JyDc1w2gg1poumLdYab05rQgt6tRsDPMTpsD7g31vz1eeSMPktslLyUicJYsh0XwB5464xGnusQ/640?wx_fmt=png&from=appmsg&randomid=knomxamk)

### 2.2 AllReduce 算法

具体的 AllReduce 操作有很多不同的实现算法，比如基于 Ring 的 AllReduce 和基于 Tree 的 AllReduce（可以参考 Massively Scale Your Deep Learning Training with NCCL 2.4 | NVIDIA Technical Blog [2]），它们之间有各自不同的场景和优劣。

Ring AllReduce：是一种环形拓扑结构的 AllReduce 算法。

- 通信过程可以理解为下述的两个阶段（后文会具体介绍）：
- Reduce-Scatter 阶段。
- AllGather 阶段。
- 优点：在每个通信步骤中，所有节点都在同时发送和接收相同大小的数据，带宽利用率高；此外，通信量也比较少。
- 缺点：假设有 K 个设备，则需要进行 2*(K-1) 次通信，延迟较高，尤其是当节点数量较多时。

Tree AllReduce：采用树状拓扑结构进行通信。

- 通信过程也包括两个阶段：
- Reduction 阶段：从叶子节点开始向根节点汇聚数据，根节点最终得到完整的结果。
- Broadcast 阶段：根节点将汇总的结果沿树结构向下广播给所有节点。
- 优点：通信步骤少，通常为 2*log(N)，因此在大规模节点时延迟较低。
- 缺点：在每一步中，只有部分节点参与通信，带宽利用率相对较低。

总体来说：

- Ring AllReduce 更适合在高带宽、低延迟的网络环境下使用，特别是当节点数量较小时，它能更好地利用带宽资源。
- Tree AllReduce 更适合在节点数量较多或网络延迟较高的情况下使用，因为它的通信延迟随节点数量增长的速度较慢。

如下图 Figure 3 所示为 Tree AllReduce 的一个示例，向上箭头是 Reduction 阶段，向下箭头是 Broadcast 阶段：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg7JyDc1w2gg1poumLdYab0sVSyCwYmZFhYEjsGRuscvuibicc10RoLHsCICYQslKibfV3QdlEJDv5NA/640?wx_fmt=png&from=appmsg&randomid=gmxzimh6)

### 2.3 ReduceScatter + AllGather

对于常见的基于 Ring 的 AllReduce 实现中，通常将一个 AllReduce 操作拆分为一个 ReduceScatter 和一个 AllGather 操作，如下图所示：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg7JyDc1w2gg1poumLdYab0dTfomsNHF5wBhK0rDicaR7nrILxk4x0lrKHGK6SQAa118kgEmtAgK1g/640?wx_fmt=png&from=appmsg&randomid=kn7vh5g2)

具体的 ReduceScatter 操作如下，每个设备（GPU）发送一部分数据给下一个节点，同时接收上一个设备的数据并累加。这个过程进行 K-1 步，ReduceScatter 后每个设备都包含一部分数据的 Sum：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg7JyDc1w2gg1poumLdYab0a5CDO7KKCco5phgLoDW00Azr0K0AOCibJpz6ASVp0bde6ZappKyQRVQ/640?wx_fmt=png&from=appmsg&randomid=5lwxp3sk)

具体的 AllGather 操作如下，每个设备（GPU）将其持有的部分结果发送给下一个设备，同时接收上一个设备的部分结果，逐步汇集完整的结果，同样需要 K-1 步。AllGather 后，每个设备都包含全量的数据：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg7JyDc1w2gg1poumLdYab01qiccmsgrw5sibttNhEgF2K2WyMSFlpIbKvjRguw4frpSKkOiawghW1UA/640?wx_fmt=png&from=appmsg&randomid=vu5r8qip)

### 2.4 NCCL AllReduce 实现

如下图所示，在 NCCL 中 RingAllReduce 的实现并不是完全分割成 ReduceScatter 和 AllGather 两个操作，而是合在一起通过 2K-2 个 Step 实现。这里的 K 是设备的数目，设备越多，需要的 Ring Step 个数越大；此外，每个 Ring Step 后都需要保持同步，这也会导致时延的增加。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg7JyDc1w2gg1poumLdYab0Dz9TiaKW7upo8Q7LvKa2oSBVFicAyFuvZFjSibqof2FlGVfoFmnrkfhxA/640?wx_fmt=png&from=appmsg&randomid=59c3j7j1)

### 2.5 AllReduce 带宽

如上所示，基于 Ring 的 AllReduce 操作可以分成 ReduceScatter 和 AllGather 两个阶段。假设设备数量为 K，每个设备的数据量为 T，并且切分为 K 份，则每一个阶段的通信量为 (K-1) * T * sizeof(dtype)，假设每个设备的总线带宽为 busBW（具体可以参考 nccl-tests/doc/PERFORMANCE.md at master [3]），则 AllReduce 对应的理论通信时延为：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg7JyDc1w2gg1poumLdYab0bY5p7YtgNmdSFJzIzGWGWLxzeLppqlOprYp8RIBJg1rCZ5OLPMZZ7Q/640?wx_fmt=png&from=appmsg&randomid=vk352xzy)

然而，实际的总线带宽并不能达到理论总线带宽。如下图所示，在 4*V100 GPU 服务器重，4 个 GPU 通过 NVLink 互联（没有 NVSwitch），每两个 GPU 之间 2 个 NVLink 连接，理论双向带宽为 100GB/s，实际测试也可以达到 97GB/s，如果 Disable NVLink（通过 PCIe），则对应的带宽只有 16GB/s。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg7JyDc1w2gg1poumLdYab0vu6C8zZUUfhR2HlXVHgygSM1g9BLq40QFw6XUprQ4BP7yGyKhjkicvA/640?wx_fmt=png&from=appmsg&randomid=v81ngyan)

AllReduce 通常更关注的是总线带宽 busBW，对于 4*V100 NVLink 互联（没有 NVSwitch），NCCL 通过如下的方式可以创建 3 个双向环，其中每个颜色都是 1 个双向环。因为每个环的单向通信带宽为 4*25GB/s，理论通信带宽的上限为 6*(4*25GB/s)=600GB/s（等价于 12 个 NVLink * 50GB/s，也就是所有 NVLink 都充分利用），那么平均每个 GPU 的理论 busBW 为 150GB/s。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg7JyDc1w2gg1poumLdYab0lAGUGrZZd7Ue9OzNwAXGZ0OiapCFAp601GH48Pw4lqtfHEPKticyEsmw/640?wx_fmt=png&from=appmsg&randomid=xe5seb5f)

实际总线带宽与理论总线带宽有一定差距存在多发面的因素，比如，数据量、GPU 连接方式、使用的通信算法、NCCL 版本等等。如下图所示为使用 nccl-tests 测试的 AllReduce 总线带宽，可以看出，当数据量比较小时，比如小于 1MB(106B)，测试的 busBW 很低，不到 30GB/s。当通信的数据量达到 128MB 时，相应的 busBW 达到 130GB/s，基本接近极限的 150GB/s。如果没有 NVLink，则实测的 busBW 只有 10GB/s 左右。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg7JyDc1w2gg1poumLdYab0urcoCGPY5EAff0HMtibDl15xHUk2IbBZj3xTMCNGoyPbGMR36EVtvhw/640?wx_fmt=png&from=appmsg&randomid=mxdc6hrj)

同样可以在单机 8*H100 GPU 服务器上进行测试，其中 8 个 GPU 通过 NVLink + NVSwitch 实现全互联，理论上每个 GPU 最大可以实现 900GB/s 的双向带宽。如下图所示为使用 nccl-tests 测试出的 AllReduce 总线带宽，其中：

- NVSL：NCCL_ALGO 设置为 NVSL，表示使用 NVLink Sharp 算法，这需要 NVSwitch 的硬件能力。
- Ring：NCCL_ALGO 设置为 Ring，表示使用 Ring 算法。
- Tree：NCCL_ALGO 设置为 Tree，表示使用 Tree 算法。
- PCIe：NCCL_P2P_DISABLE 设置为 1，表示不使用 NVLink，这里的带宽比 V100 上明显高很多，主要是在 H100 上使用了 PCIe Gen5。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg7JyDc1w2gg1poumLdYab0A3kicoCr1YIdCPXkicNn0tVmGCejV0avoN2Ih5SUiaMic3o425p8m7ncsQ/640?wx_fmt=png&from=appmsg&randomid=dha9yw97)

如下图为 NVIDIA 官方的结果，在 V100（122），H100 上 NVLink4（360，对应 Ring），H100 上 NVLink4 Sharp（480，对应 NVSL）对应性能与我们实测性能相当：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg7JyDc1w2gg1poumLdYab0Ju63v0fUiaUSzIl6QMBxcSkL9bP5ic8ltO5M0BmkrFfzQq6ZoOreaPJA/640?wx_fmt=png&from=appmsg&randomid=a1qc3lgm)

### 2.6 LLM Tensor Parallelism AllReduce

当前 LLM 推理通常会采用 Tensor Parallelism（TP）模型并行，以便在多个 GPU 上实现较大 LLM 的推理。对于标准的 Transformer Decoder Only 模型，通常会在每个 Transformer Block 中采用如下的 TP 切分方式：

如下图 （a）所示，MLP 层的两个 Linear 层采用先列切（A，Column Parallelism），然后行切（B，Row Parallelism）的方案，这样两个 Linear 之间不用通信：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg7JyDc1w2gg1poumLdYab0X66Aw6nBdd8gYic3mAlwnyjTibv2DOw6nic1xGibRCcC1K5TKyYmBkjkNA/640?wx_fmt=png&from=appmsg&randomid=40keiqnh)

如下图（b）所示，由于每个 Head 的 Attention，Softmax 都可以独立计算，因此可以按照 Head 的方式切分（等价于 Column Parallelism），然后对之后的 Linear 采用行切分（B，Row Parallelism），这样 Self-Attention 中间也不用通信：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg7JyDc1w2gg1poumLdYab0MicRNR69lEQSQZ8GkO0yJrGbuOnica15oowT2hZqZ9V64HsVgRUsfFOA/640?wx_fmt=png&from=appmsg&randomid=6z4wry3w)

如上所述，采用先列切再行切的方式，每个 Transformer Block 中都需要两个 AllReduce 操作，对于一个 40 层的模型则需要至少 80 个 AllReduce 操作。此外，由于 LLM Inference 中通常会采用 Continuous Batching，并且 Prefill 和 Decoding 阶段分开调度，也就导致 AllReduce 操作无法被很好的 Overlap，出现 AllReduce 操作时 GPU 的闲置。因此，可以通过降低 AllReduce 的时延来降低 LLM Inference 的时延，并进一步提升吞吐。

## 三、TensorRT-LLM MultiShot

### 3.1 方案

最近 NVIDIA 在 TensorRT-LLM 中利用 NVSwitch 的 MultiCast 能力对 AllReduce 进行了优化（3x Faster AllReduce with NVSwitch and TensorRT-LLM MultiShot | NVIDIA Technical Blog [4]），可以有效降低 AllReduce 操作的时延（降低时延和增加吞吐是非常不一样的场景，NCCL 中的 AllReduce 更关注吞吐，而 LLM Inference 中的 AllReduce 更希望降低时延）。

TensorRT-LLM 中的 MultiShot 实际上是真正的将 AllReduce 分成 ReduceScatter + AllGather，并且都进行了相应的优化。（PS：NVIDIA 的 Blog 中没有详细介绍这个优化，下面是我们的推测）

如下图所示为 Ring ReduceScatter 的优化，可以等价为一个 All2All 操作实现数据的重排，然后在 Local 进行 Reduce 操作。此过程只有一个 All2All 的整体通信操作，虽然实际上与 Ring 实现的方式的通信量和计算量没有变化，但可以避免 K-1 个 Ring Step 的同步，进而可以有效降低时延。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/zhVlwj96tTg7JyDc1w2gg1poumLdYab0tylf4Vr083icZxaFjcKftX8TwpCq7x5V8ZclC3dFmE4OCUWDSiaUD3Bg/640?wx_fmt=jpeg&from=appmsg&randomid=w85yeijm)

实际上 NVSwitch 的 Sharp 能力也支持在 NVSwitch 上进行 Reduce 操作。如下图所示，每个 GPU 发送全量数据到 NVSwitch，然后在 NVSwitch 上完成 Reduce，最后每个 GPU 都接收聚合后的数据。在这种情况下除了可以避免 Ring Step 的同步外，还可以将通信量（发送/接收）从 2*(K-1)*T 降低为 (K+1)*T。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/zhVlwj96tTg7JyDc1w2gg1poumLdYab0t9PoRsyntmSUyVxjp5wAG8561EUBkXtL8UHe3jlrNL2Q9eSoQ5wmyg/640?wx_fmt=jpeg&from=appmsg&randomid=6ol6o5yx)

当然，上述方式也可进一步不发送当前 GPU 需要的数据，也就是发送 (K-1)*T，NVSwitch Reduce 后，每个 GPU 接收部分结果并在本地加上剩余的部分，最终得到完整结果，这样总的通信量为 K*T。如下图所示:![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/zhVlwj96tTg7JyDc1w2gg1poumLdYab0SlYcnibWB2jHL4RLJnAoZibZtoJ0V4Cmkd1nnKpiayltH1FU5iaXOjlVBA/640?wx_fmt=jpeg&from=appmsg&randomid=fh1e61mu)

如下图所示为 Ring AllGather 的优化，其主要是用了 NVSwitch 的 MultiCast 能力。具体来说，每个 GPU 只用发送 1 份数据，然后 NVSwitch 会将其扩展并发给其他所有 GPU，等价于每个 GPU 都向其他 GPU 发送数据、并从其他 GPU 接收数据。这样有两个好处：

- 相比 Ring AllGather 操作，避免了 K-1 个 Ring Step 的同步，可以降低时延。
- 总体发送数据变为 Ring AllGather 的 1/(K-1)，也就是 T，接收数据量不变，依然为 (K-1)*T；也就是说 Ring AllGather 发送加接收为 (K-1)*T + (K-1)*T=2(K-1)*T；而优化后变为 T+(K-1)*T=K*T。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/zhVlwj96tTg7JyDc1w2gg1poumLdYab0FibiaXLm9BnIibdhibibcn1oFhSfkVMV7zdQJHvQJnY5xKwL8c86bXIz5Hw/640?wx_fmt=jpeg&from=appmsg&randomid=5dff0qpt)

基于以上的两个优化就可以将总的通信量从 Ring 方式的 4*(K-1)*T 降低为 2*T，通信步数也从 2*(K-1) 降低为 2，并且与设备数无关。（PS：这样才能与 NVIDIA Blog 中介绍的数据相匹配）
### 3.2 结果

如下图所示为 NVIDIA 进行的相关测试，LLM 推理中 AllReduce 的通信量往往不大，在通信的 Message 为 4KB - 1MB 时，使用优化后的 MultiShot 方案可以将 AllReduce 的通信时延降低到 1/3 左右。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg7JyDc1w2gg1poumLdYab0wNIX9LdPwX7MQ3dw6k0Qmz2bu0WvpHysPjRTWRGznn4HXHgZMEjhVg/640?wx_fmt=png&from=appmsg&randomid=whzpqwqd)

### 3.3 讨论

需要说明的是，这种方案对于 Message 比较小，并且并行度高的情况下比较有优势，也就比较适合追求极低时延的场景，而这正是 LLM Inference 场景关注的。此外，该功能也需要比较新的硬件比如 H100+NVSwitch，以及比较高的 CUDA 版本来支持，对于非 NVSwitch 互联的方案帮助较少。

此外，对于训练场景，通常 AllReduce 的通信量比较大，并且可以和计算进行 Overlap，因此更关注通信的吞吐，此时该方案的收益并不会特别明显。具体也可以参考 allgather performance using NVLS is poor · Issue #1506 · NVIDIA/nccl · GitHub [5] 中的相关讨论。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg7JyDc1w2gg1poumLdYab0Sue6V3TPy7nrEKsxnVW81DpwBL2jhP2o2anv3nI6Uwb5hmjEjCZicZQ/640?wx_fmt=png&from=appmsg&randomid=ccjkpxan)

## 四、Recogni AllReduce 压缩

### 4.1 摘要

在最近的论文 [2411.09510] Communication Compression for Tensor Parallel LLM Inference [6] 中，Recogni 等作者提出了通过压缩 LLM Inference 中 TP 之间的通信量来降低推理时延。具体来说，作者采用细粒度的量化技术，将选定的激活值压缩 3.5x - 4.5x。所提算法可以在几乎无损的情况下实现首 Token 生成时间（TTFT）最多减少 2x。

PS：需要说明的是，个人认为论文还有不少可以改进的地方，比如说至少提供一下 AllReduce 在整个推理中的占比，然后再对比优化前后的相应时间会更加有说服力。此外，论文中并没有说明为什么 AllReduce 没有采用更常见的 Ring AllReduce，反而采用了具有比较多劣势的 AllGather + Local Reduce 方案。

### 4.2 方案

如下图 Figure 1 所示，其优化的也是 Transformer Block 中的 2 个 AllReduce 操作。不过作者这里不是使用常规的 AllReduce 算法实现，而是通过 AllGather 在每个 GPU 上获取全量数据，然后各自执行 Sum 操作。当然，在 AllGather 之前会各自执行 Encode 压缩操作，AllGather 后每个设备还需要 N-1 次 Decode 操作以反压缩数据（当前 GPU 有对应的非压缩数据，因此每个 GPU 只需要 N-1 次反压缩）。由于 AllGather 传输的是压缩后的数据，因此通信量可以明显降低。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg7JyDc1w2gg1poumLdYab0kxQwVSFubXGwEZq3CRwZx647EIwKKEX2ibwbI0YtyZL36ohiac5ctYxA/640?wx_fmt=png&from=appmsg&randomid=h7g3tlfp)

在之前的章节我们介绍过，Ring AllReduce 可以分解为 ReduceScatter + AllGather 实现，也可以通过 AllGather + Local Reduce 实现。具体来说，首先通过 AllGather 操作每个设备都可拿到全量数据，然后在本地进行 Reduce 操作即可。这种方式实现简单，但是有几个不足：

- 通信量更大，假设每个设备数据大小为 T，则每个设备需要发送/接收 (K-1)*T 的数据，总发送/接收数据量为 (K-1)*K*T。而 ReduceScatter + AllGather 总发送/接收数据量为 2*(K-1)*T。当设备数大于等于 4 时，ReduceScatter + AllGather 的通信量更小。
- 每个设备上都会执行相同的 Reduce 操作，计算量也有所增加。
- 由于 Reduce 前每个设备要存储所有临时数据，因此需要更大的内存消耗。

PS：从上可以看出，使用 AllGather + Local Reduce 并没有特别明显的优势，不确定为什么论文中作者要采用这种方案，作者在论文中也没有具体介绍，也许主要是实现比较简单。

本文的方案中还额外引入了量化和反量化操作，量化不可避免会引入一定的误差，为了平衡量化误差和量化/反量化计算时延，需要找到更均衡的算法。研究发现，OCP 规范（OCP Microscaling Formats (MX) Specification Version 1.0 [7]）中提出的低比特 Block Wise 量化方案可以实现量化误差和压缩时延的平衡。因此，作者本文中主要采用这种方案。

对应的数据类型和配置如下所示：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg7JyDc1w2gg1poumLdYab0wewCmEhFvsMgFTheJsnYZg8c5W7tluVLFibLdGByJzNCWD480Wdaf5A/640?wx_fmt=png&from=appmsg&randomid=cakvmlm1)

### 4.3 实验&结论

测试代码主要来自 IBM 的 GitHub - foundation-model-stack/foundation-model-stack [1]。

此外，作者主要测量了 TTFT，也就是 Prefill 阶段的 Latency，没有测试每 Token 生成时延的影响。

作者首先评估了不同量化数据类型对精度的影响，如下图 Table 1 所示，可以看出 FP3 会对困惑度产生比较大的影响；FP5 还勉强可以接受，在 1% 以内；而 FP4 的影响其实已经比较大了，更不用说在具体的下游任务：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg7JyDc1w2gg1poumLdYab031ZU8xGGf0SSaBq1d9Q8SNpic6yiawOBsq5UA6gbHiaDjGyCVOX9paicaQ/640?wx_fmt=png&from=appmsg&randomid=l0zjurf9)

如下图 Table 3 所示，作者也进一步对比了本方案对 LLM 推理中 TTFT 的影响，可以看出，在 8xL4 和 4xL4 的配置中可以获得 2x 左右加速，在其他配置中反而可能导致降速。（PS：这里使用的输入长度都比较短，如果有一些更长序列的对比会更有说服力，比如 512,1024,2048）

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTg7JyDc1w2gg1poumLdYab0oO0qlI3Y8ToerpGah2AKZbqJ8nEbduibM4UbzT55evmbJibz6diaojUNg/640?wx_fmt=png&from=appmsg&randomid=glbenouh)

## 五、参考链接

1. https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
2. https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/
3. https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
4. https://developer.nvidia.com/blog/3x-faster-allreduce-with-nvswitch-and-tensorrt-llm-multishot/
5. https://github.com/NVIDIA/nccl/issues/1506
6. https://arxiv.org/abs/2411.09510
7. https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
8. https://github.com/foundation-model-stack/foundation-model-stack

