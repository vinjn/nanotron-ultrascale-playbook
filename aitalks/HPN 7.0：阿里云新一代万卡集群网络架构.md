# HPN 7.0：阿里云新一代万卡集群网络架构

**作者：** AI闲谈

---

一、背景

之前的文章中我们中我们具体介绍过万卡 GPU 集群中的网络拓扑相关信息以及在万卡 GPU 集群中进行大规模 LLM 训练面对的挑战和相应解决方案，也进一步介绍了阿里云的集合通信调度框架 C4。本文中，我们简单介绍 C4 底层的阿里云新一代智算集群网络架构 HPN 7.0。阿里在最近的智源大会上也有介绍，可以参考 https://event.baai.ac.cn/live/795，其提到了几个关键词：双上联，双平面，多轨，以及单层千卡，两层万卡。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjQxwBoB4uMEXvNibiciad7znkJYGNwU0tCmS6OuzrjXEFkLREtrOichnlmMd0LOEzkIpOic8bd9BkovicA/640?wx_fmt=png&from=appmsg&randomid=pqhofipb)

上面提到的几个介绍可以参考：
- [万卡 GPU 集群互联：硬件配置和网络设计](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247486775&idx=1&sn=abf7af24181cf5189e113fb161cc8d30&chksm=c364ca72f4134364f4e3fa4a971f767c2b07e6c2cae38c2a4ae28071fd330abaea68c36542c4&scene=21#wechat_redirect)
- [万卡 GPU 集群实战：探索 LLM 预训练的挑战](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247486852&idx=1&sn=9f9dc1df99ab6aafb28e091f4532b89e&chksm=c364cac1f41343d7b10d9d234d1c7f3371d996afda01cb94d294a38cba4f1a14fe4594992aa2&scene=21#wechat_redirect)
- [阿里 C4：通信驱动加速大规模并行训练效率](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247487014&idx=1&sn=c49df9bd2de03acfae39bf4dce1c84b6&chksm=c364c963f4134075edee235c744c68c3f411ac7cdd1b9847de9333169292ff375a56c7d8ebd0&scene=21#wechat_redirect)
- [剖析大规模 GPU 集群：针对 LLM 场景的挑战和优化](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247487054&idx=1&sn=fd540ee08fc40211d51856a146d22ac8&chksm=c364c90bf413401dc34fb9944f511a2960d4c532ea9bd8e4f88c696a5a7a6c58e549c73a8e27&scene=21#wechat_redirect)

## 二、拓扑

如下图所示（图片来自 [星融元针对LLM大模型承载网发布星智AI网络解决方案](https://mp.weixin.qq.com/s?__biz=MzU2NjQ1OTE3Mw==&mid=2247523136&idx=1&sn=9fdb15f0e098cca63cc4ba71ac838041&scene=21#wechat_redirect)）为常见的三层无阻塞 Fat-Tree 拓扑（SuperSpine-Spine-Leaf），可以将两层的 Spine-Leaf 看做一个 Pod，可以将一层 Leaf 看成一个 Group，一个 Pod 里有 8 个 Group（HB）。其中：

- Leaf Switch 有 128 个 400 Gbps 的 Port（交换带宽 51.2Tbps），每台机器 8 个 H100/H800，每个 GPU 对应一个 400 Gbps NIC，full mesh 连接一个 Group 里最多 8 个 Leaf Switch。
- 每个 Leaf Switch 有 64 个下行 400Gbps Port，能连接 64 台机器，也就是 1 个 Group 最多只能 64*8=512 GPU。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjQxwBoB4uMEXvNibiciad7znkR7rQ4a638X3tIZWsVxJwJChfYtS0Yw9a81QMunq8sc4WE1Oyp5JkCQ/640?wx_fmt=png&from=appmsg&randomid=x85bzg9f)

如下图所示（图片来自 Revolutionizing Data Center Networks: Alibaba’s SONiC Journey）为阿里云 HPN-7.0 的拓扑。可以看出，其 1 个 Pod 里依然有 8 个 Group（Segment），不过其 1 个 Group 里有 128 个 8 GPU 节点，而不是 64 个 8 GPU 节点。

- Leaf Switch 有 64 个 400Gbps 的上行 Port，128 个 200Gbps 的下行 Port（交换带宽 51.2Tbps）。每台机器 16 个 200Gbps NIC Port，full mesh 连接可以对应 16 个 Leaf Switch。
- 每个 Leaf Switch 有 128 个下行 200Gbps Port，能连接 128 台机器，也就是 1 个 Group 可以支持 128*8=1024 GPU。（单层千卡）
- 对于下图的拓扑中：
- 每个 Pod 里有 8 个 Group（Segment），也就是每个 Pod 有 8192 GPU（两层万卡）。
- 总共有 128 个 Pod，也就是可以支持 1,048,576 个 GPU（三层 10 万）。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjQxwBoB4uMEXvNibiciad7znk419oBf3U4U7F7jNK9ej12StkupakZeZmOIuRMWQG9xu2uKmKAk5rWg/640?wx_fmt=png&from=appmsg&randomid=2ptsuqt7)

PS：由上述的拓扑图可以知道，一个 Group 里的 GPU 之间的通信只用经过一次通信（只用经过 1 个 Leaf Switch）。在传统的拓扑中，1 个 Group 内最多 512 GPU 互联，总的通信带宽为 512*400Gbps=204.8Tbps；在 HPN-7.0 中，最多可以支持 1024 GPU 互联，总的通信带宽为 1024*2*200Gbps=409.6Tbps，增加了一倍。

## 三、双上联

采用双 200Gbps Port 以及双 200Gbps NIC 除了在 1 个 Group 里增加一倍的 GPU 数量及通信带宽外，另一个主要的优势是可以缓解网卡、交换机、光模块、光纤等导致的异常。现在常见的大规模 AI 模型训练基本都是同步方式，这些异常都会导致任务中断，影响训练进度，浪费计算资源。

作者将这种方式称为双上联，具体来说有如下两种方案，都可以实现一个 GPU 对应两个上行链路，并且两个上行链路连接到不同的交换机，也就是 128 台机器的 Port-00 连接到 Leaf Switch-00，Port-15 连接到 Leaf Switch-15：

- 1 个 400Gbps NIC，NIC 上 2 个 200Gbps Port：此时如果网卡故障，依然会导致训练任务中断。不过一个网卡可能更方便管理，占用的插槽也可能更少。（对应阿里 Paper：[2406.04594] Boosting Large-scale Parallel Training Efficiency with C4: A Communication-Driven Approach）
- 2 个 200 Gbps NIC，每个 NIC 上 1 个 200 Gbps Port：如果一个网卡故障，另一个网卡依然可以工作，容错性更强。但可能增加管理成本以及占用更多的插槽。（对应阿里官网介绍：灵骏可预期网络：Built for AI Infrastructure-阿里云开发者社区）

在双上联中，某一个上行链路故障或对应交换机故障时，流量可以切换到另一个 Port 提供服务（如下图绿线），并不会导致训练任务中断，只是有可能影响训练速度。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjQxwBoB4uMEXvNibiciad7znkxkO219STxnnD7gMX2nxlRgRsc0ia4OGpL3Qfdviam7zx2vEQQf35RZ2A/640?wx_fmt=png&from=appmsg&randomid=x5wns8hp)

## 四、双平面

双上联的方案有助于提高系统的可靠性，然而其同时也会加剧 ECMP（Equal-Cost Multi-Path）哈希不均的可能性。如下图所示，蓝色为发送端，橙色为接收端，发送端可以控制两个上行通路尽量均匀发送，但是由于 Spine 的存在，就很难保证 Spine 再到右侧 Leaf 的流量是均匀的。尤其是训练场景，其通常是流量数目少，但每次流量的数据比较大，会进一步加剧这种流量极度不均的现象，也即哈希极化。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjQxwBoB4uMEXvNibiciad7znkcqibKh3sMnaGnz9olmichMJvvkT9qj8C7Kh6aNv5KYMHw1pmfGnJ4YQg/640?wx_fmt=png&from=appmsg&randomid=g8kv07ig)

为了解决哈希极化问题，HPN 中在网络拓扑中实现了双平面设计。具体来说，每个 GPU 都对应 2 个 NIC Port，那么就可以将所有 GPU 对应的 NIC Port-0 构建一个网络平面，所有 NIC Port-1 构建一个网络平面，两个网络平面的网络拓扑完全一样，并且没有任何交叉。这样的话，只要发送端保证发送到两个 NIC Port 的流量是均匀的，那么在接收端就会一定接收到均匀的流量，大幅降低哈希极化的概率。如下图所示，平面 1 和平面 2 是完全镜像的，蓝色 GPU 对应的流量会均匀发送到两个平面，并且到每个平面的流量只会在内部转发，如 1,2,3,4 的路径，最终到达橙色 GPU 对应 NIC 的流量也是均匀的。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjQxwBoB4uMEXvNibiciad7znkBSnv0Z9n41bOc0HibUa3OIqAb0jzt8XKLgWuvvomEia2KdQpVubpqh0A/640?wx_fmt=png&from=appmsg&randomid=8syn1j3x)

更清晰的视角可以参考下图：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjQxwBoB4uMEXvNibiciad7znkFcOBTFoYdncSaCjxHVknC5p7OibMtNrKxk1muibUIAOR7jwFg0zdMlZw/640?wx_fmt=png&from=appmsg&randomid=ylnmran6)

## 五、多轨通信

多轨通信其实就是综合考虑多种通信链路，以实现最优通信效率。比如单节点内部通常有 NVLink 和 NVSwitch 实现全互联，比如对于 8*H100 SXM 节点，可以实现 7.2TBps 的通信能力。而节点间可以通过高速网络互联，一个 Group 的 128 个节点通过一次网络转发即可以连接。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjQxwBoB4uMEXvNibiciad7znkiapR1o3YormKU3kpyia75QRorMrjZp9C1vZQNXj2XdQepVV359qFpa5w/640?wx_fmt=png&from=appmsg&randomid=azsvuzm5)

当然，阿里云也期望未来能在更大范围（超过 8 GPU ）内实现更高性能的互联，类似 NVIDIA 最新的 NVL72 和 SuperPod 576，也就是下图中的 Scale-Up 内部互联（AI Rack）：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTjQxwBoB4uMEXvNibiciad7znkNDCwncrzNZBc25LjSU4NoxdbRTy5icfHMt501ZkYCBiabPJKGtgWPBCA/640?wx_fmt=png&from=appmsg&randomid=oybzq5ay)

## 六、参考链接

1. [https://mp.weixin.qq.com/s/Uxixl_43_poc8lgiA3Gvsg](https://mp.weixin.qq.com/s?__biz=MzU2NjQ1OTE3Mw==&mid=2247523136&idx=1&sn=9fdb15f0e098cca63cc4ba71ac838041&scene=21#wechat_redirect)
2. https://sonicfoundation.dev/revolutionizing-data-center-networks-alibabas-sonic-journey/
3. https://arxiv.org/abs/2406.04594
4. https://developer.aliyun.com/article/1252706
5. https://event.baai.ac.cn/live/795

