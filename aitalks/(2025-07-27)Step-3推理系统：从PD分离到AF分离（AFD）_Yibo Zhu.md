# Step-3推理系统：从PD分离到AF分离（AFD）

**Author:** Yibo Zhu

**Date:** 2025-07-27

**Link:** https://zhuanlan.zhihu.com/p/1932920900203807997

之前在pyq和业内朋友吹牛说会写一下最近的一些工作。其实技术细节在[Step-3系统](https://zhida.zhihu.com/search?content_id=260897020&content_type=Article&match_order=1&q=Step-3%E7%B3%BB%E7%BB%9F&zhida_source=entity)方向的tech report（[https://github.com/stepfun-ai/Step3/blob/main/Step3-Sys-Tech-Report.pdf](https://link.zhihu.com/?target=https%3A//github.com/stepfun-ai/Step3/blob/main/Step3-Sys-Tech-Report.pdf)）已经写了很多了，不过tech report终归还是得尽量严谨，客观，为了不over-claim叠了无数的甲。这里就再碎碎念一下tech report不会写的high-level思考，当然就是不严谨，有主观成分的了——

## 系统篇

太阳底下没有新鲜事，尤其是只考虑system自身时。在冯诺依曼架构，搞来搞去就三件事，计算存储通信。在一个新应用场景（比如AI），初期先单个方向地搞；分别搞差不多了，开始考虑合起来的事，怎么调度，怎么让各种资源并行利用起来；再到后来，就更加精细化的操作，把计算性质有本质不同的模块拆开来，分治，中间通信开销用pipeline掩盖。这也是现在的阶段。

如果你现在认可[PD分离](https://zhida.zhihu.com/search?content_id=260897020&content_type=Article&match_order=1&q=PD%E5%88%86%E7%A6%BB&zhida_source=entity)是很有价值，是早晚要做的，那你就无法拒绝[AF分离](https://zhida.zhihu.com/search?content_id=260897020&content_type=Article&match_order=1&q=AF%E5%88%86%E7%A6%BB&zhida_source=entity)的方向。

这和我们23年底写[DistServe](https://zhida.zhihu.com/search?content_id=260897020&content_type=Article&match_order=1&q=DistServe&zhida_source=entity)（一篇早期PD分离的paper, [https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf](https://link.zhihu.com/?target=https%3A//www.usenix.org/system/files/osdi24-zhong-yinmin.pdf)）时一样，知道会有人抗拒，说chunked prefill也能解决prefill打断decode的问题啊。当年写论文叠甲都是在考虑怎么让chunked prefill派的reviewer接受。于是会编说法，哎，chunked prefill可以和PD分离一起使用啊，他们不互斥啊。。

但其实我内心里知道这些安抚reviewer的话是没什么实际意义的。未来主流一定是PD分离的，尤其是有条件追求极致的场景。计算性质本质不同的东西，在通信能被掩盖的前提下，拆开优化相对于混一起做肯定是会有本质性能/成本优势的。

AF分离也一样。现在，为了让纯EP（特指是AF不分离混在一起类似[DeepEP](https://zhida.zhihu.com/search?content_id=260897020&content_type=Article&match_order=1&q=DeepEP&zhida_source=entity)那样的EP）余党好接受点，我也有个写paper常见说法：纯EP是AF分离的一个（A和F必须是同卡型，同数量，且部署在一起的）特例，AF分离推广了纯EP。

问：AF分离通信量问题咋解决

答：纯EP有一样的问题。通信总量其实都是一样的。而AF分离后要求的部署规模更小，因此面对网络拥塞等问题还有优势，也更容易fit进某些规模有限的超节点。

问：AF分离TPOT如何压低

答：纯EP有一样的问题。同时，AF分离后可以更有针对性地scale A或F来把延迟压到目标值。

问：AF分离，A做DP的负载均衡和F那头的expert均衡性咋做

答：纯EP有一样的问题。而AF分离后要求的部署规模更小，这些问题都会容易一些。而且还有混TP的选项（F数量不超过MoE的topK时，TP都不亏通信量）

问：AF分离，变长context是否会影响attn效率

答：纯EP有一样的问题。而AF分离后，A还可以自由地scale适应变化，还可以结合按context长度分桶的策略，只要设置一个总KV cache的阈值保证不超时就可以。

问：AF分离后，通信与PD分离的通信是否有冲突

答：纯EP有一样的问题。而且，EP的all-to-all要每个节点又发又收。而AF分离后，相当于F节点把一半的网络通信量带走了。每轮通信A要么只发，要么只收。只发的时候，收方向的带宽不就可以用来收P传过来的KV了吗

……反正不管什么问题，以“纯EP有一样的问题”为开头，“AFD反而更容易解决这个问题”为结尾就对了，包管答得滴水不漏。

但是我内心知道，以上全是废话。未来必定是AF分离为主流的。太阳底下没有新鲜事。

系统篇最后，我还是要给一些人credit，包括字节老同事的megascale-infer，写出paper的手确实快。我也看过清华两个组考虑过AF拆开的事。我们在阶跃24年想过，但没有做。我们这回是下定决心冲一把实现，证明在大尺寸模型和常见SLA约束下，确实也能做到胜过纯EP。这勇气也有DeepSeek的功劳的——正如上面的逻辑，DS证明了纯EP都能把这个尺寸模型跑起来并优化到符合SLA的水平，AF分离这个处处严格更优的没道理优化不到。

  

## 模型篇

说完和PD分离在system思想脉络上的相似，其实AF分离的意义比PD分离还是要多些的。最主要的就是对模型结构设计有梳理思路的作用。

系统篇上来就说system自身无新鲜事。但是现在做ai system还是蛮有意思的，意思就在于模型和硬件的新鲜事，而且是system可以影响他们而不只是被动接受。

AF分离后，分治A和F对分析模型成本带来的便利，我相信tech report里已经体现蛮多了。

对F的分析就是很好的例子。对总参数，稀疏度，过去的认知过于不科学。比如有人说，总参数还是会影响decode效率的，毕竟访存大了；有人说，还是激活量影响decode效率，我们搞大规模EP不就把batch攒上去了吗，就变成计算bound了。俩人一碰，分析一通，最后肯定会发现分歧就是给定EP部署规模后，attention那边context长度会影响系统batch大小，从而影响到ffn的MFU情况。那实际使用时context到底是多长呢？说不准的。同一个基模，reasoning和非reasoning形态就很不一样啊。于是就得不出一个普适的结论了，模型该怎么设计。

而AF分离后，排除A那头的影响，F的分析就变得非常单纯，tech report里我们也分析出了实际瓶颈就是网络带宽，和context length都无关。本来F就应该和context length无关啊。Step-3的稀疏度就是一个相对于dense 38B完全不亏系统性能，但是再稀疏一些就有可能要开始亏的程度。不过我们开训的时候通信库研发还没影呢，能跑到line rate的百分之多少不知道，于是稀疏度就稍微保守了一些。但我一直认为不够稀疏是小问题，大不了后期upcycle呗，所以开始的时候保守点没大错。

弄明白[MoE sparsity](https://zhida.zhihu.com/search?content_id=260897020&content_type=Article&match_order=1&q=MoE+sparsity&zhida_source=entity)的系统问题后，假如说要基于DSv3的结构魔改去加强FFN，应该优先探索的路是增加激活expert数，而不是去扩总expert数。前者有可能是不亏系统性能的，而后者可是因为over-sparse明确会亏的。over-sparse后，画个loss vs flops的图已经失去意义了，因为flops并不反应实际系统性能，真跑起来推理训练MFU都会非常难看。真正要看的图是loss vs 系统成本，这才叫model-system co-design。当然搞噱头的话是另一回事了。

所以说AF分离是对模型设计有指导性意义的东西，这也是做system工作但获得beyond system意义的有意思的地方。比如再说说未来可能的模型架构调整方向：

既然我们知道想更稀疏但又不亏系统性能的大敌是网络通信，而网络通信量正比于batch size x hidden dim，那么问题来了：如果我在MoE FFN前后先做一个low rank映射把hidden dim降到一半再通信，获得通信量减半的好处。以此换来MoE稀疏度可以翻倍却不伤系统性能（使用两倍总batch size加大计算密度填补MoE稀疏度问题，因low rank通信所以总通信量不变），这在算法上是正收益还是负收益呢？反正现在模型的一个细粒度expert的intermediate size就那点大，7168的hidden feature直接进expert也是被疯狂压缩。用一个low rank映射换个FFN参数量翻倍而不亏推理性能，模型效果是赚还是赔呢？

类似的想法还可以有很多很多。。这些是只做模型，或者只做系统的人看不到的东西。必须是真正地从co-design角度想问题。

## 硬件篇

System是模型和硬件的桥梁。它能影响模型设计，也能影响硬件设计。

我就碰到过做硬件的人说，DSv3这个模型让大家搞all-to-all，然后他问，是不是未来模型都会依赖all-to-all啊？硬件设计（尤其是互联）是不是一定特别要为all-to-all服务啊？

我觉得这个说法搞错了。不是DSv3这个模型导致的all-to-all，是DeepEP这个纯system层面的部署方法导致的。我也可以拿AF分离跑DSv3啊，这样就不是all-to-all了。在F数量不多（尤其是机器数量小于MoE topK时）就是all-gather（for dispatch）和reduce-scatter（for combine）了。只不过是变种的all-gather和reduce-scatter，因为收发节点不在一起。

那这个时候，互联还要盯着all-to-all支持吗？是不是ring也够用了？支持all-to-all要求互联支持全交换，要高效中间得摆个电交换机，最高交换容量的电交换芯片得进口，有的美国还不卖给你（nvl72）。支持ring则不需要电交换机，光纤拉一圈，理论有无限带宽（当然数据进芯片时还是得转回电，这个带宽是有限的，瓶颈是芯片本身）。这倒比较像TPUv4的互联了。说到这其实是在为我们InfininHBD的工作（SIGCOMM'25, [https://arxiv.org/pdf/2502.03885v3](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2502.03885v3) ）带货了，全数据中心灵活组ring的HBD（High Bandwidth Domain）。。助力一下国产卡。

M<->N这种二分图的通信pattern，其本质的程度一点不低于针对同构的collective原语。从早年的parameter server，到PD分离，到AF分离，全是类似的通信pattern。我们即将开源的StepMesh还是从早年用于parameter server的组件改出来的。Why？系统里分出了两个角色，这两个角色要通信呗。再往前还有mapreduce的mapper和reducer..

而且我还是那个看法，当两个角色计算性质足够不一样的时候，追求极致就得拆开优化。所以要我说，在硬件层面，互联方案倒应该好好考虑怎么支持这种通信...

同时我也能看到思路的分歧，分布式系统的人和HPC的人，第一想法还是会不一样的。我属于前者，会更多地想怎么支持灵活多变的实际场景，能不能动态scale，可靠性怎么办，对异构硬件有更开放的态度。HPC的人，主要还是极致的同构并行，灵活scale和可靠性往后排排。这也不止在系统层面（AFD vs DeepEP），也反映到硬件理念上——英伟达这种服务HPC出身的硬件厂家，就喜欢谈scale-up。而我对scale-up这个词很反感，因为会引起我对大型机被data center送终的记忆。我宁可专注high bandwidth domain。除了高带宽外，我还要它能灵活配置，要它能容灾，乃至最后追求HBD也能scale-out。我相信到了大规模系统设计阶段，HPC之路是快但相对短视的，最终还是会演变为分布式系统之路接手（100%主观，毫无客观成分，虽然我有历史证据，但不代表未来）。

哎呀，硬件篇似乎都在说互联和分布式系统，原谅我做AI infra前的背景吧，有点职业病。

其实AF分离对计算芯片设计也是有很多指导意义的，tech report里有对非旗舰硬件的适应性分析。step-3这样的模型和AF分离，会非常利好[chiplet design](https://zhida.zhihu.com/search?content_id=260897020&content_type=Article&match_order=1&q=chiplet+design&zhida_source=entity)——你每个chiplet只要有L20水平就行，足够用DP去scale attention。chiplet便宜，国产制程好做，意义重大。

计算和显存带宽之比，其实倒没必要特别追求极端。比如超大带宽配小算力，没必要，尤其是服务器侧高吞吐的场景——如果你今天还粗暴认为decode只会是访存bound的东西，你的认知出了大问题。roofline形状说不定反倒是平庸点好。应该是模型来适应你硬件，尤其是attention的计算密度（arithmetic intensity）的调整，而不是你去适应模型。你适应不过来的——开源模型里有attention计算密度512的，也有计算密度32的，你咋适应？至于FFN，也不管你硬件的roofline长啥样，AF分离后反正都能凑到合理的batch size。哦，只要你有足够的互联（又绕回来了

## 结语

我们招人。想要追求model-system-hardware co-design的，想要窥见深刻洞察的，可以直接私信我你的基本信息。