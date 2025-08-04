# 新兴 Python 算子开发：Triton、CuTeDSL、MOJO 🔥等概览

**Author:** BobHuang

**Date:** 2025-06-29

**Link:** https://zhuanlan.zhihu.com/p/1919816304271028292

​

目录

收起

一、triton-lang/triton 15.9k

1、介绍

2、扩展相关

3、生态相关

二、pytorch-labs/helion 0.16k

三、NVIDIA/cutlass(CuTeDSL) 7.7k

四、tile-ai/tilelang 1.3k

五、apache/tvm 12.4k

六、modular/modular(MOJO) 24.3k

七、halide/Halide 6.1k

八、Tiramisu-Compiler/tiraisu 0.94k

九、NVIDIA的cuTile

十、pytorch-labs/tritonbench性能对比

1、flash\_attention

2、gemm

3、fp8gemm

4、int4\_gemm

5、layer\_norm

6、softmax

7、Triton launch\_latency

附录

**6.23更** 关注性能的朋友有福了，朋友为我推荐了[pytorch-labs/tritonbench](https://link.zhihu.com/?target=https%3A//github.com/pytorch-labs/tritonbench)这个项目，我加更在最后做下Triton、tk、tilelang的flash attention性能对比，顺便做了些gemm等Triton bench。

我最近3个月都在研究Python AI 算子 DSL，在此记录下我的一些想法。目前还在学习中，若理解有偏差，烦请指正。

DSL 即 Domain Specific Language，是指为特定领域（domain）设计的专用语言，广为人知的包含 HTML、SQL和正则表达式。本文讨论的内容更准确的名词是`eDSL`，e 即`embedded`，表示复用Python语法，使用编译器来改变代码运行的方式。

AI模型的开发通常在Python上进行，并运行在GPGPU上。但是Python是不能运行在GPU上的，为了方便研究人员，OpenAI构建了`Triton`。`Triton`非常`Pythonic`，用户不需要熟练硬件架构和CUDA，就能方便得写出高性能代码。Python DSL能否在极致性能和可用性两全其美？这大概是需要奋斗且不太好达到的目标。Python DSL是不是绕道而行，有可能改变现有CUDA生态吗？目前看已经让CUDA拥抱Python了，[CUDA: New Features and Beyond](https://link.zhihu.com/?target=https%3A//www.nvidia.com/en-us/on-demand/session/gtc25-s72383)，[Nvidia](https://zhida.zhihu.com/search?content_id=259380445&content_type=Article&match_order=1&q=Nvidia&zhida_source=entity)更是宣布了要用CuTile解决以前库太多的问题。最终结果要交给时间检验，其本质还是**tradeoff**，No Silver Bullet。

### 一、[triton-lang/triton 15.9k](https://link.zhihu.com/?target=https%3A//github.com/triton-lang/triton)

AI的Kernel运算非常规整，往往做下tile就能拿到性能。所以Triton的设计就是牺牲部分通用性换来DSL的简洁，Triton不用关心线程组织，只需要关心 tile 和 核心部分hardcode float的配置。Triton还有一个重要议题就是支持Nvidia显卡最新的feature，编程模型在改变，DSL因为抽象更高级，拿到甜点性能还是非常方便的。

[](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV11tMwznEmo)

### 1、介绍

Triton作者[Tillet](https://link.zhihu.com/?target=https%3A//github.com/ptillet)关于其设计的论文发表在[MAPL2019](https://link.zhihu.com/?target=https%3A//www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)，其设计了多层 tiling、自动优化等核心特性，希望通过 类C语言的DSL + 编译器 支持 tile 编程。之后用MLIR重构了，并使用了Python做为前端语言就一发不可收拾了，2023年3月[Pytorch2.0](https://zhida.zhihu.com/search?content_id=259380445&content_type=Article&match_order=1&q=Pytorch2.0&zhida_source=entity)的发布为我们带来了Triton的[Inductor](https://zhida.zhihu.com/search?content_id=259380445&content_type=Article&match_order=1&q=Inductor&zhida_source=entity)的接入。

在Triton代码的编写中我们更关心一个Block，用户不需要感知shared memory。Triton借助Layout设计以及Pass优化，能够减轻用户写kernel的负担，也能保证一定的性能，关于Triton和CUDA的对比如下图所示，来源[Pytorch2023会议](https://link.zhihu.com/?target=https%3A//static.sched.com/hosted_files/pytorch2023/2c/Triton_compiler.pdf)

  

![](https://pic1.zhimg.com/v2-e3881beb71a1dd510055bd41b93126fc_1440w.jpg)

  

随着[FlagTree](https://link.zhihu.com/?target=https%3A//github.com/FlagTree/flagtree)的开源，目前Triton有[nvidia](https://link.zhihu.com/?target=https%3A//github.com/triton-lang/triton/tree/main/third_party/nvidia)、[amd](https://link.zhihu.com/?target=https%3A//github.com/triton-lang/triton/tree/main/third_party/amd)、[intel](https://link.zhihu.com/?target=https%3A//github.com/intel/intel-xpu-backend-for-triton/tree/main/third_party/intel)、[cpu](https://link.zhihu.com/?target=https%3A//github.com/triton-lang/triton-cpu)、[华为昇腾](https://link.zhihu.com/?target=https%3A//gitee.com/ascend/triton-ascend/tree/master/ascend)、[摩尔线程](https://link.zhihu.com/?target=https%3A//github.com/FlagTree/flagtree/tree/main/third_party/mthreads)、[沐曦](https://link.zhihu.com/?target=https%3A//github.com/FlagTree/flagtree/tree/main/third_party/metax)、[昆仑芯](https://link.zhihu.com/?target=https%3A//github.com/FlagTree/flagtree/tree/main/third_party/xpu)、[ARM china](https://link.zhihu.com/?target=https%3A//github.com/FlagTreeZhouyi/flagtree-zhouyi/tree/master/third_party/aipu)、[清微智能](https://link.zhihu.com/?target=https%3A//github.com/FlagTree/flagtree/tree/triton_v3.3.x/third_party/tsingmicro)、[天数智芯](https://link.zhihu.com/?target=https%3A//github.com/FlagTree/flagtree/tree/main/third_party/iluvatar)、[寒武纪(部分)](https://link.zhihu.com/?target=https%3A//github.com/Cambricon/triton-linalg) 共12个开源后端，其他公司也有做，但是没开源。

Triton能很轻松得写出性能不错的kernel，在矩阵乘的kernel上你能很轻松得用上tma，对比native的CUDA kernel，可以在B200上获得_近5倍_的加速。[matmul.cu](https://link.zhihu.com/?target=https%3A//github.com/OpenMLIR/LeetGPU/blob/52cb480f4427ab7c38e715850656ca57b05fde01/02-matrix-multiplication/CUDA/native.cu) vs [matmul-with-tma.py](https://link.zhihu.com/?target=https%3A//github.com/OpenMLIR/LeetGPU/blob/52cb480f4427ab7c38e715850656ca57b05fde01/02-matrix-multiplication/Triton/use_tma.py) **2025.6.29更 不止5倍，**这里搞了个大乌龙，因为input\_precision="ieee" 还是**fma**，这是fma**被展开**的性能。

![](https://pic3.zhimg.com/v2-3772e1555a026a1d5ca1286a6d02bab4_1440w.jpg)

  

我也在尝试做一个Triton的开源OpenCL后端，为想要接入Triton的公司提供样本。有兴趣可以关注[OpenMLIR/triton-spirv](https://link.zhihu.com/?target=https%3A//github.com/OpenMLIR/triton-spirv)。

[](https://www.zhihu.com/column/c_1906884474676945862)

我的另一篇文章更详细介绍了Triton的执行流程，有兴趣可以阅读。

[](https://zhuanlan.zhihu.com/p/712640431)

现在DSL大战一触即发，打响Triton保卫战刻不容缓，Triton upstream 在做kernel的bench。

[](https://link.zhihu.com/?target=https%3A//github.com/triton-lang/triton/tree/main/python/triton_kernels)

### 2、扩展相关

[ByteDance-Seed/Triton-distributed](https://link.zhihu.com/?target=https%3A//github.com/ByteDance-Seed/Triton-distributed) Seed对Triton做了扩展来支持通信，大模型时代通信计算融合是现在一个非常重要且具有挑战的议题，思泽提出了在Triton上添加通信Op，并做了实现。目前通算融合更多还是在框架层面用计算图做的，但是[MegaKernel](https://link.zhihu.com/?target=https%3A//github.com/mirage-project/mirage/tree/mpk)、[FlashDMoE](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2506.04667)一经推出都反响很大。

[](https://zhuanlan.zhihu.com/p/1900910901017679250)

Triton在甜点性能拿到后，后续优化和硬件是强相关的，开发者为了性能必须要去开发Triton，这个难度不小。[facebookexperimental/triton](https://link.zhihu.com/?target=https%3A//github.com/facebookexperimental/triton/tree/tlx) meta在搞TLX (Triton Low-level Language Extensions)，把 warp-aware, hardware-near 带回Triton，以求拿到性能。把 Low-level 带回Triton也是有收益的，能拿到性能，缺点就是Triton也要变成NVIDIA的形状。

NPU/DSA的粒度相比GPGPU要更粗，造一整套工具链轮子和打磨多年CUDA的竞争是非常难的，也可以直接在Triton上做适配。[microsoft/triton-shared](https://link.zhihu.com/?target=https%3A//github.com/microsoft/triton-shared) 最先对lower到linalg这个层级的dialect做了探索，拿到了不错的效果，后续的很多项目都基于此做了实现。另外当前Triton的Op定义对于这种粗粒度的硬件是**远远不够**的，[python/triton/language/standard.py](https://link.zhihu.com/?target=https%3A//github.com/triton-lang/triton/blob/main/python/triton/language/standard.py) 文件可以看到有些函数如sigmoid是直接用的数学实现，这些在NPU/DSA往往有自己的lower路径，另外硬件可能提供了更多的函数抽象需要在Triton这边扩展。抽象`High-level Op`比提供`Low-level op`影响相对小一点，毕竟NPU/DSA声量也不够大，基本还在手搓算子甚至是IR。这会不会又让Triton成DSA的形状呢，总之做一个公平的标准很难，OpenCL之死值得警惕。

目前Triton主线是不太关心这些的，他们非常严格得控制着自己的编程模型。这一点有点像语言委员会，或许他们的心目中理想硬件就应该是他们坚守的编程模型模样。

### 3、生态相关

有了一个可视化与分析工具[pytorch-labs/tritonparse](https://link.zhihu.com/?target=https%3A//github.com/pytorch-labs/tritonparse)。我对GPU的调试全靠print，Triton的print并不好用，去Debug IR是一件常见的事情，感谢作者。

[](https://zhuanlan.zhihu.com/p/1917933418014016114)

还有人在整活[Mogball/triton\_lite](https://link.zhihu.com/?target=https%3A//github.com/Mogball/triton_lite)，Triton 风格接口的MOJO，还提供了一个在torch.compile来把Triton替换为MOJO。我觉得项目想要达成的目标就是进一步细化粒度，想推Triton前端的统一化，替换掉它的编译器后端。

Triton更多的生态体现在诸如Pytorch、vllm、sglang、flash-attention等对于Triton的接入，项目已经形成了一定影响力。我们还可以看到[srush/Triton-Puzzles](https://link.zhihu.com/?target=https%3A//github.com/srush/Triton-Puzzles) 这样非常精美的Triton教程，甚至还有了Triton培训班。

### 二、[pytorch-labs/helion 0.16k](https://link.zhihu.com/?target=https%3A//github.com/pytorch-labs/helion)

helion是一个面向Tensor的DSL，比Triton的抽象层级更高。在这一级想做出性能是非常难的，但是他们将kernel编译到了Triton，直接拿Triton的性能。挺有意思的，如果说一款新的芯片为了生态完全可以借鉴这个思路，abstract is all you need。

### 三、[NVIDIA/cutlass(CuTeDSL) 7.7k](https://link.zhihu.com/?target=https%3A//github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL)

Nvidia看到Triton的成功还是比较眼红的，很快就开始反击了，Nvidia作为一家成熟的商业公司估计在杨军老师去给Triton做支持就有想法了。

CuTeDSL和CUDA类似，是thread级别，以CuTe抽象为中心。改用MLIR后带来的首要收益是编译速度的显著提升，当前还有Pytorch的集成。

1.  支持DLPack接口，我可以直接用Pytorch申请的tensor，直接check答案。当然其他AI框架也可以，零拷贝、跨框架的数据互操作的收益很大。  
    
2.  将静态layout转换为动态layout，通过mark\_layout\_dynamic来避免JIT functions的重复编译。  
    
3.  直接集成到AI模型中，你可以把你的算子直接替换进去，这也是Pythonic带来的收益。这也是为什么Pytorch、vllm、sglang都集成了Triton的原因，无感接入的感觉很爽。  
    

Python还能为用户带来什么呢，Nvidia不得不暴露一些interface出来，[python/CuTeDSL](https://link.zhihu.com/?target=https%3A//github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL) 安装Python包后在`site-packages`文件也可以看到。

我也将在下面的文章继续对其进行持续探索。

[](https://zhuanlan.zhihu.com/p/1918927108006188667)

### 四、[tile-ai/tilelang 1.3k](https://link.zhihu.com/?target=https%3A//github.com/tile-ai/tilelang)

基于[TVM](https://zhida.zhihu.com/search?content_id=259380445&content_type=Article&match_order=1&q=TVM&zhida_source=entity)的thread级别的primitives(原语)，有如下三种编程接口。

![](https://pic4.zhimg.com/v2-cecdc6046001f4e8266135b2ec817e3b_1440w.jpg)

  

能显式声明内存，能显式控制线程了。当然你也可以选择不控制，对于大多数用户我认为就在`Developer`，可能[meta Triton tlx](https://link.zhihu.com/?target=https%3A//github.com/facebookexperimental/triton/tree/tlx)会达到和这边差不多的效果。这个设计理念是非常好的，不仅支持3种语法，且这三种语法可以出现在同一个program中。

[](https://zhuanlan.zhihu.com/p/20718641070)

我们不容忽视的是tilelang在推理部署的实力，好搓性能又好。我感觉`CuTeDSL`很快就会和其竞争起来，因为既然想要性能那肯定要追求到底，看双方的算子大师们的进度和实际性能了。当然也可以像helion接Triton那样，tilelang接CuteDSL就好了，abstract is all you need，打不过就加入。

### 五、[apache/tvm 12.4k](https://link.zhihu.com/?target=https%3A//github.com/apache/tvm)

TVM 是一个非常完善的深度学习框架，且提供了DSL的算子书写。近几年热度在减退，Pytorch更好用已成为事实标准AI框架。

TVM中的Tensor Expression、TensorIR都是适合写算子的，Relax主要描述计算图。TileLang实际上是TensorIR 的用户层 DSL 抽象，

TE(Tensor Expression)提供了丰富的并行抽象：可以使用s.bind(axis, te.thread\_axis("blockIdx.x"))、("threadIdx.x")等将循环轴绑定到GPU线程块和线程上，支持unroll循环展开、vectorize向量化，并通过cache\_read/cache\_write引入共享内存缓存等手段优化访存。TE 具有完整的可调度性接口，并支持 AutoTVM/AutoScheduler 等自动调优框架，在此基础上可搜索最佳调度策略以实现高效的GPU内核生成。

TensorIR 设计定位于前端算子建模完成后、生成硬件代码前的阶段，承接 TE 或高层IR的计算，并提供完全可调度的循环级别结构。这个是我们想要用户控制的那个级别，但是语法风格是TVMScript式Python AST（显式循环+with T.block），偏命令式，用户还是不太能接受的。没事有TileLang。

### 六、[modular/modular(MOJO) 24.3k](https://link.zhihu.com/?target=https%3A//github.com/modular/modular)

在[AI民主化的终章](https://link.zhihu.com/?target=https%3A//www.modular.com/blog/modulars-bet-to-break-out-of-the-matrix-democratizing-ai-compute-part-10)，chris提出了自己的解决方案，就是MOJO ，他想作为AI infra公司为各vendor提供服务。

我在LeetGPU对MOJO做了尝试。

[](https://zhuanlan.zhihu.com/p/1908980999993402643)

MOJO写起来很像CUDA，也是thread级别。MOJO是强类型的语言，不支持implicit conversions（隐式类型转换），提供了`@parameter`做为编译期常量参数（compile-time constant）的修饰符。当然也有一些类似Triton的封装，但是why not CuTeDSL or tilelang。比较好的结局大概是被AMD收购，大多数vendor应该都没动力买它的服务，vendor对自己的硬件都很保密。

### 七、[halide/Halide 6.1k](https://link.zhihu.com/?target=https%3A//github.com/halide/Halide)

Halide 提供了 Python绑定，所以可以不用C++。主要用于图像处理、张量运算、信号处理等数据局部性强的场景，有独立的计算+调度语法。

### 八、[Tiramisu-Compiler/tiraisu 0.94k](https://link.zhihu.com/?target=https%3A//github.com/Tiramisu-Compiler/tiramisu)

Tiramisu 受到 Halide 启发，但设计目标更偏向多层嵌套循环、复杂调度结构和 polyhedral 分析。基于 ISL，偏向手工 schedule。

### 九、NVIDIA的[cuTile](https://zhida.zhihu.com/search?content_id=259380445&content_type=Article&match_order=1&q=cuTile&zhida_source=entity)

[X原文](https://link.zhihu.com/?target=https%3A//x.com/blelbach/status/1902113767066103949)

准备阻击 Triton 的DSL，对标Triton。Vendor比用户更容易拿到性能，估计不会开源。

![](https://pic3.zhimg.com/v2-14c354948222132dc90c27e6352416e4_1440w.jpg)

  

cuTile 软件设计

![](https://pic1.zhimg.com/v2-dfcbbabe781298abe6b032d5ae801890_1440w.jpg)

  

我在这里也说下我的看法，cuTile的性能需要跑过Triton这座大山比较多才能挤掉Triton已经占据的空间。不暴露底层接口看起来有困难的，但是从CuTeDSL那里搞点优化经验or用一些**没暴露**的硬件接口也能得到。Triton是开源的，用户可以修改Triton源码去拿到性能，开源对于想要**榨干性能**的客户是非常重要的。有Triton和CuTeDSL打样cuTile肯定是易用的，所以实际看Nvidia能为我们带来多少性能提升了，黑盒非常牛的话大家是会买账的。

### **十、**[pytorch-labs/tritonbench](https://link.zhihu.com/?target=https%3A//github.com/pytorch-labs/tritonbench)**性能对比**

我的测试环境GPU H20 SXM 96GB \* 1，CPU: 16 核， 内存: 154 GB。本机CUDA 12.8, Driver Version: 550.127.05。Conda内Python3.12.11，torch 2.8.0.dev20250623+cu128，pytorch-triton==3.3.1+gitc8757738，flash\_attn\_3==3.0.0b1，tilelang==0.1.3，tk用的是87fa717(Apr 8, 2025)。

###   
**1、flash\_attention**

![](https://pic2.zhimg.com/v2-292a6c96f748ccb5de5d5f6e5b4cb0cb_1440w.jpg)

说明：

aten：原始 PyTorch 实现，最慢，超大规模下会 OOM。

sdpa：torch.nn.functional.scaled\_dot\_product\_attention（现代版本 PyTorch 内置优化）

当然也不一定公平啊，只能说现有开源的代码是这样，baseline的艺术，你有更好的实现可以给这个项目提pr。

flash\_attention\_v3 应该还没针对h20做优化，当然h20的计算能力本来就弱。下图是评论区小伙伴在H100的bench。

![](https://pic2.zhimg.com/v2-3de67c21c0a9ee1b0944bf6da0aa82bf_1440w.jpg)

是不是硬件变多了根本来不及优化啊，当然h20是国内特供，肯定有闭源的性能好的。另外Tri dao大概已经在做Blackwell的优化了，这个组的人是真的喜欢写kernel。

### 2、gemm

针对(256, 256, 256)的shape

![](https://picx.zhimg.com/v2-cfc3a6566a71f0d36a87c7159309bf87_1440w.jpg)

### 3、fp8gemm

![](https://pic4.zhimg.com/v2-da875e4aaa49b7aab3c38b71a9271c0d_1440w.jpg)

![](https://picx.zhimg.com/v2-d3674641cd1c5c94cf14b68797589b61_1440w.jpg)

### 4、int4\_gemm

![](https://picx.zhimg.com/v2-5b2bde291f66b9e4716589d7c59ed563_1440w.jpg)

![](https://pic1.zhimg.com/v2-cee90eb6bc059a9b2d01a5f9355f86f4_1440w.jpg)

### 5、layer\_norm

![](https://pic1.zhimg.com/v2-a8364a026c769f785abe6bb1ee866dbc_1440w.jpg)

### 6、softmax

![](https://pic3.zhimg.com/v2-9fbd2ea47d65ccbb6ad942f2af0efc60_1440w.jpg)

### 7、Triton launch\_latency

![](https://pic2.zhimg.com/v2-af530bff2eed2df8ce813d832c12ce6f_1440w.jpg)

### 附录

[HazyResearch/ThunderKittens 2.5k](https://link.zhihu.com/?target=https%3A//github.com/HazyResearch/ThunderKittens) 偏框架了些，包含 DSL 风格的 kernel 定义和 schedule API，但是性能不错的，是C++

[jax-ml/jax 32.6k](https://link.zhihu.com/?target=https%3A//github.com/jax-ml/jax) 以 NumPy 风格为基础的高性能数值计算框架，支持自动微分（Autograd）、JIT 编译和 GPU/TPU 加速

[Jittor/jittor 3.2k](https://link.zhihu.com/?target=https%3A//github.com/Jittor/jittor) 深度学习框架

[NVIDIA/warp 5.2k](https://link.zhihu.com/?target=https%3A//github.com/NVIDIA/warp) 评论区的大佬提到了NVIDIA的warp，写Python来进行物理仿真，用于机器人、布料、柔体、弹簧等模拟，1:1 复刻CUDA，也具备框架的功能。