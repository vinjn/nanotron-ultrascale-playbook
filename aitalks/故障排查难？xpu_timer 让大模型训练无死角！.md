# 故障排查难？xpu_timer 让大模型训练无死角！

**作者：** 张吉

---

# 作者介绍：张吉，从事于搜推/LLM 的训练优化，专注于系统底层/网络优化。

背景

随着大型模型的参数量从十亿量级跃升至万亿级别，其训练规模的急剧扩张不仅引发了集群成本的显著上涨，还对系统稳定性构成了挑战，尤其是机器故障的频发成为不可忽视的问题。对于大规模分布式训练任务而言，可观测性能力成为了排查故障、优化性能的关键所在。所以从事大型模型训练领域的技术人，都会不可避免地面临以下挑战：
1. 训练过程中，性能可能会因网络、计算瓶颈等多种因素而不稳定，出现波动甚至衰退；
2. 分布式训练是多个节点协同工作的，任一节点发生故障（无论是软件、硬件、网卡或 GPU 问题），整个训练流程均需暂停，严重影响训练效率，而且浪费宝贵的 GPU 资源。

但在实际的大模型训练过程中，这些问题是很难排查的，主要原因如下：
1. 训练过程为同步操作，很难通过整体性能指标来排除此时哪些机器出现问题，一个机器慢可以拖慢整体训练速度；
2. 训练性能变慢往往不是训练逻辑/框架的问题，通常为环境导致，如果没有训练相关的监控数据，打印 timeline 实际上也没有任何作用，并且同时存储 timeline 文件的存储需求也较高；
3. 分析工作流复杂，比如训练 hang 住时，需要在 torch 超时前完成所有栈的打印再去分析，面对大规模任务时很难再 torch 超时内完成

在大规模分布式训练作业中，可观测的能力对于问题排查和性能提升显得尤为重要。蚂蚁在大规模训练的实践中，通过开发 xpu_timer 库，来满足 AI 训练的可观测性需求。 **未来我们会将 xpu timer 开源到 DLRover 中，欢迎大家一起合作共建 :)**xpu_timer 库是一款 profiling 工具，通过截获 cublas/cudart 库，使用 cudaEvent 为训练中的矩阵乘/集合通讯操作进行计时的工具，同时有 timeline 分析，hang 检测，hang 栈分析等功能，设计上支持多种异构平台。该工具具备以下特点：
1. 对代码无入侵，对训练性能无损耗，可以常驻于训练进程；
2. 对用户无感，框架无关
3. 低损耗/精度高
4. 可进行指标聚合/投递，便于数据的进一步处理与分析；
5. 信息存储效率高
6. 便捷的交互接口：提供友好的对外接口，便于与其他系统集成及用户直接操作，加速洞察与决策过程。

设计方案

首先，针对训练 hang/性能下降的问题，我们设计了一个常驻的 kernel 计时：

1. 大部分场景下训练 hang 住是 nccl 操作导致，通常情况只需要记录矩阵乘与集合通讯即可；
2. 针对单机出现性能下降（ECC，MCE），只需要记录矩阵乘即可，同时分析矩阵乘也可以查看用户的矩阵形状是否科学，发挥出 tensorcore 的最大性能，各个框架实现矩阵乘时直接使用 cublas。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/ZRiaNYvFqgia9wpqicHSt6gJXlo0sDmrtHfpOHmNaOHZYPjNSBoqCmM5N46ErBU2USDjndALZsWjJRFmqPNzeoGoA/640?wx_fmt=other&from=appmsg&randomid=pf0swi5c)

因此我们设计在 kernel launch 层进行截获，运行时设置 LD_PRELOAD 即可对关注的操作进行 tracing。该方法只能用于动态链接的情况，目前主流的训练框架均为动态链接。针对 NVIDIA 的 GPU，我们可以关注如下符号：
1. ibcudart.so

- cudaLaunchKernel
- cudaLaunchKernelExC

1. libcublas.so

- cublasGemmEx
- cublasGemmStridedBatchedEx
- cublasLtMatmul
- cublasSgemm
- cublasSgemmStridedBatched

在适配不同硬件时，通过不同模板类来实现不同的 tracing 功能。

Workflow

以 PyTorch 为例，Launch Thread 为 torch 主线程，working thread 为 library 内部的工作线程。这里截获上述描述的 7 个 kernel![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/ZRiaNYvFqgia9wpqicHSt6gJXlo0sDmrtHfFK23dwsM0yAxFrNowCGQ4tpc2dK3HtrsxicVPmia8rHicP9JFdNic7FxqA/640?wx_fmt=png&from=appmsg&randomid=2r2crm5j)

使用方法&效果

**前置条件**

1. NCCL 静态编译至 libtorch_cuda.so
2. torch 动态链接 libcudart.so

如果 NCCL 时动态链接的，可以提供自定义函数偏移，运行时动态解析。安装好 Python 包后会有如下命令行工具
| xpu_timer_gen_syms | 用于动态生成解析 nccl 的 library动态注入函数偏移 |
| --- | --- |
| xpu_timer_gen_trace_timeline | 用于生成 chrome trace |
| xpu_timer_launch | 用于挂载 hook 包 |
| xpu_timer_stacktrace_viewer | 用于生成超时后的可视化栈 |
| xpu_timer_print_env | 打印 libevent.so 地址打印编译信息 |
| xpu_timer_dump_timeline | 用于触发 timeline dump |

```
XPU_TIMER_XXX=xxx LD_PRELOAD=`path to libevent_hook.so` python xxx

```

实时动态抓取 timeline

每个 rank 均有一个端口服务，需要同时给所有 rank 发送命令，启动端口为 brpc 服务端口每个 rank trace 数据大小为 32B，保存 1000 条，大小为 32K，生成的 timeline json 大小为 150K * world size，远远小于 torch 的 timeline**基本用法**

```
usage: xpu_timer_dump_timeline [-h]
--host HOST 要 dump 的 host
--rank RANK 对应 host 的 rank
[--port PORT] dump 的端口，默认 18888，如果一个 node 用了所有的卡，这个不需要修改
[--dump-path DUMP_PATH] 需要 dump 的地址，写绝对路径，长度不要超过 1000
[--dump-count DUMP_COUNT] 需要 dump 的 trace 个数
[--delay DELAY] 启动这个命令后多少秒再开始 dump
[--dry-run] 打印参数

```

**单机情况**

```
xpu_timer_dump_timeline \
  --host 127.0.0.1 \
  --rank "" \
  --delay 3 \
  --dump-path /root/lizhi-test \
  --dump-count 4000

```

```
# 如下图所示，如果你的作业有 master/worker 混合情况（master 也是参与训练的）
# 可以写 --host xxx-master --rank 0
# 如果还不确定，使用 --dry-run

xpu_timer_dump_timeline \
  --host worker \
  --rank 0-3 \
  --delay 3 --dump-path /nas/xxx --dump-count 4000

xpu_timer_dump_timeline \
  --host worker --rank 1-3 \
  --host master --rank 0 --dry-run 

dumping to /root/timeline, with count 1000
dump host ['worker-1:18888', 'worker-2:18888', 'worker-3:18888', 'master-0:18888']
other data {'dump_path': '/root/timeline', 'dump_time': 1715304873, 'dump_count': 1000, 'reset': False}

```

之后会在对应的 timeline 文件夹中增加如下文件![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/ZRiaNYvFqgia9wpqicHSt6gJXlo0sDmrtHfoibOtVVMUcPoddp5OTYqJIuhVibLgInKbP8AAMWuCAPUgQwnI1lNUahg/640?wx_fmt=png&from=appmsg&randomid=19baknk5)

之后在这个文件下下运行 xpu_timer_gen_trace_timeline
```
xpu_timer_gen_trace_timeline 
```

会生成 3 个文件：
1. merged_tracing_kernel_stack 辅助文件，火焰图原始文件
2. trace.json 合并后的 timeline
3. tracing_kernel_stack.svg，矩阵乘/nccl 的 callstack

### **一个 llama-recipes 32 卡 sft 分析的 case**

timeline 大致如下，每个 rank 会展示 matmul/nccl 两行，所有 rank 都会展示。注意，这里是没有前向/反向信息的，大致可以用时长来判断，反向是前向的 2 倍![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/ZRiaNYvFqgia9wpqicHSt6gJXlo0sDmrtHfKicJI81BZszBQ31678AKK4u0P6HgyFFLLZvr9w4DyYsoWSAQl3nX0RA/640?wx_fmt=png&from=appmsg&randomid=m6vw9ces)

前向 timeline，大约 87ms![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/ZRiaNYvFqgia9wpqicHSt6gJXlo0sDmrtHfZr3wLFfHIvGTATT7ruaSfxTnTqqibAqyLIee0KCickqxpSrEbtVrzywQ/640?wx_fmt=png&from=appmsg&randomid=19hx88ha)

反向 timeline 大致 173ms![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/ZRiaNYvFqgia9wpqicHSt6gJXlo0sDmrtHfPM0E2L2nRfibxXJUZ3TWGRbTjORZrxdboXvd2leeIsGOFh1iajPELnbw/640?wx_fmt=png&from=appmsg&randomid=5w7xr841)

一共 48 layer，共耗时(173+87)*48 = 12480ms，再加上 lmhead， embedding 等其他操作，约 13s，整体时间是对的上的。并且通过 timeline 发现通讯时间远远大于计算时间，可以确定是通讯导致的瓶颈。

hang住栈分析

用 pip 安装好包后，可以通过命令行工具进行分析，默认 kernel 超过 300 秒后会打印具体的栈信息，svg 图拖到 chrome 中即可观看，分别使用 pstack/py-spy 来打印对应的栈，打印结果在训练进程的 stderr 中。如果通过 conda 安装了 gdb，会使用 gdb 的 python api 来获取栈，可以获取到 lwp 名字，默认安装的 gdb8.2 有时候获取不到，conda gdb 默认地址为 /opt/conda/bin/gdb以下为一个 2 卡模拟 NCCL 超时的栈：![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/ZRiaNYvFqgia9wpqicHSt6gJXlo0sDmrtHfy1rjsh05oMUULrhbptgvyqzK2YbGtPdngPVsWVexcOuQCvp3MPn5Fg/640?wx_fmt=png&from=appmsg&randomid=k0pw169d)

### **以下为一个单机 8 卡 llama7B sft 训练的例子**

通过 python 包提供的工具，可以生成聚合栈的火焰栈图，这里可以看到没有 rank 1 的栈，因为在 8 卡训练时通过 kill -STOP rank1 模拟 hang，因此 rank1 处于 stop 状态。
```
xpu_timer_stacktrace_viewer --path /path/to/stack

```

在合并栈时，我们认为相同的 callpath 可以合并，也就是这个 stacktrace 完全一致，因此卡在主线程的地方大多会一样，但是如果有一些 loop，活跃的线程，打印的栈顶可能会不一致，但是在底层运行的会是相同的栈，比如 python 栈中线程都会卡在 _bootstrap@threading.py 上，另外火焰图的 samples 数没有任何意义。当检测到 hang 后，所有的 rank 生成对应的 stacktrace 文件（rank1 suspend 了，所以没有），每个文件中包含了 python/c++ 的完整栈。![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/ZRiaNYvFqgia9wpqicHSt6gJXlo0sDmrtHfhDQf57ANX8GiaaTOUAcAIiakBpUHeUMwU5XxejichLt6ib42ABnfCxM8hw/640?wx_fmt=png&from=appmsg&randomid=ixjhurf8)

合并后的栈如下所示，用不同的颜色区分栈的类别，在 python 栈上可能只有青色和绿色：
1. 青色是 CPython/Python
2. 红色是 C/其他系统相关
3. 绿色是 Torch/NCCL
4. 黄色是 C++

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/ZRiaNYvFqgia9wpqicHSt6gJXlo0sDmrtHfCowKGk8OrwHLficBDCCelQYMqicWDvPe8iciaOurzIuaehqiafAwfEaZC2g/640?wx_fmt=png&from=appmsg&randomid=uj81bvek)

Python 栈如下，其中蓝色的框图为具体的栈，命名规则为：func@source_path@stuck_rank|leak_rank
1. func 当前函数名，如果 gdb 获取不到会显示 ??
2. source_path，这个符号在进程中的那个 so/source 地址
3. stuck_rank 代表哪些 rank 的栈进入到这里，连续的 rank 号会被折叠为 start-end，如 rank 0,1,2,3 -> 0-3
4. leak_rank 代表哪些栈没有进入到这里，这里 rank 号同样会被折叠

所以图中的含义为 rank0，rank2-7 都卡在了 synchronize 下，1 rank 没有进来，因此可以分析 rank1 有问题（实际被 suspend 了）。这个信息只有在栈顶才会被添加![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/ZRiaNYvFqgia9wpqicHSt6gJXlo0sDmrtHf3PhKQt6teIia0QwdCt4BfGRKmicb0o8g2DRTDSOBhaRwiaCKSJnj5icYicg/640?wx_fmt=png&from=appmsg&randomid=m3p3g0wj)

与之对应的可以看到 cpp 的栈可以看到主线程卡到了 synchronize 中，最终卡到了 cuda.so 中的获取时间上，同样是只有 rank1 没有这个栈可以认为 __libc_start_main 所在的栈代表进程的 entrypoint![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/ZRiaNYvFqgia9wpqicHSt6gJXlo0sDmrtHf7icsknNJH4882U1y1uqmcsIib7bpuZ3egX3QaPFmN68kqoo4iapAbOWhw/640?wx_fmt=png&from=appmsg&randomid=obmn43d9)

通常，可以认为栈最深的链路只有一个，如果出现了分叉，证明不同的 rank 卡在了不同的链路上。

Kernel 调用栈分析

timeline 中不像 torch 的 timeline 有 callstack，对此在生成 timeline 时会生成对应的栈文件名是 tracing_kernel_stack.svg，将这个文件拖到 chrome 中即可观察
- 绿色的是 NCCL 操作
- 红色的是 matmul 操作
- 青色的是 Python 栈

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/ZRiaNYvFqgia9wpqicHSt6gJXlo0sDmrtHfrETbLfVX8WN4kkDj5xDUYTwqFy3In2Uv5bXXXo70Uv7MxZbjaW2V3A/640?wx_fmt=png&from=appmsg&randomid=gec1rgyt)

Grafana大盘展示

## ![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/ZRiaNYvFqgia9wpqicHSt6gJXlo0sDmrtHfGJrgdImJNicA8Ju9Ixymgm2wHN4Sxz0gfC5nFDJSLicXEwOrhwRiaLFew/640?wx_fmt=png&from=appmsg&randomid=cddhq8z8)

未来计划

1. 加入 NCCL/eBPF 等更细粒度的 tracing，以便于更精确地分析和诊断训练过程中出现的挂起问题的根本原因；
2. 将支持包括各种国产显卡在内的更多硬件平台。

关于 DLRover

DLRover（Distributed Deep Learning System）是蚂蚁集团 AI Infra 团队维护的开源社区，是基于云原生技术打造的智能分布式深度学习系统。DLRover 使得开发人员能够专注于模型架构的设计，而无需处理任何工程方面的细节，例如硬件加速和分布式运行等；开发深度学习训练的相关算法，让训练更高效、智能，例如优化器。目前，DLRover 支持使用 K8s、Ray 进行自动化操作和维护深度学习训练任务。更多 AI Infra 技术请关注 DLRover 项目。

加入 DLRover 钉钉技术交流群：31525020959

DLRover Star一下：

https://github.com/intelligent-machine-learning/dlrover

文章推荐

[提高 AI 训练算力效率：蚂蚁 DLRover 故障自愈技术的创新实践](http://mp.weixin.qq.com/s?__biz=MzkyNzQyMjkxNQ==&mid=2247488262&idx=1&sn=9cf1e3eca25449474f4ec7afe5814461&chksm=c22913aaf55e9abc55fa95e78934499624cf1ccb0691b26ec53f62e468073a85cb730eabbbdb&scene=21#wechat_redirect)

[走近 AI Infra 架构师：在高速飞驰的大模型“赛车”上“换轮子”的人](http://mp.weixin.qq.com/s?__biz=MzkyNzQyMjkxNQ==&mid=2247488195&idx=1&sn=f58b818735b7e240e1db2237aa28471b&chksm=c229126ff55e9b7967180312239ddc8086727f942c760b5aa26613e55c519abab29b2d6db4f6&scene=21#wechat_redirect)

[【在线回放】NVIDIA GTC 2024 大会 | 如何降低 AI 工程成本？蚂蚁从训练到推理的全栈实践](http://mp.weixin.qq.com/s?__biz=MzkyNzQyMjkxNQ==&mid=2247488218&idx=1&sn=fe36e0b0e9ed85fb9895878aafe273b3&chksm=c2291276f55e9b6053ba8549e6ae7b0abda97dafe54ed310ed30353bd6a3ab6ee8a8bd65082c&scene=21#wechat_redirect)

![Image](https://mmbiz.qpic.cn/mmbiz_png/ZRiaNYvFqgia9MI7jwaicAymicVCNF9yI7BhhrcaRbN2EqNYzEv9mg4UAUpS4rVqevTCAJkQib3mI4ZqZ9vSicibBOORw/640?wx_fmt=other&wxfrom=5&wx_lazy=1&wx_co=1&tp=webp&randomid=mdnmsq9g)

点击「阅读全文」，在 GitHub 关注 DLRover

