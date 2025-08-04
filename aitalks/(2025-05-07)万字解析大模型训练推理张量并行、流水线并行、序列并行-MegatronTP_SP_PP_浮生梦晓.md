# 万字解析大模型训练推理张量并行、流水线并行、序列并行-MegatronTP/SP/PP

**Author:** 浮生梦晓

**Date:** 2025-05-07

**Link:** https://zhuanlan.zhihu.com/p/1898828998898853670

## [张量并行](https://zhida.zhihu.com/search?content_id=256922632&content_type=Article&match_order=1&q=%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C&zhida_source=entity)\-TP

### [Megatron-LM](https://zhida.zhihu.com/search?content_id=256922632&content_type=Article&match_order=1&q=Megatron-LM&zhida_source=entity): Training Multi-Billion Parameter Language Models Using

Model Parallelism

张量并行及[流水线并行](https://zhida.zhihu.com/search?content_id=256922632&content_type=Article&match_order=1&q=%E6%B5%81%E6%B0%B4%E7%BA%BF%E5%B9%B6%E8%A1%8C&zhida_source=entity)都属于[模型并行](https://zhida.zhihu.com/search?content_id=256922632&content_type=Article&match_order=1&q=%E6%A8%A1%E5%9E%8B%E5%B9%B6%E8%A1%8C&zhida_source=entity)的一种情况。

（ps:大家咋一个劲的收藏，帮忙点点赞呗: )）

张量并行可以将矩阵乘法等运算操作按矩阵行或者矩阵列来切分，在不同设备上并行执行计算，最后通过集合通信来合并结果。

张量并行的主要挑战在于如何切分参数和计算任务，以保证计算的一致性和通信的高效性。例如，在进行矩阵乘法时，必须确保各设备上的部分结果在数学上是一致的。此外，通信开销也是一个重要考虑因素，需要在计算和通信之间找到平衡点，以达到最佳性能。

-   矩阵切分计算原理：

![](images/v2-6534d6eb600596c393cd30224728aa66_1440w_fe7fb4091074.jpg)

![](images/v2-abd5d1592de652a52e5eb66de34ce074_1440w_874fa7d69556.jpg)

MLP实验：

```text
class MLP(nn.Module):
    def __init__(self, hidden_size,intermediate_size,dropout_prob,seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.gate_proj = nn.Linear(hidden_size,intermediate_size,bias=False)
        self.up_proj = nn.Linear(hidden_size,intermediate_size,bias=False)
        self.down_proj = nn.Linear(intermediate_size,hidden_size,bias=False)
        self.act_fn  = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self,hidden_state):
        output =  self.down_proj(self.act_fn(self.gate_proj(hidden_state) * self.up_proj(hidden_state)))
        return output
    
class TP_MLP(nn.Module):
    def __init__(self, hidden_size,intermediate_size,dropout_prob,seed=42):        
        super().__init__()
        torch.manual_seed(seed)
        self.gate_proj = nn.Linear(hidden_size,intermediate_size,bias=False)  
        self.up_proj = nn.Linear(hidden_size,intermediate_size,bias=False)  
        self.down_proj = nn.Linear(intermediate_size,hidden_size,bias=False)
        self.act_fn  = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self,hidden_state):
        intermediate_size = self.gate_proj.weight.data.shape[0]
        hidden_size = self.gate_proj.weight.data.shape[1]
        gate_proj1_data = self.gate_proj.weight.data[:intermediate_size//2,:]
        gate_proj2_data = self.gate_proj.weight.data[intermediate_size//2:,:]
        self.gate_proj1 = nn.Linear(hidden_size,intermediate_size//2,bias=False)
        self.gate_proj1.weight.data = gate_proj1_data
        self.gate_proj2 = nn.Linear(hidden_size,intermediate_size//2,bias=False)
        self.gate_proj2.weight.data = gate_proj2_data
        up_proj1_data = self.up_proj.weight.data[:intermediate_size//2,:]
        up_proj2_data = self.up_proj.weight.data[intermediate_size//2:,:]
        self.up_proj1 = nn.Linear(hidden_size,intermediate_size//2,bias=False)
        self.up_proj1.weight.data = up_proj1_data
        self.up_proj2 = nn.Linear(hidden_size,intermediate_size//2,bias=False)
        self.up_proj2.weight.data = up_proj2_data
        self.down_proj1 = nn.Linear(intermediate_size//2,hidden_size,bias=False)
        self.down_proj2 = nn.Linear(intermediate_size//2,hidden_size,bias=False)
        down_proj1_data = self.down_proj.weight.data[:,:intermediate_size//2]
        down_proj2_data = self.down_proj.weight.data[:,intermediate_size//2:]
        self.down_proj1.weight.data = down_proj1_data
        self.down_proj2.weight.data = down_proj2_data

        output1 = self.down_proj1(self.act_fn(self.gate_proj1(hidden_state) * self.up_proj1(hidden_state)))
        #（bsz,seq,inter//2）* (bsz,infer//2, hidden) -> (bsz,seq,hidden)
        output2 = self.down_proj2(self.act_fn(self.gate_proj2(hidden_state) * self.up_proj2(hidden_state)))
        output = output1 + output2

        return output

batch_size = 4
sequence_length = 128
hidden_size = 512
Num_gpus = 8
drop_prob = 0.01
intermediate_size = hidden_size * 4
X = torch.randn((batch_size,sequence_length,hidden_size)) 
mlp = MLP(hidden_size=hidden_size,intermediate_size=intermediate_size,dropout_prob=drop_prob)
mlp_tp = TP_MLP(hidden_size=hidden_size,intermediate_size=intermediate_size,dropout_prob=drop_prob)
mlp.gate_proj.weight = mlp_tp.gate_proj.weight
mlp.up_proj.weight = mlp_tp.up_proj.weight
mlp.down_proj.weight = mlp_tp.down_proj.weight
mlp.act_fn = mlp_tp.act_fn
mlp.dropout = mlp_tp.dropout
X_mlpoutput = mlp(X)
X_mlpoutput_tp = mlp_tp(X)
torch.allclose(X_mlpoutput,X_mlpoutput_tp,atol=1e-5)
```

![](images/v2-b5602430c6a2f08c308b75b834baf832_1440w_469b021d9a1d.jpg)

-   **Embedding拆分，列切分，行切分实现复杂：**

```text
vocab_size = 10000
hidden_size = 512
batch_size = 4
sequence_length = 128
embedding = nn.Embedding(vocab_size, hidden_size)
X = torch.randint(0,vocab_size, (batch_size, sequence_length))
X_embed = embedding(X)
embedding1 = nn.Embedding(vocab_size, hidden_size//2)
embedding1.weight.data = embedding.weight.data[:,:hidden_size//2].clone()
embedding2 = nn.Embedding(vocab_size, hidden_size//2)
embedding2.weight.data = embedding.weight.data[:,hidden_size//2:].clone()
X_embed1 = embedding1(X)
X_embed2 = embedding2(X)
X_embed_v1 = torch.cat([X_embed1,X_embed2],dim=-1)
torch.allclose(X_embed,X_embed_v1)
```

![](images/v2-fff287408d42e208f81fd32f11b53cb3_1440w_04d2193f5ceb.jpg)

![](images/v2-4d80b595b88953556317ac888658ea3a_1440w_b1f102cbb6b8.jpg)

通信分析：

前向传递的分布式操作与反向传播相反，但每次操作是一个完整的[All-Reduce](https://zhida.zhihu.com/search?content_id=256922632&content_type=Article&match_order=1&q=All-Reduce&zhida_source=entity),每一层attention和mlp这前向及反向过程中各存在一个All-Reduce，共计4个AllReduce操作。

## 流水线并行-PP

最好提出流水线模型并行的论文：

### [GPipe](https://zhida.zhihu.com/search?content_id=256922632&content_type=Article&match_order=1&q=GPipe&zhida_source=entity): Easy Scaling with Micro-Batch Pipeline Parallelism

文章中将模型的不同层加载到不同的GPU，输入逐层进行传递来构建计算题，中间激活值保留在各自GPU上用于反向传播使用。

![](images/v2-03b51be3c6f17cacf84af0bc94f2ea01_1440w_e1a8a69d79ea.jpg)

本篇论文内容上较少，存在大量实验结果篇章，考虑到文章较早，不再分析结果。

整体实现方式见上图a，将模型按照层进行切分到不同GPU上，如图b所示，在4个GPU上开展流水线，但流水线会带来一些问题，在时间步上会存在大量气泡，即GPU资源闲置的状态，为了提升GPU利用率，原文在模型分层并行基础上又添加了数据并行，一定程度上减少了时间步上的“气泡”。

[F-then-B策略](https://zhida.zhihu.com/search?content_id=256922632&content_type=Article&match_order=1&q=F-then-B%E7%AD%96%E7%95%A5&zhida_source=entity)：在每次迭代时间步的最后进行统一的参数更新（对同一批数据前向传递后不会马上进行反向传播，会计算另外批次的前向传递计算）。

本文还提到了重计算思想（引用之前文章）：

N:大batchsize ； L：模型层数； K：GPUS ； M：小batchsize

-   如果不使用模型分层及重计算，整体的内存峰值大概在O（N x L）
-   使用模型分层及重计算，整体的内存峰值大概在O（N + (L / K) \*（N / M））
-   作者计算得到单itep气泡率：O((K-1)/ (M + K -1 )))，实验中发现M ≥ 4 × K时气泡可以忽略不计。
-   从下表中每个GPU上的峰值内存可以看出，使用流水线并行时重计算是必要的，但是重计算又会造成最后一个gpu存储和计算压力过大，本文没有提到这个问题。

![](images/v2-291cf20c199cb3d455faf32ebc2b9ce6_1440w_2ea183a451a8.jpg)

### [PipeDream](https://zhida.zhihu.com/search?content_id=256922632&content_type=Article&match_order=1&q=PipeDream&zhida_source=entity): Generalized Pipeline Parallelism for DNN Training

微软的针对Gpipe中F-then-B策略模式的缺陷提出的1F1B策略模式。

![](images/v2-a091fdf46a270ac190c3536c0c922cc6_1440w_1058e995e993.jpg)

上图中，左图是模型分层并行示意图（无数据并行），右图为Gpipe示意图，在F-then-B策略模式下，每次迭代完成后，会集中进行梯度更新，会导致GPU资源浪费十分严重。

![](images/v2-1917a66f590168faafc7d24e9786523b_1440w_a13afa05405b.jpg)

本文提出了1F1B模式，在完成前向传递后直接进行反向回传更新梯度，不会等待一个batch数据完成。

但这种情况会造成一个问题，数据1前向反向执行完成后，2,3,4还没有完成，即只有1数据梯度更新后对于数据5的前向传递有影响，但5的反向传递结束时2,3,4部分数据对梯度的影响也更新了，会导致数据5做梯度更新时参数状态不一致，论文中提出了weight stashing机制（权重暂存），即每次都用最新的参数（5数据前传时用1数据梯度更新后的参数），为每一个GPU都不同stage（不同数据）创建一个buffer，用来保存同一个stage前传的参数，保证其反向传播时使用。

![](images/v2-a8834f64d6b161649cfcfd8495616348_1440w_abd4c9e46676.jpg)

### Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM

受gpipe流水线并行启发，将其与自家的张量并行相结合。使用1F1B策略模式。

![](images/v2-71e997e85d74ff86973eb6ebb0e197a9_1440w_5a057deeef7a.jpg)

如上图：综合使用Tensor并行与流水线并行（3D并行）

![](images/v2-4e22922882463749491c684a7082e803_1440w_711f15bdd044.jpg)

论文中分析了Gpipe的弊端问题，因为是F-Then-B的运行策略，导致模型没完成应该minibatch(非micobach)就会有一个GPU设备空置期（粗黑线部分），同时这种策略带来的气泡也比较严重，同时由于重计算原因，F-Then-B模式会导致最后的GPU激活值占用显存严重。

论文中对于Gpipe中气泡做了估算：

-   m: 一个batch中的microbatch size
-   p: pipline stages ,pipeline 维度上的GPU数量
-   t(id) : 每个iteration的时间
-   tf tb : 每个microbarch 数据在每个stage中的前向和反向的时间
-   因此每个batch数据一个iteration的气泡的总占用时间为：t(pb) = (p-1)(tf + tb) (p-1):前向和反向都要经历p-1个阶段，与m无关。
-   每个batch数据一个iteration总时间：m(tf + tb)
-   因此气泡占用率：

![](images/v2-d6fa98a7b4d8d295c67406852867f43b_1440w_87627d32b18d.jpg)

因此，当m远大于p都时候，气泡可以忽略不计，但是m过大会造成数据存储压力太大。（中间值存储），因此论文又分析到PipeDream，PipeDream调度算法中首先是warmup阶段，之后是稳定运行阶段（稳定运行阶段每个GPU会1个froward及1个backward的交替执行），最后进入梯度更新阶段，这种策略不会让m个microbatch的激活值同时进行存储，可以减缓数据存储压力，但是整体的气泡时间是和Gpipe相同的，只是节省了数据存储。

![](images/v2-4c22062509fbdcd0ce43450f7595ffe3_1440w_f673aefd4a8c.jpg)

本论文就在PipeDream基础上提出了interleaved 1F1B，每个GPU不再处理一个连续的模型子集层，而是多个层（例如：假设在4个GPU上执行pipeline并行，原来是gpu0上是第1-4层，gpu1上是5-8层….，使用interleaved策略后成了，gpu0上是1,2,9,10层，gpu1上是3,4,11,12层…）使用这种方法使得摸个gpu在pipeline中单次stage时间变短。

如果按照每个gpu上有v个stage来计算(上例中v=2),那么每个microbatch在每个stage中的前向及反向将变为tf/v 和tb/v，所以每个iteration中气泡垫总时间就成了：

![](images/v2-6f28ecd49e74825165bc7fbf214c68fa_1440w_dfa3e2a5d272.jpg)

再计算每个iteration中气泡的占用率：

![](images/v2-2823a89f385f2eb2f8f33c4d8f5aac5e_1440w_b61a4d13943c.jpg)

气泡率会下降v倍，但是interleaved1F1B会造成额外的通信开销（提升v倍）

![](images/v2-c8d01afb5d4fb6126d2ce9e55eab984d_1440w_feeea1c2d63e.jpg)

上图在之前TP论文中已经展示，实际上megatron这篇Pipeline论文并未只提出interleaved1F1B对于模型训练的改进，也是将PP与TP进行混合使用，所以这文章中再次提出了TP概念，并针对PP和TP并行进行了详细分析，不再展示。

论文中花了很大篇幅关于TP+PP的通讯及计算优化

![](images/v2-247424f576826c4f42b4af250ffd2856_1440w_60b4873a58e5.jpg)

对于通讯优化，因为同一node内采用NVLink，不同node采用的无线带宽连接，减少不同node直接传输的数据内容，尽量使用使用NVLink操作。

计算优化，编写了多种运算内核。

论文附录中关于模型训练Flops的估算：

```text
前置知识：
 ：transformer layers, 
h:hidden size,
s: sequence length, 
V:vocabulary size , 
B:training batch size.

  ×  ×  ×  matrix multiplication requires 2  ×  ×  FLOPs
(factor of 2 needed to account for multiplies and adds).

注意：若前向传递总Flops为T，则反向传播总Flops为2T（需要计算权重及inputs），
如果需要重计算则需要在backward前重新进行一次前向传递，Flops为T，故总为4T。

```

![](images/v2-a7de911e04859a3bcab0cf34348e303c_1440w_0d7d99229852.jpg)

![](images/v2-1abd6e2584d5814aeac20c84a0919890_1440w_d3453c312685.jpg)

## [序列并行](https://zhida.zhihu.com/search?content_id=256922632&content_type=Article&match_order=1&q=%E5%BA%8F%E5%88%97%E5%B9%B6%E8%A1%8C&zhida_source=entity)\-SP

序列并行最提出自两篇论文，两篇论文中提及的实际概念不同。

### 国内[Colossal-AI](https://zhida.zhihu.com/search?content_id=256922632&content_type=Article&match_order=1&q=Colossal-AI&zhida_source=entity)：Sequence Parallelism: Long Sequence Training from System Perspective

输入数据按照序列维度切分，也就是一句话切分成多段话，在endocer模块的主要模块中，mlp，归一化层直接可以进行序列并行，但是attention不行（QKV计算依赖），作者提出了Ring-attention。（借鉴了Ring-Allreduce算法）

![](images/v2-e3ca7cc5f7a5bbfc5d877e6bb013f7b0_1440w_280877839188.jpg)

![](images/v2-13962ec6f50755fdda2c343410d76c5c_1440w_a702ee8f6e5a.jpg)

-   [Ring-Attention](https://zhida.zhihu.com/search?content_id=256922632&content_type=Article&match_order=1&q=Ring-Attention&zhida_source=entity)实现了RingQK和RingAV两种计算方式，因此分布式的Attenion分为两个步骤：

-   计算query和key的乘积
-   计算V

（原谅字体较丑）

![](images/v2-ddcaa6619c244de8d1b6abb1c4463de7_1440w_aeca3b3accb1.jpg)

![](images/v2-1c35a0c4048eb4a433ba09964d8cc0f2_1440w_fed87cdb610f.jpg)

### **RMS\_Norm——序列并行代码示意：**

```text
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    
batch_size = 4
sequence_length = 128
hidden_size = 512
Num_gpus = 8
X = torch.randn((batch_size,sequence_length,hidden_size)) 
X_SP = torch.split(X, sequence_length//Num_gpus, dim=1)
rms_norm = RMSNorm(hidden_size=hidden_size)
X_rmsoutout = rms_norm(X)
X_rmsoutout.shape
X_SP_rmsoutput = []
for i in X_SP:
    i_rmsoutput = rms_norm(i)
    X_SP_rmsoutput.append(i_rmsoutput)
X_SP_rmsoutput = torch.cat(X_SP_rmsoutput, dim=1)
```

![](images/v2-321b234a9c412e38d2a89799de425ce9_1440w_2e25c290ac27.jpg)

### MLP——序列并行

```text
class MLP(nn.Module):
    def __init__(self, hidden_size,intermediate_size,dropout_prob):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size,intermediate_size,bias=False)
        self.up_proj = nn.Linear(hidden_size,intermediate_size,bias=False)
        self.down_proj = nn.Linear(intermediate_size,hidden_size,bias=False)
        self.act_fn  = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self,hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state) * self.up_proj(hidden_state)))
    
batch_size = 4
sequence_length = 128
hidden_size = 512
Num_gpus = 8
drop_prob = 0.01
intermediate_size = 256
X = torch.randn((batch_size,sequence_length,hidden_size)) 
X_SP = torch.split(X, sequence_length//Num_gpus, dim=1)

mlp = MLP(hidden_size=hidden_size,intermediate_size=intermediate_size,dropout_prob=drop_prob)
X_mlpoutput = mlp(X)
X_SP_mlpoutput = []
for i in X_SP:
    i_mlpoutput = mlp(i)
    X_SP_mlpoutput.append(i_mlpoutput)
X_SP_mlpoutput = torch.cat(X_SP_mlpoutput, dim=1)
torch.allclose(X_mlpoutput, X_SP_mlpoutput)
```

![](images/v2-4a0712c6b9c7bed2256619a2133e26e0_1440w_ba9a8f2b0d4c.jpg)

## Megatron-SP：REDUCING ACTIVATION RECOMPUTATION IN LARGE TRANSFORMER MODELS

这篇论文主要实现TP与SP的混合使用，目的是为了降低单独TP带来的缓存占用，文章一直在分析使用不同切分策略整体的激活值占用缓存情况。

![](images/v2-0a517bcdabc38ced6c27ab67b72810cf_1440w_41cd1f032251.jpg)

计算激活值内存前置条件：

-   假设16位存储（2bytes）
-   只计算占用高的存储部分，忽略小的buffers

下图是标准的decoder-only架构组成：

![](images/v2-cd64396f38ad9cc7c22b8ce2dfdadafc_1440w_c3e4d7c5fc7d.jpg)

![](images/v2-d2fb435d252e22e40d0695b9d4943e84_1440w_8afad38dc6dd.jpg)

Attention block 中需要存储的激活值:

-   Q，K，Vstates: 只需要存储输入，三者输入相同，都是hidden state:,共：2bsh
-   QK(T) : 存储Q和K ，共：4bsh
-   Softmax输出值: (batch,head\_num,seq,seq) ,，存储供反向传播用，共：2bssa
-   Softmax dropout：mask，softmax的一半，共：bssa
-   dropout后和V：2bssa + 2bsh
-   共计：11sbh + 5assb

MLP中需要存储的激活值：

-   两个线性层的输入值：2bsh + 8bsh （默认mlp intermidiate为4倍）
-   激活层输入：8bsh
-   dropout：存储mask：bsh
-   共计19bsh

层归一化：

-   每层有两个，每个输入为2bsh，共4bsh

故每一层所有需要存储的激活值共：sbh(34 + 5as/h)

以我们训练qwen2的7B为例，训练时中间激活值存储大小为：

-   s = 2048 （假设）
-   b = 8 （假设）
-   h = 3584
-   a = 28
-   layer\_num = 28
-   共约187.4G

### TP（张量并行）：

![](images/v2-2c8ea693bda023987bc3db0aca89ebb2_1440w_08614d1d4d28.jpg)

当仅仅使用Megatron-TP，计算后整体的激活值存储为：

sbh(10+24/t + 5as/ht)

以我们训练qwen2的7B为例，训练时中间激活值存储大小为：

-   s = 2048 （假设）
-   b = 8 （假设）
-   t = 8 (假设) tensor parallel size
-   h = 3584
-   a = 28
-   layer\_num = 28
-   共约37.8G

SP（序列并行）+TP（张量并行）

![](images/v2-df8d0ed0b62de4a9b16d45902481453b_1440w_c2ad008d069c.jpg)

![](images/v2-1e4b25b8bc9f5782b7c08cbd27f3e0d5_1440w_da64fdb067f1.jpg)

每层激活值大小：

![](images/v2-eb9b71ca9aad4943fe4c579f07ad98dd_1440w_24f1c9421dfa.jpg)

以我们训练qwen2的7B为例，训练时中间激活值存储大小为：

-   s = 2048 （假设）
-   b = 8 （假设）
-   t = 8 (假设) tensor parallel size
-   h = 3584
-   a = 28
-   layer\_num = 28
-   共约23.43G

### PP（Pipeline 流水线并行）

论文中没有给出示意图，论文以1F1B为例

![](images/v2-a33dae828178886ab024c8a1776d0444_1440w_706070fa1ca8.jpg)

![](images/v2-6544bf9087edb26f97905c23e8804044_1440w_34f30b9b50e3.jpg)

**汇总：**

![](images/v2-7675fa8ca8c1cca866e7e9ae040318b3_1440w_1431df505bef.jpg)

  

  

引用论文：

[https://arxiv.org/pdf/2105.13120](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2105.13120)

[https://arxiv.org/pdf/2205.05198](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2205.05198)

[https://arxiv.org/pdf/1811.06965](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1811.06965)

[https://arxiv.org/pdf/1909.08053](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1909.08053)

[https://arxiv.org/pdf/2104.04473](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2104.04473)

[https://www.pdl.cmu.edu/PDL-FTP/BigLearning/sosp19-final271.pdf](https://link.zhihu.com/?target=https%3A//www.pdl.cmu.edu/PDL-FTP/BigLearning/sosp19-final271.pdf)