# Triton Puzzles - 寒假摸鱼 (1)

**Author:** Garl

**Date:** 2025-01-26

**Link:** https://zhuanlan.zhihu.com/p/20269643126

Cornell Tech 的 [Sasha Rush 教授](https://link.zhihu.com/?target=https%3A//rush-nlp.com/) 有7个puzzle (triton, [llm training](https://zhida.zhihu.com/search?content_id=253076219&content_type=Article&match_order=1&q=llm+training&zhida_source=entity), [transformer](https://zhida.zhihu.com/search?content_id=253076219&content_type=Article&match_order=1&q=transformer&zhida_source=entity), etc.)。我挑了其中的

[![](https://pic4.zhimg.com/v2-838cf2a3e8eaddd7b3d275e0c3e8a25b_ipico.jpg)GitHub - srush/Triton-Puzzles: Puzzles for learning Triton​github.com/srush/Triton-Puzzles/](https://link.zhihu.com/?target=https%3A//github.com/srush/Triton-Puzzles/)

和 [LLM Training Puzzles](https://link.zhihu.com/?target=https%3A//github.com/srush/LLM-Training-Puzzles/tree/main) 来学习（摸鱼）一下。

  

我是triton的初学者（给老板丢大脸了），前后花了差不多15-20小时（最后一题3小时 ）才解决。

写这个专栏的目的是为了复习一下当初解题的思路并研究一下有无更好的解法。

[GitHub Jupyter Notebook Solution](https://link.zhihu.com/?target=https%3A//github.com/GarlGuo/My-Triton-Puzzles-solution/blob/master/My_Triton_Puzzles.ipynb)

[Colab Solution](https://link.zhihu.com/?target=https%3A//colab.research.google.com/drive/1jmWvhFo6pXRKyjyhHqiZzvewjomYProj%3Fusp%3Dsharing)  

注意：

1.  这些题并没有真正被 triton 而是被 [triton interpreter](https://link.zhihu.com/?target=https%3A//github.com/Deep-Learning-Profiling-Tools/triton-viz) 执行，然后给出了 memory load & store 可视化界面（不过在我本地是关掉的 因为我实在看不懂这个 load & store viz 到底给debug带来了哪种帮助；我还是用原始的print来debug）
2.  有一些题我在本地做了更广泛的的解法，所以题目+解法本身可能和原始的 [Triton puzzle](https://link.zhihu.com/?target=https%3A//github.com/srush/Triton-Puzzles/blob/main/Triton-Puzzles.ipynb) 不直接兼容（我同时改了 [pytorch ref impl](https://zhida.zhihu.com/search?content_id=253076219&content_type=Article&match_order=1&q=pytorch+ref+impl&zhida_source=entity) 和 triton impl）。我先按我改后的版本过一遍。
3.  我当时写的时候过份依赖于 手算 index broadcasting，因为当时我不知道 tl.make\_block\_ptr。 这导致有些解法很难直接读懂，我后面再改改（我假如早点知道 tl.make\_block\_ptr 后面几道题目至少砍个5小时 ）。
4.  我的解法很可能不是最优的，因为我没规划 [L2 cache optimization](https://zhida.zhihu.com/search?content_id=253076219&content_type=Article&match_order=1&q=L2+cache+optimization&zhida_source=entity) 和其他 memory access pattern。后面我再记录一下最后几题在 A100/H100 上的实际改进。

  

  

## Puzzle 1

![](https://pic2.zhimg.com/v2-682fb22a80d13c6f8f346f5c9e622297_1440w.jpg)

注意这里 block size B0 和 vector size N0 一样

### PyTorch Ref Impl

![](https://pic3.zhimg.com/v2-76d5a471cd58c5a32d85315c8f883420_1440w.jpg)

### [Triton Kernel Impl](https://zhida.zhihu.com/search?content_id=253076219&content_type=Article&match_order=1&q=Triton+Kernel+Impl&zhida_source=entity)

直接 load 一个 0-B0 的 offset 就行，很简单。

（我看不懂为什么还需要一个 program id axis，在 B0 = N0 的情况下）

![](https://picx.zhimg.com/v2-ea79b60872f83cabe835f34d27526387_1440w.jpg)

  

  

## Puzzle 2

![](https://pica.zhimg.com/v2-a0eb48c4053521d9311f9afe539b2b52_1440w.jpg)

这里 B0 &lt; N0，我们要创建一个一维的 thread block

  

### Triton Kernel Impl

每个thread读取 x\_ptr 时 加一个 pid \* B0 loading offset 就行。

![](https://pica.zhimg.com/v2-f814193de15fd98fe993b75b492e6a88_1440w.jpg)

PyTorch Ref Impl 和第一题一样，这里每个thread 要加上 pid \* B0

  

  

## Puzzle 3

![](https://picx.zhimg.com/v2-db015e530805229fd0f070aed625b2ad_1440w.jpg)

注意这里是类似于 outer product 的形式，然后 B0=N0, B1=N1

### PyTorch

![](https://picx.zhimg.com/v2-7384a503127927e95fd3fb1868997247_1440w.jpg)

这里是类似于 外积，不过 x 在 column, y 在 row。我把原题的 32 32 改成了 64 32 以更好的显示出 x 和 y 的区别

### Triton

和 Puzzle1 一样，因为 B0 = N0 & B1 = N1，我就省去了算 thread idx offset 的步骤

注意 z 的 stride 是 (N0, 1)

![](https://pic4.zhimg.com/v2-3e1d1ccc0737a5b33bad9801045e2191_1440w.jpg)

如图所示

  

  

## Puzzle 4

![](https://pic3.zhimg.com/v2-f809c9f131125d3b35ed26c37b0f9bb4_1440w.jpg)

这道题和 Puzzle 3 的区别是 B0 &lt; N0, B1 &lt; N1，然后我们要用两个 thread block axis

### Triton

![](https://pic3.zhimg.com/v2-8b3e1acc76a5c42dc1a9151d243ad7ec_1440w.jpg)

和前一道题的唯一区别是 我们在读 x 和 y 的时候 要按各自的 thread block axis 多算一个offset，挺公式化的

  

  

## Puzzle 5

![](https://pica.zhimg.com/v2-acccdb6606572cc4b41c83e530c47de2_1440w.jpg)

和 Puzzle 4 的唯一区别是 x + y 变成了 relu(x \* y)

### Triton

![](https://pic2.zhimg.com/v2-5ce95434e3201e7aa65a15cd72a2142d_1440w.jpg)

我们只需要相应地改变 最后一行 计算的一步 就行

  

  

## Puzzle 6

![](https://picx.zhimg.com/v2-075a3f90a69c9952dd9cfc48bfe78b75_1440w.jpg)

我们要算 dx，注意这里 X: (N1, N0), Y: (N1), Z: (N1, N0)

### PyTorch

![](https://pic2.zhimg.com/v2-1bc38b06a7666c9570647ae21b8fb575_1440w.jpg)

PyTorch的实现比较清晰，但要注意 row, col 在这里分别对应 N1, N0 (这个写法确实很怪，但下面的函数调用是这样要求的）

PS: 我当时卡在这里20分钟琢磨为啥 (N0, N1) 的写法不work，才发现这里其实是 (N1, N0)

![](https://pica.zhimg.com/v2-ab8c5ea085ae3b17101bbc6cfce93940_1440w.jpg)

可以看出 X: (90, 100) 对应 (N1, N0)

### Triton

![](https://picx.zhimg.com/v2-2fbfbe58c113d6496cdc36c9f8094aab_1440w.jpg)

关键的计算步 tl.where 直接解决，其他的 load & store 和 Puzzle 5 类似

  

  

## Puzzle 7

![](https://pic2.zhimg.com/v2-d02b941bf0ef720b9de24578bfc423a3_1440w.jpg)

1维grid，要实现 .sum(dim=1)

### PyTorch

![](https://picx.zhimg.com/v2-aac7c89a4c74b832a0bc309b7183a76f_1440w.jpg)

### Triton

我们需要循环读 ceil(T/B1) 个 column block，每次读 (B0, B1) 个元素然后累加到 acc (B0) 最后写回 z 就行

![](https://pica.zhimg.com/v2-bfb68b7df42ea74389b31bdb0445ae68_1440w.jpg)

acc: (B0,), 每次读 (B0, B1) 个 x 元素并写入结果到 acc，循环的时候算好 col\_mask 就行

  

  

## Puzzle 8

![](https://pic3.zhimg.com/v2-0151f22c2bd9d231c8856286eadde1d0_1440w.jpg)

这里用两次 loop 要用到 online softmax trick

### PyTorch

![](https://pic1.zhimg.com/v2-58bd11bcf11991d24e8203b1b1dbee0c_1440w.jpg)

标准的 numerically stable softmax: 为了数值稳定性在算 exp(x) 前 分子分母同时减掉 max(x, dim=1)

### Triton

如题所示，naive 的实现要3个loop的原因：

-   Loop 1: 算 max(x, dim=1)
-   Loop 2: 算 sum(exp(x - x\_max))
-   Loop 3: 算真正的 softmax

伪代码长这样

![](https://pic1.zhimg.com/v2-5fca39220268e4bab82df387aa484156_1440w.jpg)

我们要破局的关键在于把 loop1 和 loop2 合并起来，“动态地”计算 safe row\_sum\_exp

这里我们参考 [online softmax](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1805.02867)

![](https://pica.zhimg.com/v2-6385b293cc6182f375cc92667747ad6a_1440w.jpg)

破局关键：第一次循环的时候 用新的 row\_max 来更新旧的 row\_sum\_exp

解法其实挺直接的

![](https://picx.zhimg.com/v2-f811e3e5493c267f4bd2a7b517cffacd_1440w.jpg)

最关键的就 Loop 1 的 online update of row\_sum\_exp 这一行，然后 Loop 2 算算softmax结果就行

  

  

## Puzzle 9

![](https://pic4.zhimg.com/v2-48b33ef4afd5e2cc152a6db8cfe273c9_1440w.jpg)

老板的FA1终极简化版lol，这里我们要用单线程，然后顺序遍历整个 sequence 来算 attention （不包含O）。同时 embed dim = 1

### PyTorch

![](https://pic1.zhimg.com/v2-f869840201aa1cddfed9fbc2ff43fbf4_1440w.jpg)

简化版的FA1: 注意这里 q k v 是 (T) 向量而不是 (T, d) 矩阵

### Triton

![](https://pic3.zhimg.com/v2-bfc19b42f5985986ea4684b56aae1c26_1440w.jpg)

把 row\_qkv\_sum 的在线更新搞对就行，最后直接写回 z。其他的基本照抄 pytorch 版本就行了lol

  

## Puzzle 10

![](https://pica.zhimg.com/v2-319f971ea3262dbe08cdef537b9ca336_1440w.jpg)

2D Convolution，其中 #channel = 1。一维 grid 实现 data-parallel (batching)

题目给定的超参:

-   图片: (N0, H, W)
-   Conv kernel: (kW, kH)
-   Thread block size: B0

### PyTorch

![](https://pic2.zhimg.com/v2-e4b1b527e1aac00c9818f290bc1e6017_1440w.jpg)

注意 PyTorch 实现需要 padding，但 triton 直接 tl.load(..., other=0) 可以解决

### Triton

![](https://pica.zhimg.com/v2-5d0282bfaf65a936947f93eb5a3ff496_1440w.jpg)

这里 tl.make\_block\_ptr 可以更易懂，关键在于 (1) 几处 loading 的 offset 算对 (2) padding 通过正确的 mask + other=0 解决

  

  

## Puzzle 11

PS: 这道题我花了3小时。我第一次写的时候手动画了一个 (2, 2, 2) @ (2, 2, 2) 的矩阵乘法来手动验证所有indices是否正确 最有耐心的一集

![](https://picx.zhimg.com/v2-94eb3c031ed62fb625bec1c801899239_1440w.jpg)

矩阵乘法（这里是 BMM）我们先回顾一下 triton tutorials 的部分

**注意**这里我们有一个 B\_MID 超参，我们也需要对 intermediate dot prod 累加到 C 里面（遍历 ceil(MID/B\_MID)次 ）

[](https://link.zhihu.com/?target=https%3A//triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)

如图所示：

-   我们用一个loop来遍历 MID，每次累加到一个 acc 里面。这是 红色箭头 的意思（真正要实现的）。
-   我们对 batch, x\_row, y\_col 来并行，这是 蓝色箭头的意思。

![](https://pic3.zhimg.com/v2-28a0b84cae9033f38e90b817ad0bb96a_1440w.jpg)

给定 3维grid (batch, x\_row, y\_col)，我们只需一个loop来遍历 MID，如图中的红色箭头的移动方向 并且累加到 C 就行

  

同时我们注意题目给出的超参涵义:

我们有一个 3维 grid 规定了：

-   N2 -> batch size
-   N0 -> size of x row
-   N1 -> size of y col

我们 triton kernel 内部需要遍历：

-   MID -> intermediate dot product size

### Triton

![](https://picx.zhimg.com/v2-f97be06957b62f27ef0dd29167367bef_1440w.jpg)

tl.make\_block\_ptr 应该会简化不少。这道题想明白 我要读什么，我要怎么遍历 （上面的红色蓝色箭头）就迎刃而解了

  

Bonus question:

-   _**假设我们是4维 grid（也对 MID 并行），我们要改哪两步呢?**_

  

  

### Puzzle 12

PS：这道题我花了3小时，其中 2.5 小时在算/检查 indices 有无错误上，基本上 indices 算对后整道题就写出来了。我 triton 解法全是 indices 的原因是为了方便 debug。这次我吸取了我 puzzle 11 的教训变得更有耐心（折磨自己 ）的一集

  

**注意** 这道题的 PyTorch ref impl 我修改成了更普遍的情况（因为原先 MID = 64 = 8 group of 8，我改成了 MID = 128 = 16 group of 8）这样 # group in mid 16 != group size 8

![](https://pic4.zhimg.com/v2-a30d174b576029cd03da7f965c471eb3_1440w.jpg)

我们要实现 MM (W @ A)，不过 W 是 Int4，我们要先 dequantize 到 FP32

如图，我们要算 Q(W) @ A

W 是沿着 col (mid) 进行的 uniform int4 quant.

**每8个W param 作为一个group，每个group 配备一个 Int4 的 shift 和 一个 FP32 的 scale**

我们要对 weight 和 shift 同时dequantize，然后乘以 scale 来复原回原先 W 的 block

  

### PyTorch ref impl

很直观

![](https://pica.zhimg.com/v2-2764fd3147c60970cbf7612275581b92_1440w.jpg)

我第一次读也花了点时间，不过其实挺直观的

### Triton kernel

具体的分析懒得写了 白纸上走一遍流程就行，写起来都是些很无聊的代码。

![](https://pic1.zhimg.com/v2-da8b30d3085fd6ff613b106d92d4d96a_1440w.jpg)

大部分代码在算正确的indices，少部分才是核心的 dequantize，最后一行才是 W @ A

Triton Puzzles 也全部解决了

  

  

摸鱼系列还有一篇:

[](https://zhuanlan.zhihu.com/p/20265169815)

美好的摸鱼总是短暂的，转眼间寒假就过去了lol