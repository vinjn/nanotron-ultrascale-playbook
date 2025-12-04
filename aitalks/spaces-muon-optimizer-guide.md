[Muon优化器指南：快速上手与关键细节](https://spaces.ac.cn/archives/11416)
==========================================================

By 苏剑林 | 2025-11-19 | 4456位读者 | 

这段时间，相信很多读者已经刷到过Muon优化器的相关消息。实际上，Muon的提出时间大致是去年的10月份，由 [Keller Jordan](https://x.com/kellerjordan0/status/1842300916864844014) 在推特上提出，距今也不过一年多一点。然而，就在这一年里，Muon已经经历了百亿、千亿乃至万亿参数模型的训练考验，足以表明它是一个相当有竞争力的优化器。

如今，Muon已经内置在[Torch](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html)、[Keras](https://keras.io/api/optimizers/muon/)等训练框架中，就连[Megatron](https://github.com/NVIDIA/Megatron-LM/blob/dev/megatron/core/optimizer/muon.py)这样的大型框架也逐渐开始支持，这意味它已经获得了业界的普遍认可。不过，对于仅熟悉Adam的读者来说，如何快速有效地切换到Muon，可能依然是一件让人困惑的事情。所以，本文试图给出一个快速上手教程。

简要介绍 [#](https://spaces.ac.cn/archives/11416#%E7%AE%80%E8%A6%81%E4%BB%8B%E7%BB%8D)
----------------------------------------------------------------------------------

Muon的正式提出者是 [Keller Jordan](https://x.com/kellerjordan0/status/1842300916864844014) ，目前任职于OpenAI。开头说了，Muon最早发表在推特上，而直到现在，作者也只是多写了篇博客[《Muon: An optimizer for hidden layers in neural networks》](https://kellerjordan.github.io/posts/muon/)而不是一篇Paper，作者的观点是“是否写成Paper，跟优化器是否有效，没有任何关系\[[原文](https://x.com/kellerjordan0/status/1890178773586489716)\]”。

Muon是一个专门为矩阵参数定制的优化器，也有一些相关工作具有类似的特点，比如[Shampoo](https://papers.cool/arxiv/1802.09568)，还有更早一些的[Stochastic Spectral Descent](https://spaces.ac.cn/archives/10592)，等等。很多工作或多或少都能关联上Muon，但没有一个是能够完全覆盖Muon的，所以在笔者看来Muon算是一个全新的工作。

在国内，最早向大家科普Muon的文章，应该是笔者的博客[《Muon优化器赏析：从向量到矩阵的本质跨越》](https://spaces.ac.cn/archives/10592)，而首次在较大规模的模型上验证Muon，应该是我们二月份发布的[Moonlight](https://papers.cool/arxiv/2502.16982)，其所提的Moonlight版Muon，用到了后来的万亿参数的[K2](https://papers.cool/arxiv/2507.20534)中。K2之后，[GLM-4.5](https://papers.cool/arxiv/2508.06471)同样用到了这个Muon变体。

跟Muon的作者之一Jeremy Bernstein在他的博客[《Deriving Muon》](https://jeremybernste.in/writing/deriving-muon)所说的一样，对笔者而言，Muon的独特之处在于它可以基于更本质的优化原理推导而来，并在实践中有效，相比之下，虽然Adam也很有效，但它更像是一种启发式方案。

四个版本 [#](https://spaces.ac.cn/archives/11416#%E5%9B%9B%E4%B8%AA%E7%89%88%E6%9C%AC)
----------------------------------------------------------------------------------

本文不打算介绍Muon的数学细节，也不打算介绍Muon的实现，而是主要介绍从Adam切换到Muon的一些技术细节和注意事项。刚才说了，Muon是专用于矩阵参数优化的，并且是非Element-wise的更新规则，这可能导致新用户上手起来会比较困惑。

还有，据笔者所知，Muon目前至少有四个略微不同的版本，这种多版本现象也促进了这种困惑。如果用户不了解其中的细节，可能会因为调错超参数（特别是学习率）而导致不好的效果，下面会着重理清楚这些内容。首先，对于一个矩阵W∈Rdin×dout，G是它的梯度，四个Muon变体分别是：

Mt\=βMt−1+GtWt\=⎧⎩⎨⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪Wt−1−ηt(msign(Mt)+λWt−1)Wt−1−ηt(max(1,dout/din)−−−−−−−−−−−−−√msign(Mt)+λWt−1)Wt−1−ηt(dout/din−−−−−−√msign(Mt)+λWt−1)Wt−1−ηt(0.2×max(dout,din)−−−−−−−−−−−√msign(Mt)+λWt−1)(朴素版)(KellerJordan版)(MuP版)(Moonlight版)

如果要开Nesterov动量则将msign(Mt)换成msign(βMt+Gt)，其中msign在实现中通常以`zeropower_via_newtonschulz`命名，具体实现细节普通用户可以不管。

四个版本的唯一区别是msign前的缩放因子，其中“KellerJordan版”和“MuP版”大同小异，“Moonlight版”则稍微特别一点。Keras只实现了“KellerJordan版”，而Torch则实现了“KellerJordan版”和“Moonlight版”，朴素版目前似乎比较少见，对笔者来说常用的是自己写的“MuP版”。

两个维度 [#](https://spaces.ac.cn/archives/11416#%E4%B8%A4%E4%B8%AA%E7%BB%B4%E5%BA%A6)
----------------------------------------------------------------------------------

这里我们要注意一个重要细节，“KellerJordan版”和“MuP版”对din,dout的顺序是敏感的，所以第一件事情就是要搞清楚din,dout的含义，并不是说矩阵的第一个维度就一定是din、第二个维度就一定是dout。

din、dout的含义分别是线性层的输入、输出维度，所以哪个是din哪个是dout，要看线性层的具体实现。比如Keras的Dense层实现是xW，那么矩阵W的第一个维度是din、第二个维度是dout；然而，Torch的Linear层实现的事xW⊤，所以矩阵W的第二个维度是din，第一个维度才是dout。

所以，如果要实现“KellerJordan版”的Muon，对于Torch的Linear层，缩放因子应该是`max(1, W.shape[0]/W.shape[1])**0.5`，而对于Keras则应该是`max(1, W.shape[1]/W.shape[0])**0.5`。所以，当前Keras的Muon实现其实是不正确的，因为它照搬了Torch的缩放因子实现（[源码](https://github.com/keras-team/keras/blob/v3.12.0/keras/src/optimizers/muon.py#L198)）。

如果是自己写的模型，则需要根据自己的写法谨慎判断了，比如不排除在Torch中混用了自带的Linear层和手写的`x @ W`，这样就不能一概而论是`W.shape[0]/W.shape[1]`还是`W.shape[1]/W.shape[0]`了。当然，如果你嫌搞清楚这些比较麻烦，那就可以考虑用“Moonlight版”，它的缩放因子关于din,dout是对称的。

超参设置 [#](https://spaces.ac.cn/archives/11416#%E8%B6%85%E5%8F%82%E8%AE%BE%E7%BD%AE)
----------------------------------------------------------------------------------

搞清楚din,dout后，剩下就是学习率ηt和权重衰减系数λ怎么设置了。这里的假设是用户已经有Adam的调参经验，在Adam下已经得到了不错的效果，想要快速迁移到Muon体验一波。

我们先来看“Moonlight版”，它的缩放因子是通过对齐Adam的Update RMS得到的，如果想要了解细节，可以参考[《Muon续集：为什么我们选择尝试Muon？》](https://spaces.ac.cn/archives/10739)，至于0.2这个“Magic Number”，可以参考[《为什么Adam的Update RMS是0.2？》](https://spaces.ac.cn/archives/11267)。简单来说，“Moonlight版”Muon对齐了Adam的更新幅度，所以从Adam迁移过来的最简单的做法是：**啥也不用改**，用回Adam的ηt和λ就行。

然后看剩余三个版本。我们知道，主流模型通常有个hidden\_size（记为d），模型的矩阵形状多数不会明显偏离d×d，因此我们以din\=dout\=d来近似处理，此时这三个版本都是一样的，相比“Moonlight版”则少了个0.2d−−√。既然“Moonlight版”对齐了Adam更新幅度可以不改变超参，那么这三个版本的学习率应该要放大0.2d−−√倍，才能对齐Adam的更新幅度，相应地，λ则要除以0.2d−−√。

代入d\=1024,2048,4096，结果分别是6.4,9,12.8，如果记不住0.2d−−√，那么可以简单记住，如果我们使用另外三个版本的Muon，那么**直接将Adam的学习率放大10倍**来作为Muon的学习率。如果直接代入Adam的学习率到Muon中，就会因为欠拟合得到Muon远不如Adam的结论，据笔者所知，Muon的一些差评便来源于此。

这样看还是“Moonlight版”更好用？“Moonlight版”确实有不错的实践效果，但如果就此说它更好用，其实是站在Adam的角度下评价了。“MuP版”或“KellerJordan版”的好处是学习率可迁移，即在小模型调好学习率后，直接用到大模型往往也有不错的效果，这部分可以参考Jeremy Bernstein的博客[《Deriving Muon》](https://jeremybernste.in/writing/deriving-muon)或笔者的博客[《高阶MuP：更简明但更高明的谱条件缩放》](https://spaces.ac.cn/archives/10795)

其他参数 [#](https://spaces.ac.cn/archives/11416#%E5%85%B6%E4%BB%96%E5%8F%82%E6%95%B0)
----------------------------------------------------------------------------------

如果说Muon只管矩阵参数，那么其余参数怎么办呢？比如线性层的Bias项、RMSNorm的gamma项，这些是1维的参数；又比如卷积层可能出现3维、4维数组的参数。

这里先要更正一下，Muon并不是只管矩阵参数，Muon是只管“**稠密输入的线性层的矩阵参数**”，如果读者觉得这个比较费解，那么只需要记住Embedding层和最后的分类层（包括GPT的LM Head）的矩阵参数都不能用Muon，否则效果会明显差。这些不能用Muon的矩阵参数，还有1维、3维及更高维的参数，如果读者不想费太多心思，那么直接用Adam就行，基本上Muon实现都是混合了Adam的，用户可选某些层用Adam。

如果读者愿意捣鼓，那么像卷积层的3、4维参数，也可以用上Muon。以Conv2D为例，卷积核形状通常是(w,h,din,dout)，它的等效实现其实是把(w,h,din)的Patch输入展平为w×h×din的向量，然后卷积核也Reshape为(w×h×din,dout)，最后做矩阵乘法，所以它想要用Muon，那就要先将动量Reshape为(w×h×din,dout)，计算msign后再Reshape回去更新。

类似地，还有RMSNorm的gamma参数，可以视为跟对角矩阵做乘法，于是将它的动量视为对角矩阵，也可以算msign，结果等效于SignSGDM；Embedding层可以视为多个(1,d)矩阵去计算msign，结果是Normalized SGDM（参考[《Muon优化器赏析：从向量到矩阵的本质跨越》](https://spaces.ac.cn/archives/10592)）。如果还想折腾，比如Multi-Head的Attention，每个Head的投影矩阵是不是可以考虑单独拿出来分别做msign...

生命不息，折腾不止～

期望结果 [#](https://spaces.ac.cn/archives/11416#%E6%9C%9F%E6%9C%9B%E7%BB%93%E6%9E%9C)
----------------------------------------------------------------------------------

最后，如果用户按照上述说明正确设置并跑起来了，那么就可以开始祈祷幸运之神的降临了。

我们应该期待一个什么样的结果呢？如果没有出现梯度爆炸等异常情况，那么多数情况下Muon会比Adam略好一些，当然也不排除某些情况Muon会略差，但不论如何，它们的差距不会非常大。如果出现一方比另一方好非常多的现象，那么可能得反思一下哪边的设置出现问题了。

不过，这都不是绝对的，比如某些极端的设置下，确实也会出现Muon比Adam好很多，Adam怎么调也不行的现象。总之，祝你幸运。如果出现有意思的现象，欢迎一起交流分析。

_**转载到请包括本文地址：**[https://spaces.ac.cn/archives/11416](https://spaces.ac.cn/archives/11416 "Muon优化器指南：快速上手与关键细节")_

_**更详细的转载事宜请参考：**_[《科学空间FAQ》](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8 "《科学空间FAQ》")

**如果您还有什么疑惑或建议，欢迎在下方评论区继续讨论。**

**如果您觉得本文还不错，欢迎[分享](https://spaces.ac.cn/archives/11416#share)/[打赏](https://spaces.ac.cn/archives/11416#pay)本文。打赏并非要从中获得收益，而是希望知道科学空间获得了多少读者的真心关注。当然，如果你无视它，也不会影响你的阅读。再次表示欢迎和感谢！**

打赏

**如果您需要引用本文，请参考：**

苏剑林. (Nov. 19, 2025). 《Muon优化器指南：快速上手与关键细节 》\[Blog post\]. Retrieved from [https://spaces.ac.cn/archives/11416](https://spaces.ac.cn/archives/11416)

@online{kexuefm-11416,  
        title={Muon优化器指南：快速上手与关键细节},  
        author={苏剑林},  
        year={2025},  
        month={Nov},  
        url={\\url{https://spaces.ac.cn/archives/11416}},  
}