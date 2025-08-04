# Stable Diffusion 模型演进：LDM、SD 1.0, 1.5, 2.0、SDXL、SDXL-Turbo 等

**作者：** AI闲谈

---

一、背景

这里我们继续介绍 Stable Diffusion 相关的三个图像生成工作，Latent Diffusion Model（LDM）、SDXL 和 SDXL-Turbo。这三个工作的主要作者基本相同，早期是在 CompVis 和 Runway 等发表，后两个主要由 Stability AI 发表。

LDM 对应的论文为：[2112.10752] High-Resolution Image Synthesis with Latent Diffusion Models

LDM 对应的代码库为：High-Resolution Image Synthesis with Latent Diffusion Models

SDXL 对应的论文为：[2307.01952] SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis

SDXL 对应的代码库为：Generative Models by Stability AI

SDXL-Turbo 对应的论文为：[2311.17042] Adversarial Diffusion Distillation

SDXL-Turbo 对应的代码库为：Generative Models by Stability AI

如果之前没有了解过 Diffusion Model，建议可以阅读 Jay Alammar 的 The Illustrated Stable Diffusion – Jay Alammar。

## 文本生成图相关总结也可参考：

1. [文生图模型演进：AE、VAE、VQ-VAE、VQ-GAN、DALL-E 等 8 模型](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247485323&idx=1&sn=4408ac639f54f87c62cb64503cc2e9d9&chksm=c364c0cef41349d8f7a0c2d388b3de7bdfef049c8024b09e382e20a8e337e7c7acbca7b0a8e7&scene=21#wechat_redirect)
2. [OpenAI 文生图模型演进：DDPM、IDDPM、ADM、GLIDE、DALL-E 2、DALL-E 3](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247485383&idx=1&sn=13c638d36899e6b3f8935be850b8ba79&chksm=c364c082f4134994d7672f4c35d5044b7271ec9978ac6f4fc5015da01f10f5388d4983c1deaa&scene=21#wechat_redirect)3. [Google 图像生成模型 ViT-VQGAN & Parti-20B](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247485482&idx=1&sn=d508b9e561db18763d6abe7860246cb0&chksm=c364cf6ff4134679717c65ed5e4baf9f927c048e68948aa05920c0dd001e3b2116147c5678dd&scene=21#wechat_redirect)4. [Google 图像生成模型 MaskGIT & Muse, 并行解码 60 倍加速](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247485483&idx=1&sn=eaa53acbeb203f9d010a4506ca9ac6bc&chksm=c364cf6ef4134678db1caba5efa5c5a615a3804f796975349ed2c070431c98a6b3905dc57109&scene=21#wechat_redirect)5. [Google 最强文生图模型 Imagen & Imagen 2](http://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247485527&idx=1&sn=d9ef1fed4ad899cf220ccca16c841433&chksm=c364cf12f41346048b6b1aa63e21bf0d1ccca3516e94667c8923511e9c4d6c8516572579b722&scene=21#wechat_redirect)
![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWFudehqGHgmfxTy4R9tyvmUx7UanvabkS5IQYbboCpdO7KstDMKJicFA/640?wx_fmt=png&from=appmsg&randomid=3zmfmusl)

## 二、摘要

2021 年 05 月 OpenAI 发表 Diffusion Models Beat GANs，扩散模型（Diffusion Model，DM）的效果开始超越传统的 GAN 模型，进一步推进了 DM 在图像生成领域的应用。

不过早期的 DM 都直接作用于像素空间，因此如果要优化一个强大的 DM 通常需要花费数百 GPU 天时，并且因为需要迭代多步，推理的成本也很高。为了实现在有效的计算资源上训练 DM，同时保持其质量和灵活性，作者提出将 DM 应用于强大的预训练 AutoEncoder 的隐空间（Latent Space），这也就是为什么提出的模型叫 LDM。与以前的工作相比，这种方式训练 DM 首次实现了在降低复杂性和保留细节之间的平衡，并大大提高视觉的逼真度。

此外，作者还在模型中引入交叉注意力层，可以将文本、边界框等条件很方便地引入到模型中，将 DM 转化为强大而灵活的生成器，实现高分辨率的生成。作者提出的 LDM 模型同样在图像修复、类别条件生成等方面取得很好的效果，同时与基于像素空间的扩散模型相比，大大降低计算要求。

如下图 Figure 5 所示为其文本引导图像生成的结果：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWHlz6eicdLUkOJ6IBkTnEI9x3ONId8IDWibVAuyzW42X3DscyalM5asXA/640?wx_fmt=png&from=appmsg&randomid=qfq92rdz)

在 Stable Diffusion（LDM）的基础上，SDXL 将 U-Net 主干扩大了三倍：模型参数增加主要是使用了第二个 Text Encoder，因此也就使用更多的 Attention Block 和 Cross Attention 上下文。此外，作者设计了多分辨率训练方案，在多个不同长宽比的图像上训练。作者还引入了一个细化模型，用于进一步提升生成图像的视觉逼真度。结果表明，与之前版本的 Stable Diffusion 相比，SDXL 的性能有了显著提升，并取得与其他非开源模型相当的效果。此次的模型和代码同样完全开源。

如下图所示为 SDXL 生成的图像：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWs2tX5micxpz1xcWuCrWNRkfYDlrHzoVYreNt0KR7DyeN7eHAINkDmTw/640?wx_fmt=png&from=appmsg&randomid=6ncpry5s)

在 SDXL 的基础上，作者提出了对抗性扩散蒸馏技术（Adversarial Diffusion Distillation，ADD），将扩散模型的步数降低到 1-4 步，同时保持很高的图像质量。结果表明，模型在 1 步生成中明显优于现有的几步生成方法，并且仅用 4 步就超越了最先进的 SDXL 的性能。训练出的模型称为 SDXL-Turbo。

如下图 Figure 1 所示为 SDXL-Trubo 生成的图像：![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUW5UgtD5gHhHaGVmfyPuNFt4hibKWV66BScHaJ7Z1Va3pYfKQOJmzzvSA/640?wx_fmt=png&from=appmsg&randomid=f61frf1r)

## 三、Latent Diffusion Model（LDM）

### 3.1. LDM 模型概览

LDM 和其他扩散生成模型结构类似，整体来说包含三个组件：

- Auto Encoder：下图左侧部分，包含红框的 Encoder 和蓝框的 Decoder，其中 Encoder 主要用于训练中生成 target z，推理阶段不需要。而 Decoder 用于从隐空间编码（latent code）恢复出图像。
- Conditioning：下图右侧部分，用于对各种条件信息进行编码，生成的 embedding 会在扩散模型 U-Net 中使用。不同的条件可能会有不同的 Encoder 模型，也有不同的使用方式（对应下图中的 switch），比如：
- 对于文本类型条件，可以使用 Bert Encoder，也可以使用 CLIP 中的 Text Encoder 将文本编码为 embedding。
- 对于图像类型条件，比如图像修复、分割条件，可以将其编码后与噪声 Concat 作为输入，而不是通过 Attention 机制交叉。
- Denoising U-Net：下图中间部分，用于从随机噪声 zT 中通过几步迭代生成 latent code，然后使用 Decoder 恢复出图像。其中的各种条件信息都会通过 Cross Attention 进行交叉融合。需要说明的是，U-Net 生成的目标是 x 经 Encoder 编码后的 embedding，通常也称为隐向量，而不是直接生成图像像素，因此说 U-Net 是作用在隐空间（Latent Space）。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWCtShyMcdcUaa7H7iaRIWTIHCcR8LibDyUaMXRQBPM42z8pq9hdibGqicmg/640?wx_fmt=png&from=appmsg&randomid=77xwuw80)

### 3.2. AutoEncoder

AutoEncoder 中的 Encoder 用于对图像 x 进行压缩，假设输入图像分辨率为 HxW，则 f 压缩率对应的 latent code 大小为 H/f x W/f。也就是说，如果图像分辨率为为 512x512，则 f=4 的压缩率对应的 latent code 大小为 64x64，也就是 z 的大小为 64x64。针对不同的压缩率 f，作者也进行了一系列实验，对应的模型为 LDM-{f}，总共有 LDM-1，LDM-2，LDM-4，LDM-8，LDM-16，LDM-32。需要说明的是，LDM-1 相当于没有压缩，也就是直接作用于像素空间，LDM-32 相当于 32 倍压缩，512x512 分辨率图像对应的 latent code 只有 16x16。

作者在类别条件生成任务上对几种压缩比进行了实验验证，如下图 Figure 6 所示，可见 LDM-4、LDM-8、LDM-16 获得最好的平衡。LDM-32 的压缩率太高，反而影响了生成质量：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWIVia6fgyeIsuEGzarWMCqCia0ib60bVSiaJz3oYoGywn7G8icTEibdYUlERA/640?wx_fmt=png&from=appmsg&randomid=hgbwvlk7)

如下图 Table 6 所示，作者同样在图像修复任务上验证了不同压缩率、Cross Attention 的影响，可以看出 LDM-4 的训练、推理吞吐相比 LDM-1 有明显提升，并且 Attention 对吞吐的影响也不大。同时 LDM-4 还获得更好的效果（更低的 FID）：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWv9MZ1ib38icENtqy3Oic47V7Zb5Uwj96ARbdWibvFGkIYboz02FeljZhbg/640?wx_fmt=png&from=appmsg&randomid=26715f23)

### 3.3. Latent Diffusion Models

本文中作者使用的 U-Net 模型是基于 OpenAI Diffusion Models Beat GANs 中的 Ablated U-Net 修改而来，具体来说是将其中的 Self-Attention 替换为 T 个 Transformer block，每个 block 中包含一个 Self-Attention，一个 MLP 和一个 Cross-Attention，如下图所示，其中的 Cross Attention 就是用于和其他条件的 embedding 进行交叉融合：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWSnUDpuDe2JFqUH4cjYHdtSIxBVEBzAbzRu1AibibaFpwTpXSzfNysY9A/640?wx_fmt=png&from=appmsg&randomid=w421rc3y)

### 3.4. Conditioning 机制

LDM 支持多种条件类型，比如类别条件、文本条件、分割图条件、边界框条件等。

对于文本条件，可以使用常用的文本 Encoder，比如 Bert 模型，或者 CLIP 的 Text Encoder，其首先将文本转换为 Token，然后经过模型后每个 Token 都会对应一个 Token embedding，所以文本条件编码后变为一个 Token embedding 序列。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUW7vsR9a6cuEpo8aHAau15TgEAOMF47tiaaESthMHamT5IgVtG6GfWH9w/640?wx_fmt=png&from=appmsg&randomid=k1al7r0w)

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWAGND8ibpsYdfJH8ptblckvkr8IPMLKSMMlN8TW5Yt1SgYrP3wNYzk3g/640?wx_fmt=png&from=appmsg&randomid=z4bumd2l)

对于 layout 条件，比如常见的边界框，每个边界框都会以（l,b,c）的方式编码，其中 l 表示左上坐标，b 表示右下坐标，c 表示类别信息。![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUW3p3zaJkwoXEdbV6hHaZFcLfqictcQRMDYTIN7WpetUYITkSE3pfrrDQ/640?wx_fmt=png&from=appmsg&randomid=d93zu7yh)

对于类别条件，每个类别都会以一个可学习的 512 维向量表示，同样通过 Cross-Attention 机制融合。![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWXRZs1H47vg7aicksNjyufqicUAKjzAFzZ8JXLNMG4bR1cF6WjBYOibcIQ/640?wx_fmt=png&from=appmsg&randomid=cff2ejxe)

对于分割图条件，可以将图像插值、卷积后编码为 feature map，然后作为条件。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWqO2LPZbx5x3VZiaOb2DbUicPg8MJo4hJ6FHrbMy2VojdcIsaeNzSjp7w/640?wx_fmt=png&from=appmsg&randomid=biysfpvj)

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWsKsshlARBFsCLDYiczlxGLRRGDBibvfTWZoaemuhk4HycAzHvVMqdVNw/640?wx_fmt=png&from=appmsg&randomid=5a4hcew2)

其中文本条件和 layout 条件都通过 Transformer Encoder 编码，对应的超参如下图 Table 17 所示，也就是文本最多只能编码为 77 个 Token，Layout 最多编码为 92 个 Token：![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWVZlh85h818SicvaWJRCpiaVxlHC1vk7JcensvwlWBC3xZfZnmYibSVdgQ/640?wx_fmt=png&from=appmsg&randomid=lkxtgjkb)

所谓的 layout-to-image 生成如下图所示，给定多个边界框，每个边界框有个类别信息，生成的图像要在对应的位置生成对应的目标：![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWFBOHFuwWlmwGgtEXzMWQwsH23a32haxZDRCjeNxp3J3IuRQaWnbVQA/640?wx_fmt=png&from=appmsg&randomid=r55q1f3g)

### 3.5. 实验结果

#### 3.5.1. 无条件生成

如下图 Table 1 所示，作者在多个任务上评估了 LDM-4 和 LDM-8 的无条件图像生成效果，可以看出，在大部分任务上都获得了很不错的结果：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUW8y8TjPMmJsNX1aAskOhu17RtpTx67e5nm3G8qLCcZJHvJGF3d2dkIw/640?wx_fmt=png&from=appmsg&randomid=5869yh73)

#### 3.5.2. 类别条件生成

如下图 Table 3 所示，作者同样在 ImageNet 上与 ADM（Diffusion Model Beat GANs）等模型进行了类别条件图像生成对比，可见在 FID 和 IS 指标上获得了最优或次优的结果：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWW2P8mkkFjRdUgEjxltg7nCTN9uGlk0hDXiaSZ6nSUtPacykib8GibLRRg/640?wx_fmt=png&from=appmsg&randomid=6b9s5zod)

#### 3.5.3. LDM-BSR

作者同样将 BSR-degradation 应用到超分模型的训练，获得了更好的效果，BSR degradation Pipeline 包含 JPEG 压缩噪声、相机传感器噪声、针对下采样的不同图像插值方法，高斯模糊核以及高斯噪声，并以随机顺序应用于图像（具体可参考代码 https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/image_degradation/bsrgan_light.py），最终获得了不错的效果：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWRKRNkkfaNWRLpn39UsSxuaXDfvWjZcJBuXicQCMZrUHCaiaO7kb67drA/640?wx_fmt=png&from=appmsg&randomid=ffq271ha)

### 3.6. 计算需求

作者与其他模型对比了训练和推理的计算需求和相关的参数量、FID、IS 等指标，提出的模型在更小的代价下获得更好的效果：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUW5pmsWmrn2XPicUPd7sK4dv2yqXcWcIic2HQ2m50kI3o5bkpvY9YASGkg/640?wx_fmt=png&from=appmsg&randomid=6ld6nhd9)

## 四、SDXL

### 4.1. SDXL 模型概览

如下图所示，SDXL 相比 SD 主要的修改包括（模型总共 2.6B 参数量，其中 text encoder 817M 参数量）：

- 增加一个 Refiner 模型，用于对图像进一步地精细化
- 使用 CLIP ViT-L 和 OpenCLIP ViT-bigG 两个 text encoder
- 基于 OpenCLIP 的 text embedding 增加了一个 pooled text embedding

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWyTZ35QrLCqBmmxibjoKYiaNpRU9g4ZfYw3ykuyMkCazicyWAB40t8UXVA/640?wx_fmt=png&from=appmsg&randomid=iai3pbfs)

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWVdT8ia1k6nxCLCeibbiaB52ODlBaF9tyggMCrouXASIVbNiawLbFq8f1LA/640?wx_fmt=png&from=appmsg&randomid=qrv34d2c)

### 4.2. 微条件（Micro-Conditioning）

#### 4.2.1. 以图像大小作为条件

在 SD 的训练范式中有个明显的缺陷，对图像大小有最小长宽的要求。针对这个问题有两种方案：

- 丢弃分辨率过小的图像（例如，SD 1.4/1.5 丢弃了小于 512 像素的图像）。但是这可能导致丢弃过多数据，如下图 Figure 2 所示为预训练数据集中图像的长、宽分布，如果丢弃 256x256 分辨率的图像，将导致 39% 的数据被丢弃。
- 另一种方式是放大图像，但是可能会导致生成的样本比较模糊。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWar4H2ug7EEWSxgnfH5KOiap5IAoNRuKYFibCMTq6kY7sLAvwDqQgX9UA/640?wx_fmt=png&from=appmsg&randomid=y58rs584)

针对这种情况，作者提出将原始图像分辨率作用于 U-Net 模型，并提供图像的原始长和宽（csize = (h, w)）作为附加条件。并使用傅里叶特征编码，然后会拼接为一个向量，把它扩充到时间步长 embedding 中并一起输入模型。

如下图所示，在推理时指定不同的长宽即可生成相应的图像，（64,64）的图像最模糊，（512, 512）的图像最清晰：![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWDoU3IicyAdFibLXuzibKyZica8KdQCOAKXN2coRAuVNLINrpGibHbGp683w/640?wx_fmt=png&from=appmsg&randomid=nv4ekyvc)

#### 4.2.2. 以裁剪参数作为条件

此外，以前的 SD 模型存在一个比较典型的问题：生成的物体不完整，像是被裁剪过的，如下图 SD1.5 和 SD 2.1 的结果。作者猜测这可能和训练阶段的随机裁剪有关，考虑到这个因素，作者将裁剪的左上坐标（top, left）作为条件输入模型，和 size 类似。如下图 Figure 4 中 SDXL 的结果，其生成结果都更加完整：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWoaO2W9wHXWFyqRdeDjyMMnCCBGwEZicuqmnVXfMLoZWh3e2jTicAUg3w/640?wx_fmt=png&from=appmsg&randomid=hzemcz69)

如下图 Figure 5 所示，在推理阶段也可以通过裁剪坐标来控制位置关系：![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWPjCJtrDYdic95YLKR3viclicqCEacb515PD7riaHA8S1PlTu4b9oqIAKUw/640?wx_fmt=png&from=appmsg&randomid=13v7u6wn)

### 4.3. 多分辨率训练

真实世界的图像会包含不同的大小和长宽比，而文本到模型生成的图像分辨率通常为 512x512 或 1024x1024，作者认为这不是一个自然的选择。受此启发，作者以不同的长宽比来微调模型：首先将数据划分为不同长宽比的桶，其中尽可能保证总像素数接近 1024x1024 个，同时以 64 的整数倍来调整高度和宽度。如下图所示为作者使用的宽度和高度。在训练过程中，每次都从同样的桶中选择一个 batch，并在不同的桶间交替。此外，和之前的 size 类似，作者会将桶的高度和宽度 （h, w）作为条件，经傅里叶特征编码后添加到时间步 embedding 中：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWKYpOZf0bichJ7Ewy4mKZzVU9tHzmZJAUNMWuHy0Aib7tCvup6qLZS3MA/640?wx_fmt=png&from=appmsg&randomid=onldzmu6)

### 4.4. 训练

SDXL 模型的训练包含多个步骤：

- 基于内部数据集，以 256x256 分辨率预训练 6,000,000 step，batch size 为 2048。使用了 size 和 crop 条件。
- 继续以 512x512 分辨率训练 200,000 step。
- 最后使用多分辨率（近似 1024x1024）训练。

根据以往的经验，作者发现所得到的的模型有时偶尔会生成局部质量比较差的图像，为了解决这个问题，作者在同一隐空间训练了一个独立的 LDM（Refiner），该 LDM 专门用于高质量、高分辨率的数据。在推理阶段，直接基于 Base SDXL 生成的 Latent code 继续生成，并使用相同的文本条件（当然，此步骤是可选的），实验证明可以提高背景细节以及人脸的生成质量。

### 4.5. 实验结果

如下图所示，作者基于用户评估，最终带有 Refiner 的 SDXL 获得了最高分，并且 SDXL 结果明显优于 SD 1.5 和 SD 2.1。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWC7XF9UicxPEzXN4oOFMwPAgv6GCmKicQTTmFWmIsXe0Yb760Muo5LiaVA/640?wx_fmt=png&from=appmsg&randomid=dk4sob0q)

如下图 Figure 10 所示为 SDXL（没有 Refiner） 和 Midjourney 5.1 的对比结果，可见 SDXL 的结果略胜一筹：![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWs5aMQbPbnZR6O7iasR45cG6IwKalElfib0ciar8BuXwrnoGDdsibdnNx9g/640?wx_fmt=png&from=appmsg&randomid=dvvpqaqd)

如下图 Figure 11 所示为 SDXL（带有 Refiner） 和 Midjourney 5.1 的对比结果，可见 SDXL 的结果同样略胜一筹：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWUTdLeLtwaRfENE9moXgGETicFucjt5pRQsl6QTJzv7eFtibBSyKAqhfQ/640?wx_fmt=png&from=appmsg&randomid=bchd5ebd)

## 五、SDXL-Turbo

### 5.1. SDXL-Turbo 方法

SDXL-Turbo 在模型上没有什么修改，主要是引入蒸馏技术，以便减少 LDM 的生成步数，提升生成速度。大致的流程为：

- 从 Tstudent 中采样步长 s，对于原始图像 x0 进行 s 步的前向扩散过程，生成加噪图像 xs。
- 使用学生模型 ADD-student 对 xs 进行去噪，生成去噪图像 xθ。
- 基于原始图像 x0 和去噪图像 xθ 计算对抗损失（adversarial loss）。
- 从 Tteacher 中采样步长 t，对去噪后的图像 xθ 进行 t 步的前向扩散过程，生成 xθ,t。
- 使用教师模型 DM-student 对 xθ,t 进行去噪，生成去噪图像 xψ。
- 基于学生模型去噪图像 xθ 和教师模型去噪图像 xψ 计算蒸馏损失（distillation）。
- 根据损失进行反向传播（注意，教师模型不更新，因此会 stop 梯度）。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUW1ZQdhCNgx3HZRvRGtKibpNTRROFmVoSCrCPfbQGsRRKselr0ItklsXQ/640?wx_fmt=png&from=appmsg&randomid=mtk07cf2)

需要说明的是，通常 ADD-student 模型需要预训练过程，然后再蒸馏。此外，Tstudent 的 N 比较小，作者设置为 4，而 Tteacher 的 N 比较大，为 1000。也就是学生模型可能只加噪 1,2,3,4 步，而教师模型可能加噪 1-1000 步。

此外，作者在训练中还用了其他技巧，比如使用了 zero-terminal SNR；教师模型不是直接作用于原始图像 x0，而是作用于学生模型恢复出的图像 xθ，否则会出现 OOD（out of distribution） 问题；作者还应用了 Score Distillation Loss，并且与最新的 noise-free score distillation 进行了对比。

### 5.2. 消融实验

作者进行了一系列的消融实验：

- (a) 在判别器（Discriminator）中使用不同模型的结果。
- (b) 在判别器中使用不同条件的效果，可见使用文本+图像条件获得最好结果。
- (c) 学生模型使用预训练的结果，使用预训练效果明显提升。
- (d) 不同损失的影响。
- (e) 不同学生模型和教师模型的影响。
- (f) 教师 step 的影响。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWTv560APRaM6v91sKVWUZfv3adEtcM5Lrp48fX0FW0JzeCRJHrV6P1Q/640?wx_fmt=png&from=appmsg&randomid=55fw15zf)

### 5.3. 实验结果

如下图所示，作者与不同的蒸馏方案进行了对比，本文提出的方案只需一步就能获得最优的 FID 和 CLIP 分数：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWIicc22CuAiaAxGYvRI1rrxYUsicXjZhM9q11ydwwHYWRHm4udf7N1pP3w/640?wx_fmt=png&from=appmsg&randomid=lbkllg3j)

如下图 Figure 5 和 Figure 6 所示为性能和速度的对比，ADD-XL 1 步比 LCM-XL 4 步的效果更好，同时 ADD-XL 4 步可以超越 SDXL 50 步的结果，总之，ADD-XL 获得了最佳性能：![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUW9UMQXiahhZNSo69xjt4fg2a1sXEibibiau70ayqta0IrfrxLkoUMjDEwvw/640?wx_fmt=png&from=appmsg&randomid=6hex407w)

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUW7MeX390vUPZ2Fqx9ibgwiaiaAGSfSJO7xicBJjBpUNkooTT8ZicU1C1SgEw/640?wx_fmt=png&from=appmsg&randomid=fyqwvpy7)

## 六、演进

### 6.1. Latent Diffusion

Stable Diffusion 之前的版本，对应的正是论文的开源版本，位于代码库 High-Resolution Image Synthesis with Latent Diffusion Models 中。

该版本发布于 2022 年 4 月，主要包含三个模型：

- 文生图模型：基于 LAION-400M 数据集训练，包含 1.45B 参数。
- 图像修复模型：指定区域进行擦除。
- 基于 ImageNet 的类别生成模型：在 ImageNet 上训练，指定类别条件生成，获得了 3.6 的 FID 分数。使用了 Classifier Free Guidance 技术。

代码实现参考了 OpenAI 的 Diffusion Models Beat GANs 代码实现。

### 6.2. Stable Diffusion V1

Stable Diffusion 的第一个版本，特指文生图扩散模型，位于代码库 GitHub - CompVis/stable-diffusion: A latent text-to-image diffusion model 中。

该版本发布于 2022 年 8 月，该模型包含 2 个子模型：

- AutoEncoder 模型：U-Net，8 倍下采样，包含 860M 参数。
- Text Encoder 模型：使用 CLIP ViT-L/14 中的 Text encoder。

模型首先在 256x256 的分辨率下训练，然后在 512x512 的分辨率下微调。总共包含 4 个子版本：

- sd-v1-1.ckpt：
- 在 LAION-2B-en 数据集上以 256x256 分辨率训练 237k step。
- 在 LAION-high-resolution（LAION-5B 中超过 1024x1024 分辨率的 170M 样本）上以 512x512 分辨率继续训练 194k step。
- sd-v1-2.ckpt：
- 复用 sd-v1-1.ckpt，在 LAION-aesthetics v2 5+（LAION-2B-en 中美观度分数大于 5.0 的子集） 上以 512x512 分辨率继续训练 515k step。
- sd-v1-3.ckpt：
- 复用 sd-v1-2.ckpt，在 LAION-aesthetics v2 5+ 上以 512x512 分辨率继续训练 195k step，使用了 Classifier Free Guidance 技术，以 10% 概率删除文本条件。
- sd-v1-4.ckpt：
- 复用 sd-v1-2.ckpt，在 LAION-aesthetics v2 5+ 上以 512x512 分辨率继续训练 225k step，使用了 Classifier Free Guidance 技术，以 10% 概率删除文本条件。

对应的 FID 和 CLIP 分数如下图所示，可见从 v1-1 到 v1-2，再到 v1-3 提升都很明显，v1-3 和 v1-4 差距不大：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWTFEWUB2NoQgm2icIGuQrQ48Izu6ibEnuibE4nWs1toTfic7X9O8q6ZNC2w/640?wx_fmt=png&from=appmsg&randomid=71tppjjn)

### 6.3. Stable Diffusion V1.5

Stable Diffusion 的 V1.5 版本，由 runway 发布，位于代码库 GitHub - runwayml/stable-diffusion: Latent Text-to-Image Diffusion 中。

该版本发布于 2022 年 10 月，主要包含两个模型：

- sd-v1-5.ckpt：
- 复用 sd-v1-2.ckpt，在 LAION-aesthetics v2 5+ 上以 512x512 分辨率继续训练 595k step，使用了 Classifier Free Guidance 技术，以 10% 概率删除文本条件。
- sd-v1-5-inpainting.ckpt：
- 复用 sd-v1-5.ckpt，在 LAION-aesthetics v2 5+ 上以 512x512 分辨率以 inpainting 训练了 440k step，使用 Classifier Free Guidance 技术，以 10% 概率删除文本条件。在 U-Net 的输入中额外加了 5 个 channel，4 个用于 masked 的图像，1 个用于 mask 本身。

对应的 FID 和 CLIP 分数如下图所示，可以看出，v1.5 相比 v1.4 的提升也不是很明显：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWjictxSgtnkrB46oPlttQJQ4aiaEgVwx4F4Map9iaLB7pyFfEAbw0WBMUg/640?wx_fmt=png&from=appmsg&randomid=vxetuq8s)

如下图所示为图像修复的示例：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUWYA6FTt1NWu37jqye0ufE36yReva8FRg6BzaTWtyocUO1FmUWOkOzUA/640?wx_fmt=png&from=appmsg&randomid=8h1828s7)

### 6.3. Stable Diffusion V2

Stable Diffusion 的 V2 版本，由 Stability-AI 发布，位于代码库 GitHub - Stability-AI/stablediffusion: High-Resolution Image Synthesis with Latent Diffusion Models 中。

V2 包含三个子版本，分别为 v2.0，v2.1 和 Stable UnCLIP 2.1：

- v2.0：
- 发布于 2022 年 11 月，U-Net 模型和 V1.5 相同，Text encoder 模型换成了 OpenCLIP-ViT/H 中的 text encoder。
- SD 2.0-base：分别率为 512x512
- SD 2.0-v：基于 2.0-base 微调，分辨率提升到 768x768，同时利用 [2202.00512] Progressive Distillation for Fast Sampling of Diffusion Models 提出的技术大幅降低 Diffusion 的步数。
- 发布了一个文本引导的 4 倍超分模型。
- 基于 2.0-base 微调了一个深度信息引导的生成模型。
- 基于 2.0-base 微调了一个文本信息引导的修复模型。
- v2.1：
- 发布于 2022 年 12 月，模型结构和参数量都和 v2.0 相同。并在 v2.0 的基础上使用 LAION 5B 数据集（较低的 NSFW 过滤约束）微调。同样包含 512x512 分辨率的 v2.1-base 和 768x768 分辨率的 v2.1-v。
- Stable UnCLIP 2.1：
- 发布于 2023 年 3 月，基于 v2.1-v（768x768 分辨率） 微调，参考 OpenAI 的 DALL-E 2（也就是 UnCLIP），可以更好的实现和其他模型的联合，同样提供基于 CLIP ViT-L 的 Stable unCLIP-L 和基于 CLIP ViT-H 的 Stable unCLIP-H。

如下图所示为 v2.0 和 v2.0-v 与 v1.5 的对比，可见其都有明显提升：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgdcmGAZ0wl1a0uya0eCfUW0EKtEEd3icYFBJLrHuK6eVfG01l0sEvHLnNK6PrXzlqmPd6o3RAUiatQ/640?wx_fmt=png&from=appmsg&randomid=5nymbv3a)

### 6.4. Stable Diffusion XL

Stable Diffusion 的 XL 版本，由 Stability-AI 发布，位于代码库 Generative Models by Stability AI。

该版本发布于 2023 年 06 月，主要包含两个模型：

- SDXL-base-0.9：基于多尺度分辨率训练，最大分辨率 1024x1024，包含两个 Text encoder，分别为 OpenCLIP-ViT/G 和 CLIP-ViT/L。
- SDXL-refiner-0.9：用来生成更高质量的图像，不应直接使用，此外文本条件只使用 OpenCLIP 中的 Text encoder。

2023 年 07 月发布 1.0 版本，同样对应两个模型：

- SDXL-base-1.0：基于 SDXL-base-0.9 改进。
- SDXL-refiner-1.0：基于 SDXL-refiner-0.9 改进。

2023 年 11 月发表 SDXL-Trubo 版本，也就是优化加速的版本。

## 七、参考链接

1. https://arxiv.org/abs/2112.10752
2. https://github.com/CompVis/latent-diffusion
3. https://arxiv.org/abs/2307.01952
4. https://github.com/Stability-AI/generative-models
5. https://arxiv.org/abs/2311.17042
6. https://github.com/Stability-AI/generative-models
7. https://jalammar.github.io/illustrated-stable-diffusion/

