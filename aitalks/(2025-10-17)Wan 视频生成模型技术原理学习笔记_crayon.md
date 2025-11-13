# Wan 视频生成模型技术原理学习笔记

**Author:** crayon

**Date:** 2025-10-17

**Link:** https://zhuanlan.zhihu.com/p/1962674745909674644

最近重新梳理了一下模型的pipeline，靠着问ai，整理出了这么一份笔记。

## [Wan 视频扩散模型](https://zhida.zhihu.com/search?content_id=264400195&content_type=Article&match_order=1&q=Wan+%E8%A7%86%E9%A2%91%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B&zhida_source=entity)学习笔记

Wan 是最近比较火的视频生成扩散模型，可以理解为”[Stable Diffusion](https://zhida.zhihu.com/search?content_id=264400195&content_type=Article&match_order=1&q=Stable+Diffusion&zhida_source=entity) 的视频版”。它的核心挑战在于同时建模空间结构（图像内容）和时间一致性（运动轨迹），还要在有限显存下搞定高分辨率视频生成。

整个系统由三个核心模块组成：[Video VAE](https://zhida.zhihu.com/search?content_id=264400195&content_type=Article&match_order=1&q=Video+VAE&zhida_source=entity) 负责压缩视频到时空 latent 表征，Diffusion Transformer 在 latent 空间中做扩散去噪，Text Encoder 把文本 prompt 转换为语义向量来引导生成。

* * *

## Wan 模型总览与核心创新

作为阿里自研的视频生成模型，Wan 的创新性主要体现在以下几个方面：

### 多阶段层级生成架构

Wan 采用了 **分层生成（[Hierarchical Generation](https://zhida.zhihu.com/search?content_id=264400195&content_type=Article&match_order=1&q=Hierarchical+Generation&zhida_source=entity)）** 策略来解决高分辨率视频的显存瓶颈：

1.  **Base Model**：生成低分辨率、低帧率的基础 latent（如 480p@16fps）
2.  **[Temporal Super-Resolution](https://zhida.zhihu.com/search?content_id=264400195&content_type=Article&match_order=1&q=Temporal+Super-Resolution&zhida_source=entity)**：提升时间分辨率（如 16fps → 48fps）
3.  **[Spatial Super-Resolution](https://zhida.zhihu.com/search?content_id=264400195&content_type=Article&match_order=1&q=Spatial+Super-Resolution&zhida_source=entity)**：提升空间分辨率（如 480p → 1080p）

这种架构类似于 Sora、HunyuanVideo 等顶级模型，是当前解决视频生成显存问题的主流方案。

### 大规模 Transformer 主干

Wan 使用 **[Video DiT](https://zhida.zhihu.com/search?content_id=264400195&content_type=Article&match_order=1&q=Video+DiT&zhida_source=entity)（Diffusion Transformer）** 作为主干网络：

-   模型规模约百亿级参数（官方未公开具体数值）
-   采用与 [Qwen 系列](https://zhida.zhihu.com/search?content_id=264400195&content_type=Article&match_order=1&q=Qwen+%E7%B3%BB%E5%88%97&zhida_source=entity)兼容的文本编码器，**中文理解能力显著优于纯英文模型**
-   时空注意力分解设计，支持长视频生成

### 多模态训练语料

训练数据涵盖：

-   中英文双语视频-文本对（上亿级规模）
-   影视素材、广告、短视频等多种类型
-   音视频脚本、镜头描述等结构化标注

这使得 Wan 在多语言理解、复杂场景建模上具有优势。

### 显存优化策略

针对高分辨率、长视频生成的显存挑战，Wan 采用了多种优化技术：

-   **分块注意力（Chunked Attention）**：将长序列分块处理
-   **[FlashAttention](https://zhida.zhihu.com/search?content_id=264400195&content_type=Article&match_order=1&q=FlashAttention&zhida_source=entity) 集成**：加速注意力计算，降低显存占用
-   **[混合精度训练](https://zhida.zhihu.com/search?content_id=264400195&content_type=Article&match_order=1&q=%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6%E8%AE%AD%E7%BB%83&zhida_source=entity)（FP16/BF16）**：减少一半显存
-   **动态显存调度**：类似 ZeRO-offload，在 CPU 和 GPU 间动态交换

### Pipeline 整体架构

```text
用户 Prompt（中/英文）
       ↓
┌──────────────────────────┐
│  Text Encoder (Qwen/CLIP) │
│  - 编码文本语义            │
│  - 输出条件向量            │
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│  Video VAE Encoder        │
│  - 3D 卷积压缩            │
│  - 降维 8~16 倍           │
└──────────┬───────────────┘
           ↓
     Latent Space (z)
           ↓
┌──────────────────────────┐
│ Diffusion Transformer      │
│  - 时空注意力分解          │
│  - 文本交叉注意力          │
│  - 逐步去噪 (50 步)        │
└──────────┬───────────────┘
           ↓
  去噪后的 Latent
           ↓
┌──────────────────────────┐
│  Video VAE Decoder        │
│  - 3D 转置卷积            │
│  - 恢复视频帧             │
└──────────┬───────────────┘
           ↓
     生成的视频帧
```

### 与其他模型的技术对比

| 模型 | 主干架构 | 文本编码器 | 层级生成 | 核心特征 | 中文支持 |
| --- | --- | --- | --- | --- | --- |
| Stable Video Diffusion | 3D UNet | CLIP | ✗ | 基于 SD 扩展 | 弱 |
| VideoCrafter | DiT | T5-XXL | ✗ | 可控生成 | 一般 |
| HunyuanVideo | Video DiT | Qwen | ✓ | 多阶段 | 强 |
| Wan（阿里） | Video DiT | Qwen 2.5 | ✓ | 高分辨率+语义一致 | 原生支持 |

Wan 的核心优势在于：

1.  **中文原生支持**：使用 Qwen 系列，中文 prompt 理解力远超 CLIP
2.  **多阶段架构**：可生成 1080p 以上的高分辨率视频
3.  **工业级优化**：显存效率高，支持消费级显卡推理

* * *

## 从输入到输出的完整流程

整个生成过程其实很直观：

```text
文本（prompt）
    ↓
Text Encoder → 文本特征
    ↓
随机噪声 latent（初始视频块）
    ↓
Diffusion Transformer（逐步去噪）
    ↓
预测干净 latent（表示生成视频）
    ↓
Video VAE Decoder（解码）
    ↓
生成连续视频帧
```

简单来说就是：”文字” → “潜在时空表征” → “连续视频”。

* * *

## Video VAE：3D视频自编码器

在像素级别生成视频成本实在太高了，所以需要先用 Video VAE 压缩视频到时空 latent：

$ x \in \mathbb{R}^{T\times H\times W\times 3} \quad\Rightarrow\quad z \in \mathbb{R}^{C\times T'\times H'\times W'} $

压缩比例一般是：

-   时间维度：1/4
-   空间维度：1/8 ~ 1⁄16

### 编码器结构

核心算子是 **3D卷积（Conv3D）**：

```text
import torch.nn as nn

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 4, 4),
            stride=(1, 4, 4),  # 时间步长=1，空间步长=4
            padding=(1, 1, 1)
        )
    
    def forward(self, x):
        return self.conv(x)
```

时间方向步长是 1（保留更多时间信息），空间方向步长是 4（降低空间分辨率），这样可以提取时空局部特征。

然后用残差结构保持梯度稳定，防止信息在深层网络中衰减：

$ y = x + \text{Conv3D}(\text{ReLU}(\text{Conv3D}(x))) $

来看个具体例子：

```text
输入视频:  (B, 3, 16, 512, 512)   [批次, 通道, 时间帧数, 高, 宽]
    ↓
Conv3D + ResBlock ×3
    ↓
latent:    (B, 8, 16, 64, 64)    [空间压缩8倍, 时间维度保持16帧]
```

压缩率相当可观，这也是为什么能在消费级显卡上跑视频生成的关键。

### 解码器结构

解码器就是编码器的反向镜像，使用 **ConvTranspose3D**（转置卷积）逐步恢复空间和时间分辨率：

```text
class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.deconv_blocks = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, 64, 
                             kernel_size=(3,4,4), 
                             stride=(1,4,4)),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, 
                             kernel_size=(3,4,4), 
                             stride=(1,4,4)),
            nn.ReLU(),
            nn.Conv3d(32, 3, kernel_size=(3,3,3), padding=(1,1,1))
        )
    
    def forward(self, z):
        return self.deconv_blocks(z)
```

训练目标是重建损失加 KL 散度：

$ \mathcal{L}_{\text{VAE}} = \|x - \hat{x}\|_2^2 + \beta \cdot \text{KL}(q(z|x)\|p(z)) $

第一项保证重建质量，第二项保证 latent 分布的规整性。$\beta$ 通常设置为 0.00001 ~ 0.001，是个玄学参数，需要根据数据集调整。

### VAE 数学原理深入理解

VAE 的核心思想是学习一个概率分布，而不是直接映射。具体来说：

**编码器学习后验分布**：给定输入视频 $x$，编码器输出均值 $\mu(x)$ 和方差 $\sigma^2(x)$：

$ q_\phi(z|x) = \mathcal{N}(z; \mu(x), \sigma^2(x)I) $

**重参数化技巧**：为了让采样过程可导，使用重参数化：

$ z = \mu(x) + \sigma(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) $

这样梯度可以通过 $\mu$ 和 $\sigma$ 反向传播。

**KL 散度的闭式解**：对于高斯分布，KL 散度有解析解：

$ \text{KL}(q(z|x)\|p(z)) = \frac{1}{2}\sum_{i=1}^{d}\left(\mu_i^2 + \sigma_i^2 - \log\sigma_i^2 - 1\right) $

这个公式的直观理解：

-   $\mu_i^2$：惩罚均值偏离原点
-   $\sigma_i^2 - 1$：惩罚方差偏离 1
-   $-\log\sigma_i^2$：防止方差坍缩到 0

**为什么需要 KL 散度项？**

没有 KL 约束的话，编码器会把每个视频映射到完全不同的 latent 空间区域，导致：

1.  Latent 空间不连续，无法插值
2.  解码器很难泛化到训练时没见过的 latent
3.  生成时从随机噪声出发会失败

KL 散度强制所有视频的 latent 都集中在标准正态分布附近，保证了 latent 空间的平滑性。

* * *

## Diffusion Transformer：视频扩散主体

### 扩散原理

扩散模型的训练思想很简单：

1.  **前向加噪**：向 latent 逐步加噪声  
    $    x_t = \sqrt{\alpha_t}x_0 + \sqrt{1-\alpha_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0,I)    $  
    其中 $\alpha_t = \prod_{i=1}^{t}(1-\beta_i)$，$\beta_t$ 为噪声调度
2.  **反向去噪**：训练模型预测噪声  
    $    \hat{\epsilon} = \epsilon_\theta(x_t, t, c) \quad \text{其中 } c \text{ 为文本条件}    $  
    

模型学会”去噪”，就能从噪声恢复 latent。这个过程就像是教模型如何从一团迷雾中逐渐看清图像。

### 扩散过程的数学推导

**前向扩散的马尔可夫链**：

扩散过程可以看作一个马尔可夫链，每一步都向数据中添加一点噪声：

$ q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I) $

这里 $\beta_t$ 是噪声调度，通常从 $\beta_1=10^{-4}$ 线性增长到 $\beta_T=0.02$。

**任意时刻的分布（重参数化性质）**：

通过递推，可以直接从 $x_0$ 采样 $x_t$，而不需要逐步加噪：

$ x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon $

其中 $\bar{\alpha}_t = \prod_{i=1}^{t}\alpha_i = \prod_{i=1}^{t}(1-\beta_i)$。

这个公式非常关键，它允许训练时随机选择任意时间步 $t$，而不需要从 $t=0$ 一步步加噪到 $t$。

**为什么用这个形式？**

这个形式保证了：

-   当 $t \to 0$ 时，$\bar{\alpha}_t \to 1$，$x_t \approx x_0$（几乎没噪声）
-   当 $t \to T$ 时，$\bar{\alpha}_t \to 0$，$x_t \approx \epsilon$（纯噪声）
-   过程是连续的、可微的

**反向过程的推导**：

反向过程想要从 $x_t$ 恢复 $x_{t-1}$。根据贝叶斯定理：

$ q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I) $

其中：

$ \tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t $

问题是我们不知道 $x_0$，所以用神经网络 $\epsilon_\theta$ 预测噪声 $\epsilon$，然后反推：

$ x_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} $

代入上面的公式，就得到了采样公式。

**训练目标的推导**：

完整的变分下界（ELBO）推导会得到：

$ \mathcal{L}_{\text{simple}} = \mathbb{E}_{t,x_0,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right] $

这个简化版本在实践中效果最好。原因是它给每个时间步相同的权重，而完整的 ELBO 会给不同时间步不同的权重。

**噪声调度的选择**：

噪声调度 $\beta_t$ 的设计非常重要：

**Linear Schedule**（线性调度）：

$ \beta_t = \beta_{\text{start}} + \frac{t}{T}(\beta_{\text{end}} - \beta_{\text{start}}) $

简单但在视频生成中效果一般。

  

**Cosine Schedule**（余弦调度）：

$ \bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)^2 $

在视频生成中效果更好，因为它在前期加噪更慢，保留更多信息。

  

直观理解：

-   Linear：噪声均匀增加，可能太激进
-   Cosine：前期慢后期快，更符合人类感知

### Video DiT 模型主体

输入的时空 latent：

$ z_t \in \mathbb{R}^{C\times T'\times H'\times W'} $

会被展平成 token 序列：

$ \{z_1, z_2, \ldots, z_N\}, \quad N = C \times T' \times H' \times W' $

然后进入标准的 Transformer 结构：

```text
class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, 
                                         batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
    
    def forward(self, x, text_cond):
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # Cross-attention with text
        x_norm = self.norm1(x)
        attn_cross, _ = self.attn(x_norm, text_cond, text_cond)
        x = x + attn_cross
        
        # FFN
        x = x + self.mlp(self.norm2(x))
        return x
```

这里有个巧妙的设计：**时空注意力分解（Factorized 3D Attention）**。

直接做全 3D attention 的复杂度是 $O(T^2H^2W^2)$，完全不可行。Wan 的做法是：

1.  先做时间维 attention：关注不同帧之间的关系
2.  再做空间维 attention：关注每帧内部的空间结构
3.  最后融合两者的结果

这样计算量从 $O(T^2H^2W^2)$ 降到 $O(TH^2W^2 + T^2HW)$，显存才扛得住。

### 时空注意力分解的详细机制

**为什么需要分解？**

假设视频 latent 是 $(T=16, H=64, W=64)$，如果直接做 3D attention：

-   Token 数量：$N = 16 \times 64 \times 64 = 65536$
-   Attention 矩阵大小：$65536 \times 65536 \approx 4.3$ 亿
-   存储需求（float32）：$4.3 \times 10^9 \times 4 \text{ bytes} = 17.2 \text{ GB}$

这还只是一层、一个 head 的情况！显然不可行。

**时空分解的数学表示**：

**方法1：串行分解（Sequential）**

先做时间 attention，再做空间 attention：

$ \text{Output} = \text{SpatialAttn}(\text{TemporalAttn}(X)) $

具体来说：

1.  **时间 Attention**：将 $(T, H, W, C)$ 重塑为 $(HW, T, C)$，在时间维度做 attention  
    $    Z_{\text{temp}} = \text{Attention}(Q_t, K_t, V_t), \quad Q_t, K_t, V_t \in \mathbb{R}^{HW \times T \times d}    $  
    复杂度：$O(HW \cdot T^2 \cdot d)$
2.  **空间 Attention**：将结果重塑为 $(T, HW, C)$，在空间维度做 attention  
    $    Z_{\text{out}} = \text{Attention}(Q_s, K_s, V_s), \quad Q_s, K_s, V_s \in \mathbb{R}^{T \times HW \times d}    $  
    复杂度：$O(T \cdot (HW)^2 \cdot d)$

总复杂度：$O(HWT^2d + TH^2W^2d) = O(T(HW)^2 + (HW)T^2)$

对于 $(T=16, HW=4096)$：

-   全 3D：$N^2 = 65536^2 \approx 4.3 \times 10^9$
-   分解后：$T(HW)^2 + (HW)T^2 = 16 \times 4096^2 + 4096 \times 16^2 \approx 2.7 \times 10^8$

降低了约 **16 倍**！

**方法2：并行分解（Parallel）**

同时做时间和空间 attention，然后融合：

$ \text{Output} = \alpha \cdot \text{TemporalAttn}(X) + \beta \cdot \text{SpatialAttn}(X) $

其中 $\alpha, \beta$ 是可学习的权重。这种方法：

-   优点：两个分支可以并行计算，更快
-   缺点：缺少时空信息的交互

**实际实现细节**：

```text
class FactorizedAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        # 时间注意力
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads)
        # 空间注意力
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads)
        
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        # 时间注意力：(B, C, T, H, W) -> (B*H*W, T, C)
        x_temp = x.permute(0, 3, 4, 2, 1).reshape(B*H*W, T, C)
        x_temp, _ = self.temporal_attn(x_temp, x_temp, x_temp)
        x_temp = x_temp.reshape(B, H, W, T, C).permute(0, 4, 3, 1, 2)
        
        # 空间注意力：(B, C, T, H, W) -> (B*T, H*W, C)
        x_spatial = x_temp.permute(0, 2, 1, 3, 4).reshape(B*T, C, H*W)
        x_spatial = x_spatial.permute(0, 2, 1)  # (B*T, H*W, C)
        x_spatial, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial)
        x_spatial = x_spatial.permute(0, 2, 1).reshape(B, T, C, H, W)
        x_spatial = x_spatial.permute(0, 2, 1, 3, 4)
        
        return x_spatial
```

**窗口注意力进一步优化**：

即使分解后，$(HW)^2$ 对于高分辨率仍然很大。可以用窗口注意力：

只在局部窗口内计算 attention（典型窗口大小 $8 \times 8$ 或 $16 \times 16$）：

-   复杂度从 $O((HW)^2)$ 降到 $O(HW \cdot w^2)$，其中 $w$ 是窗口大小
-   代价是失去了全局信息，但可以通过多层或 shifted window 缓解

### 文本引导机制

文本经过编码器（比如 CLIP 或 Qwen）得到特征向量：

```text
from transformers import CLIPTextModel, CLIPTokenizer

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

prompt = "一个人在沙滩上奔跑，阳光明媚"
tokens = tokenizer(prompt, return_tensors="pt")
text_features = text_encoder(**tokens).last_hidden_state  # (1, seq_len, 768)
```

通过 **cross-attention** 融合进 Transformer：

$ \text{Attention}(Q=z, K=t, V=t) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V $

其中 $Q$ 来自视频 latent，$K,V$ 来自文本特征 $t$。模型据此在去噪过程中引入语义信息，确保生成的视频符合文本描述。

**Wan 的文本编码器特点**：

Wan 采用 **Qwen 系列文本模型**（如 Qwen-2.5），相比传统的 CLIP 或 T5，有以下显著优势：

1.  **中文理解能力强**：  
    

-   CLIP 主要在英文数据上训练，中文 token 数量少
-   Qwen 原生支持中文，词汇表覆盖更全面
-   对成语、俗语、文化背景的理解更准确

  

1.  **语义表达更丰富**： “\`python  
    CLIP 的表现（弱）  
    prompt\_cn = “一位古装美女在竹林中翩翩起舞”  
    CLIP 可能把”翩翩起舞”理解为简单的 “dancing”

\# Qwen 的表现（强） # Qwen 能理解”翩翩”表示轻盈优雅的动作风格 # 能联想到中国古典舞蹈的特征

```text
3. **长文本处理能力**：
   - CLIP 限制在 77 个 tokens
   - Qwen 支持 2048+ tokens，可以接受详细的场景描述

这使得 Wan 在处理中文 prompt 时，生成质量明显优于使用 CLIP 的模型。

#### Cross-Attention 的深入理解

**Self-Attention vs Cross-Attention**：

**Self-Attention**（自注意力）：
<!--MATH_PH_24-->
<!--MATH_PH_113--> 都来自同一个输入 <!--MATH_PH_114-->，用于捕捉输入内部的依赖关系。

**Cross-Attention**（交叉注意力）：
<!--MATH_PH_25-->
其中：
- <!--MATH_PH_115-->：来自视频 latent（Query："我要找什么？"）
- <!--MATH_PH_116-->：来自文本特征（Key/Value："我能提供什么？"）

**直观理解**：

可以把 Cross-Attention 看作一个"查询-检索"过程：
1. 视频 latent 的每个位置问："我应该包含什么内容？"
2. 在文本特征中查找相关信息（通过 <!--MATH_PH_117--> 计算相似度）
3. 根据相似度加权聚合文本信息（通过 softmax 和 <!--MATH_PH_118-->）

**注意力权重的可视化**：

假设 prompt 是："一个人在沙滩上奔跑"，注意力权重可能是：
```

## 视频位置 最关注的文本 token

前景中心区域 → “人” (0.6), “奔跑” (0.3) 下半部分 → “沙滩” (0.7), “上” (0.2) 背景 → “沙滩” (0.5), “一个” (0.3) 运动轨迹 → “奔跑” (0.8)

````text
**Classifier-Free Guidance（CFG）**：

实际使用中，通常结合有条件和无条件生成来增强控制：

<!--MATH_PH_26-->

其中：
- <!--MATH_PH_119-->：有文本条件的预测
- <!--MATH_PH_120-->：无条件预测（文本为空）
- <!--MATH_PH_121-->：引导强度（通常 7.5-15）

这个技巧的作用：
- <!--MATH_PH_122-->：完全忽略文本，随机生成
- <!--MATH_PH_123-->：正常的条件生成
- <!--MATH_PH_124-->：过度强调文本，生成更符合描述但可能失真

**多模态融合的其他方法**：

除了 Cross-Attention，还有：

1. **AdaLN（Adaptive Layer Normalization）**：
   <!--MATH_PH_27-->
   文本条件 <!--MATH_PH_125--> 通过 MLP 预测缩放和偏移参数 <!--MATH_PH_126-->。

2. **Concatenation**：
   直接把文本特征拼接到视频 latent：
   <!--MATH_PH_28-->
   
3. **FiLM（Feature-wise Linear Modulation）**：
   <!--MATH_PH_29-->

Cross-Attention 在视频生成中效果最好，因为它允许精细的、位置相关的文本控制。

### 训练损失

训练过程就是标准的去噪回归：

<!--MATH_PH_30-->

完整训练流程：

```python
def train_step(video_batch, prompt_batch):
    # 1. 编码视频到 latent
    z0 = vae_encoder(video_batch)  # (B, C, T', H', W')
    
    # 2. 随机采样时间步 t
    t = torch.randint(0, num_timesteps, (batch_size,))
    
    # 3. 采样高斯噪声
    epsilon = torch.randn_like(z0)
    
    # 4. 前向扩散：加噪
    zt = sqrt_alpha[t] * z0 + sqrt_one_minus_alpha[t] * epsilon
    
    # 5. 编码文本提示
    text_cond = text_encoder(prompt_batch)
    
    # 6. 模型预测噪声
    epsilon_pred = diffusion_transformer(zt, t, text_cond)
    
    # 7. 计算损失
    loss = F.mse_loss(epsilon_pred, epsilon)
    
    # 8. 反向传播
    loss.backward()
    optimizer.step()
    
    return loss
````

简单粗暴，但就是有效。

* * *

## 推理流程

生成视频时的反向过程：

```text
@torch.no_grad()
def generate_video(prompt, num_frames=16, height=512, width=512, num_steps=50):
    # 1. 编码文本
    text_cond = text_encoder(prompt)
    
    # 2. 初始化随机噪声 latent
    z = torch.randn(1, 8, num_frames, height//8, width//8, device=device)
    
    # 3. 逐步去噪（反向过程）
    timesteps = torch.linspace(num_timesteps-1, 0, num_steps).long()
    
    for t in timesteps:
        # 预测噪声
        epsilon_pred = diffusion_transformer(z, t, text_cond)
        
        # 使用调度器更新 latent
        z = scheduler.step(epsilon_pred, z, t)
    
    # 4. 解码为视频帧
    video = vae_decoder(z)  # (1, 3, 16, 512, 512)
    
    # 5. 转换为视频文件
    save_video(video, "output.mp4", fps=24)  # 24fps 或 30fps
    
    return video
```

核心公式（DDPM 采样）：

$ z_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(z_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\hat{\epsilon}_\theta(z_t, t, c)\right) + \sigma_t \epsilon $

整个采样过程通常需要 20-50 步（DDIM/DPM-Solver），每步都要过一遍 DiT，所以生成速度是个瓶颈。

### 采样器原理详解

**DDPM（Denoising Diffusion Probabilistic Models）**：

DDPM 是最原始的采样方法，严格按照训练时的反向过程：

$ z_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(z_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\hat{\epsilon}_\theta(z_t, t, c)\right) + \sigma_t \epsilon $

其中 $\sigma_t = \sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t}$。

特点：

-   需要完整的 $T$ 步（通常 1000 步）
-   每步都添加随机噪声 $\sigma_t \epsilon$（保持随机性）
-   质量最好，但速度最慢

**DDIM（Denoising Diffusion Implicit Models）**：

DDIM 的关键洞察：去噪过程不一定要是随机的（stochastic），可以是确定的（deterministic）！

$ z_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\underbrace{\left(\frac{z_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(z_t, t)}{\sqrt{\bar{\alpha}_t}}\right)}_{\text{预测的 }x_0} + \underbrace{\sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta(z_t, t)}_{\text{指向 }z_t\text{ 的方向}} $

当 $\sigma_t = 0$ 时，过程完全确定。

关键优势：

-   **可以跳步**：不需要每一步都采样，可以从 $t=1000$ 直接跳到 $t=950$，再到 $t=900$…
-   **确定性**：相同的初始噪声 + prompt 会得到相同的结果（便于调试）
-   **速度**：20-50 步就能得到不错的结果

**DPM-Solver（Diffusion Probabilistic Model Solver）**：

DPM-Solver 将扩散过程看作一个 ODE（常微分方程）求解问题：

$ \frac{dz_t}{dt} = f(z_t, t) = -\frac{1}{2}\frac{d\log\alpha_t}{dt}z_t + \frac{1}{2}\frac{d\log\alpha_t}{dt}\sqrt{\frac{1-\bar{\alpha}_t}{\bar{\alpha}_t}}\epsilon_\theta(z_t, t) $

使用高阶 ODE solver（如 Runge-Kutta）求解这个方程，可以：

-   用更少的步数达到同样的精度
-   10-20 步就能接近 DDIM 50 步的质量

**DPM-Solver++ 改进**：

进一步优化，使用：

1.  **自适应步长**：在噪声大的地方（$t$ 大）用更大步长
2.  **高阶近似**：使用二阶或三阶 Taylor 展开

**各采样器的数学对比**：

| 采样器 | 核心思想 | 典型步数 | 确定性 | 复杂度 |
| --- | --- | --- | --- | --- |
| DDPM | 马尔可夫链反向采样 | 1000 | 否 | O(T) |
| DDIM | 非马尔可夫确定性轨迹 | 20-50 | 是 | O(S), S \ll T |
| DPM-Solver | ODE 求解 + 高阶方法 | 10-20 | 是 | O(S)，更优常数 |

**实际选择建议**：

```text
# 质量优先（研究、演示）
sampler = DDPMSampler(num_steps=1000)

# 平衡（生产环境）
sampler = DDIMSampler(num_steps=50)

# 速度优先（实时应用）
sampler = DPMSolverPlusPlus(num_steps=20, order=2)
```

**采样步数的影响**：

以 DDIM 为例，不同步数的质量权衡：

-   10 步：明显 artifacts，细节模糊
-   20 步：基本可用，细节尚可
-   50 步：良好质量，大部分场景够用
-   100 步：优秀质量，但边际收益递减
-   200+ 步：几乎看不出改进，浪费计算

* * *

## 多阶段层级生成架构

前面介绍的是单阶段生成流程，但实际上 **Wan 采用的是多阶段层级生成**，这也是 Sora、HunyuanVideo 等顶级模型的共同特征。

### 为什么需要多阶段生成？

直接生成高分辨率、高帧率视频面临三个核心问题：

1.  **显存限制**：  
    

-   高分辨率视频的 latent 空间维度很大
-   Attention 矩阵随序列长度平方增长
-   单卡难以容纳完整计算图

  

1.  **训练难度**：  
    

-   高分辨率数据稀缺且昂贵
-   直接在高分辨率上训练容易过拟合
-   梯度传播困难

  

1.  **语义与细节的解耦**：  
    

-   低分辨率阶段专注语义和运动结构
-   高分辨率阶段专注纹理细节
-   分开建模更高效

  

### Wan 的三阶段架构

**阶段 1：Base Model（基础生成）**

生成低分辨率、低帧率的 latent（如 480p@16fps）：

```text
输入：文本 prompt
输出：基础视频 latent (如 480p@16fps)
```

这一阶段的重点是：

-   **语义一致性**：确保内容符合文本描述
-   **运动结构**：建立物体的运动轨迹
-   **时空布局**：确定场景的基本构成

**阶段 2：Temporal Super-Resolution（时间上采样）**

提升时间分辨率，让运动更流畅（如 16fps → 48fps）：

```text
输入：低帧率 latent (16fps)
输出：高帧率 latent (48fps)
```

核心技术：

1.  **光流一致性损失**：  
    $    \mathcal{L}_{\text{flow}} = \sum_{t=1}^{T-1}\|I_{t+1} - \text{Warp}(I_t, F_{t\to t+1})\|^2    $  
    确保插值帧与光流预测一致。
2.  **双向插值**： 同时参考前后帧：  
    $    I_{\text{mid}} = \alpha \cdot \text{Warp}(I_{\text{prev}}, F_{\text{fwd}}) + (1-\alpha) \cdot \text{Warp}(I_{\text{next}}, F_{\text{bwd}})    $  
    
3.  **时间注意力加权**： 关键帧（原始 16 帧）权重更高，插值帧参考关键帧。

**阶段 3：Spatial Super-Resolution（空间上采样）**

提升空间分辨率，增加细节（如 480p → 1080p）：

```text
输入：低分辨率、高帧率 latent (480p@48fps)
输出：高分辨率、高帧率 latent (1080p@48fps)
```

核心技术：

1.  **Frame-wise VAE Upsampler**： 对每一帧独立做上采样，然后用时间一致性约束：  
    $    \mathcal{L}_{\text{temp}} = \sum_{t=1}^{T-1}\|\nabla I_t - \nabla I_{t-1}\|^2    $  
    
2.  **高频细节注入**： 使用 wavelet transform 分离低频和高频：  
    low\_freq, high\_freq = wavelet\_decompose(frame) # Base model 生成低频 # SR model 专注生成高频细节 upsampled = combine(upsample(low\_freq), generate(high\_freq))
3.  **GAN 损失**（可选）： 使用判别器增强纹理真实感：  
    $    \mathcal{L}_{\text{GAN}} = \mathbb{E}[\log D(x_{\text{real}})] + \mathbb{E}[\log(1-D(G(z)))]    $  
    

### 层级生成的优势

相比单阶段生成：

1.  **显存效率**：  
    

-   单阶段需要一次性加载整个高分辨率计算图
-   多阶段可分别运行，每阶段显存需求更低

  

1.  **训练效率**：  
    

-   Base model 用大规模低分辨率数据快速收敛
-   SR model 用较少高质量数据精细调优

  

1.  **质量提升**：  
    

-   分阶段建模让每个模型专注特定任务
-   时空一致性显著改善

  

1.  **灵活性**：  
    

-   可以只跑 Base model（快速预览）
-   根据需求选择是否上采样

  

### 实际推理流程

```text
@torch.no_grad()
def generate_video_hierarchical(prompt, target_resolution='1080p', target_fps=48):
    # 阶段 1：生成基础 latent (480p@16fps)
    base_latent = base_model.generate(
        prompt=prompt,
        resolution='480p',
        fps=16,
        num_steps=50
    )
    
    # 阶段 2：时间上采样 (16fps -> 48fps)
    if target_fps > 16:
        tsr_latent = temporal_sr_model.upsample(
            base_latent,
            target_fps=target_fps  # 48fps
        )
    else:
        tsr_latent = base_latent
    
    # 阶段 3：空间上采样 (480p -> 1080p)
    if target_resolution == '1080p':
        ssr_latent = spatial_sr_model.upsample(
            tsr_latent,
            target_resolution='1080p'
        )
    else:
        ssr_latent = tsr_latent
    
    # 解码为视频
    video = vae_decoder(ssr_latent)
    
    return video
```

这种架构是目前视频生成模型的标准范式，Wan、Sora、HunyuanVideo 都采用了类似的设计。

* * *

## 训练细节与显存优化

### 数据集与预处理

**数据规模**：

-   训练数据：大规模视频-文本对
-   数据来源：影视素材、广告、短视频、游戏录屏
-   多语言支持：中英文为主，兼顾其他语言

**预处理流程**：

1.  **质量过滤**：  
    def filter\_video(video, text): # 1. 美学评分 aesthetic\_score = aesthetic\_model(video) if aesthetic\_score < threshold\_aesthetic: return False # 2. CLIP 相似度（文本-视频对齐） clip\_score = clip\_similarity(text, video) if clip\_score < threshold\_clip: return False # 3. 技术质量（分辨率、帧率、编码） if not meets\_technical\_requirements(video): return False return True
2.  **时长采样**：  
    

-   Base model：短片段训练
-   TSR model：相同片段，更高帧率标注
-   SSR model：关键帧 + 高分辨率版本

  

1.  **数据增强**：  
    augmentations = \[ RandomCrop(), # 随机裁剪 RandomHorizontalFlip(), # 水平翻转 ColorJitter(0.1), # 颜色抖动 TemporalCrop(), # 时间裁剪 FrameRateSampling(), # 帧率采样 \]

### 训练策略

**分辨率渐进训练（Progressive Training）**：

不同阶段使用不同分辨率，逐步提升：

```text
第 1 阶段：
  - 低分辨率、低帧率
  - 较大 batch size
  - 较高学习率
  - 目标：快速收敛基本结构

第 2 阶段：
  - 中等分辨率、中等帧率
  - 中等 batch size
  - 降低学习率
  - 目标：提升细节和时间一致性

第 3 阶段：
  - 高分辨率、高帧率
  - 较小 batch size
  - 低学习率
  - 目标：精细调优
```

**混合精度训练（Mixed Precision）**：

使用 FP16/BF16 降低显存和加速训练：

```text
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for video, text in dataloader:
    optimizer.zero_grad()
    
    # 前向传播用 FP16
    with autocast():
        loss = model(video, text)
    
    # 反向传播处理精度
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

效果：

-   显存占用大幅减少
-   训练速度显著提升
-   质量几乎无损

**梯度检查点（Gradient Checkpointing）**：

不保存所有中间激活，需要时重新计算：

```text
from torch.utils.checkpoint import checkpoint

class DiTBlock(nn.Module):
    def forward(self, x, text):
        # 使用 checkpoint 包裹
        return checkpoint(self._forward, x, text)
    
    def _forward(self, x, text):
        # 实际计算
        x = self.attention(x)
        x = self.cross_attention(x, text)
        x = self.mlp(x)
        return x
```

效果：

-   显存明显减少
-   训练时间略有增加
-   适合显存不足的情况

**ZeRO 优化器（分布式训练）**：

使用 DeepSpeed ZeRO-3 跨多卡分配参数：

```text
from deepspeed import initialize

model, optimizer, _, _ = initialize(
    model=model,
    config={
        "zero_optimization": {
            "stage": 3,  # 参数、梯度、优化器状态都分片
            "offload_optimizer": {
                "device": "cpu"  # 优化器状态放 CPU
            }
        },
        "fp16": {"enabled": True},
        "gradient_checkpointing": {"enabled": True}
    }
)
```

效果：

-   多卡可训练大规模参数模型
-   单卡显存占用大幅降低
-   训练吞吐量显著提升

### 关键损失函数

除了基本的去噪损失，Wan 还使用了多个辅助损失：

**1\. 时空一致性损失**：

$ \mathcal{L}_{\text{consistency}} = \underbrace{\|\nabla_t z_t - \nabla_t z_{t-1}\|^2}_{\text{时间平滑}} + \underbrace{\|\nabla_x z_t\|_{\text{TV}}}_{\text{空间平滑}} $

确保相邻帧之间变化平滑，避免抖动。

**2\. 感知损失（Perceptual Loss）**：

$ \mathcal{L}_{\text{perceptual}} = \sum_{l}\|\phi_l(x) - \phi_l(\hat{x})\|^2 $

其中 $\phi_l$ 是预训练 VGG 的第 $l$ 层特征。比 L2 损失更符合人类感知。

**3\. 光流一致性损失**（用于 TSR）：

$ \mathcal{L}_{\text{flow}} = \sum_{t}\|I_{t+1} - \text{Warp}(I_t, F_{t\to t+1})\|^2 $

确保插值帧符合运动轨迹。

**总损失**：

$ \mathcal{L}_{\text{total}} = \lambda_1\mathcal{L}_{\text{diff}} + \lambda_2\mathcal{L}_{\text{consistency}} + \lambda_3\mathcal{L}_{\text{perceptual}} + \lambda_4\mathcal{L}_{\text{flow}} $

这些权重需要根据具体任务调整。

* * *

## 关键设计汇总

| 模块 | 技术核心 | 功能 | 计算复杂度 | 关键优势 |
| --- | --- | --- | --- | --- |
| Video VAE | 3D卷积 + 残差结构 | 压缩/解压视频 | O(THW) | 降低计算量 8-16x |
| Diffusion Transformer | 分解式3D注意力 + Cross-Attention | 建模时空 latent | O(TH²W² + T²HW) | 长程依赖建模 |
| 文本编码器 | Qwen 2.5 | 文本→语义向量 | O(seq_len²) | 中文原生支持 |
| 多阶段生成 | Base + TSR + SSR | 层级上采样 | 分阶段降低显存 | 1080p 高分辨率 |
| 训练优化 | FP16 + ZeRO + Checkpointing | 高效训练 | 显存减少 50% | 消费级卡可训练 |
| 推理加速 | DDIM / DPM-Solver | 快速采样 | 20-50 步 | 10x 速度提升 |

* * *

## 实际案例

假设我们输入这样的 prompt：

> “一个人在沙滩上奔跑，阳光明媚，海浪拍打着海岸。”

生成过程大概是这样的：

1.  **文本编码**：提取”人”、”沙滩”、”奔跑”、”阳光”、”海浪”等语义特征
2.  **初始状态**：随机 latent，看起来就是一团噪声
3.  **去噪步骤1-10**：模糊的色块开始出现，能隐约看出天空和沙滩的区域
4.  **去噪步骤11-30**：人物轮廓逐渐清晰，能看出在运动
5.  **去噪步骤31-50**：细节完善，动作连贯，海浪的动态效果出现
6.  **解码输出**：最终的高清视频，每帧都很清晰，动作流畅自然

可视化过程：

```text
第 1 步：随机噪声
    [█████████████████████]  (全是随机块)

第 10 步：粗轮廓显现
    [░░█████░░░░███░░░░░░]  (看到大轮廓)

第 30 步：细节逐渐填充
    [░░▒▒███▒▒▒▒███▒▒░░░░]  (人形、海滩、海浪)

第 50 步：最终高质量视频
    [人物 + 沙滩 + 海浪 + 光影效果]
```

整个过程就像在迷雾中逐渐看清画面，先是大的轮廓，然后是细节，最后是精致的纹理。

* * *

## 一些技术细节和坑

### 时空一致性问题

生成视频最大的挑战不是单帧质量，而是**时空一致性**。你可能会遇到：

-   人物突然变形
-   场景突然跳变
-   动作不连贯

Wan 通过时间维 attention 来解决这个问题，但还是会有玄学成分。有时候换个 prompt 写法效果就完全不同。

### 时空一致性的深层原因

**为什么会不一致？**

1.  **独立的去噪过程**： 即使有时间 attention，每一帧的去噪仍然有一定独立性。在高噪声阶段（$t$ 大），帧间的关联较弱，可能产生不同的高频细节。
2.  **Attention 的局部性**： Attention 虽然能建模长程依赖，但在实践中，由于 softmax 的特性，往往更关注近邻帧。对于 16 帧的视频，第 1 帧和第 16 帧的关联可能很弱。
3.  **噪声的累积效应**： 每步采样都可能引入微小的误差，在视频中这些误差会跨帧累积，导致”漂移”现象。

**改进方法**：

**1\. 分层生成（Hierarchical Generation）**：

先生成关键帧，再插值中间帧：

```text
# 第一阶段：生成关键帧 (如 0, 4, 8, 12, 16)
keyframes = generate_video(prompt, frames=[0, 4, 8, 12, 16])

# 第二阶段：以关键帧为条件，生成中间帧
for i in [2, 6, 10, 14]:
    frames[i] = generate_frame(
        prompt, 
        prev_frame=keyframes[i-2], 
        next_frame=keyframes[i+2]
    )
```

**2\. 光流约束（Optical Flow Constraint）**：

在训练时添加光流一致性损失：

$ \mathcal{L}_{\text{flow}} = \|I_{t+1} - \text{Warp}(I_t, F_{t\to t+1})\|_2^2 $

其中 $F_{t\to t+1}$ 是从第 $t$ 帧到第 $t+1$ 帧的光流。

**3\. 循环一致性（Cycle Consistency）**：

确保前向生成和后向生成一致：

$ \mathcal{L}_{\text{cycle}} = \|I_t - G_{\text{backward}}(G_{\text{forward}}(I_t))\|_2^2 $

**4\. 更长的上下文窗口**：

使用 sliding window attention，让每一帧能看到更多前后帧：

```text
# 普通：每帧只看前后 2 帧
attention_window = 5

# 改进：每帧看前后 8 帧
attention_window = 17
```

代价是计算量增加。

### 显存和速度

即使有 VAE 压缩，生成长视频仍然很吃显存。通常的做法是：

-   先生成短片段
-   然后用滑动窗口的方式拼接
-   或者直接分段生成再后期融合

速度受显卡性能影响较大，生成时间随分辨率和帧数增长。

### 显存瓶颈的数学分析

**显存占用的主要来源**：

1.  **模型参数**：  
    

-   Attention 层的权重矩阵
-   FFN 层的权重
-   参数量随模型规模增长

  

1.  **Attention 矩阵**：  
    

-   时间 attention 矩阵
-   空间 attention 矩阵（通常最大）
-   每层都需要存储

  

1.  **中间激活**：  
    

-   每层的输出特征
-   多层累积
-   受 batch size 和 latent 尺寸影响

  

1.  **梯度（训练时）**：  
    

-   训练时需要存储梯度
-   显存需求约为推理的2倍

  

**长视频生成策略**：

**策略1：Sliding Window（滑动窗口）**

```text
def generate_long_video(prompt, total_frames=64, window_size=16, overlap=4):
    frames = []
    
    for start in range(0, total_frames, window_size - overlap):
        end = min(start + window_size, total_frames)
        
        # 生成窗口
        if start == 0:
            window = generate_video(prompt, num_frames=window_size)
        else:
            # 以前面的帧为条件
            window = generate_video(
                prompt, 
                num_frames=window_size,
                init_frames=frames[-overlap:]  # 重叠部分作为条件
            )
        
        # 添加非重叠部分
        frames.extend(window[overlap:] if start > 0 else window)
    
    return frames
```

这种方法的问题：拼接处可能有不连续。

**策略2：AutoRegressive（自回归）**

```text
def generate_autoregressive(prompt, total_frames=64, chunk_size=16):
    frames = []
    
    # 生成第一个 chunk
    chunk = generate_video(prompt, num_frames=chunk_size)
    frames.extend(chunk)
    
    # 逐个生成后续 chunk
    while len(frames) < total_frames:
        # 以最后几帧为条件
        chunk = generate_next_chunk(
            prompt,
            condition_frames=frames[-4:],  # 最后 4 帧
            num_frames=chunk_size
        )
        frames.extend(chunk)
    
    return frames[:total_frames]
```

**策略3：Latent Space Interpolation（潜空间插值）**

生成首尾两个片段，在 latent 空间插值：

```text
# 生成开始和结束
start_latent = generate_latent(prompt_start)  # t=0 的 latent
end_latent = generate_latent(prompt_end)      # t=T 的 latent

# 线性插值
latents = []
for alpha in np.linspace(0, 1, num_frames):
    latent = (1-alpha) * start_latent + alpha * end_latent
    latents.append(latent)

# 解码
video = vae_decoder(torch.stack(latents))
```

这种方法快但控制力弱。

**显存优化技巧**：

1.  **梯度检查点（Gradient Checkpointing）**： 不存储所有中间激活，需要时重新计算。以时间换空间。
2.  **混合精度（Mixed Precision）**： 使用 float16 而非 float32，显存减半，速度显著提升。
3.  **CPU Offloading**： 将不活跃的层参数放到 CPU，需要时再搬到 GPU。

### 采样器的选择

常见的采样器有 DDPM、DDIM、DPM-Solver 等。不同采样器在质量和速度上有权衡：

-   **DDPM**：质量最好，但需要完整步数（通常1000步）
-   **DDIM**：可以用更少步数（20-50步），质量略有下降
-   **DPM-Solver**：目前最快，10-20步就能出不错的结果

实际使用中，DPM-Solver 是性价比最高的选择。

### 工程优化建议

**显存优化：**

-   梯度累积：模拟更大 batch size
-   Attention 分解：时空分离显著降低显存
-   VAE 缓存：预计算所有编码，节省训练显存

**质量优化：**

-   采样器选择：DPM-Solver++ 平衡速度与质量
-   噪声调度：采用 cosine schedule 而非 linear
-   分类器引导：在推理时增强文本控制强度

**加速策略：**

-   模型剪枝：去除低重要性头部
-   量化：8-bit 推理显著加速
-   批处理：同时生成多个视频摊销开销

* * *

## 总结

Wan 的核心思想可以概括为：**Video VAE + Diffusion Transformer + 文本交叉注意力**。

| 阶段 | 输入 | 输出 | 说明 |
| --- | --- | --- | --- |
| 编码 | 视频帧 (B,3,T,H,W) | 时空 latent (B,C,T’,H’,W’) | 降维 8~16 倍 |
| 加噪 | latent + 时间步 t | 噪声版 latent | 前向扩散 |
| 去噪 | 噪声 latent + 文本 + t | 预测噪声 | 反复调用（主要耗时） |
| 解码 | 干净 latent | 视频帧 (B,3,T,H,W) | 升维恢复 |

视频生成的核心在于”时空一致性”的建模。VAE 提供了高效的 latent 表征，Transformer 提供了长程依赖建模能力，文本条件通过跨模态注意力实现语义约束。

当然，工程实现上有很多折衷：压缩率怎么选、attention 怎么分解、采样器如何优化效率，这些都需要大量实验调参。不过整体架构还是相当优雅的。

如果你也在做视频生成相关的工作，Wan 的这套**时空 latent + 分解注意力**的思路很值得借鉴。特别是分解式注意力的设计，可以推广到其他需要处理高维数据的场景。

最后一个小建议：真要用这个模型的话，多准备些显存，多试试不同的 prompt 写法。视频生成还是个很玄学的领域，有时候一个词的差异就能让效果天差地别。

* * *

## Wan 2.x 的改进

Wan 1.x 虽然效果不错，但还有不少问题。Wan 2.x 针对这些痛点做了系统性改进，主要集中在四个方面：更长的视频、更好的一致性、更快的速度、更强的控制。

### 改进1：更长的视频生成能力

**Wan 1.x 的限制**：

-   只能生成较短的视频片段
-   超过一定长度显存或质量会出现问题
-   长视频只能通过拼接，但拼接处不连续

**Wan 2.x 的解决方案：Memory-Efficient Attention**

引入了 **Ring Attention** 机制，灵感来自 FlashAttention：

```text
class RingAttention(nn.Module):
    """
    将长序列分块处理，但保持全局信息流动
    典型: chunk_size=8-16 帧一块
    """
    def __init__(self, chunk_size=8):
        super().__init__()
        self.chunk_size = chunk_size  # 每块8帧
    
    def forward(self, x):
        # x: (B, T, H, W, C), T 可以很大
        B, T, H, W, C = x.shape
        
        # 分块
        chunks = x.split(self.chunk_size, dim=1)
        
        outputs = []
        kv_cache = None  # 保存之前 chunk 的 key/value
        
        for chunk in chunks:
            # 当前 chunk 的 Q
            q = self.to_q(chunk)
            k = self.to_k(chunk)
            v = self.to_v(chunk)
            
            if kv_cache is not None:
                # 与之前的 K/V 做 attention
                k_full = torch.cat([kv_cache[0], k], dim=1)
                v_full = torch.cat([kv_cache[1], v], dim=1)
            else:
                k_full, v_full = k, v
            
            # Attention
            attn = torch.einsum('bthwc,bTHWc->bthwTHW', q, k_full)
            attn = attn.softmax(dim=-1)
            out = torch.einsum('bthwTHW,bTHWc->bthwc', attn, v_full)
            
            outputs.append(out)
            
            # 更新 cache（只保留最近的）
            kv_cache = (k_full[:, -cache_size:], v_full[:, -cache_size:])
        
        return torch.cat(outputs, dim=1)
```

通过这种方式，可以生成更长的连贯视频，显存增长幅度可控。

### 改进2：更强的时空一致性

**Wan 1.x 的问题**：

-   人物在不同帧中可能”变脸”
-   背景元素会漂移
-   动作不够流畅

**Wan 2.x 的解决方案：Multi-Scale Temporal Modeling**

引入了**多尺度时间建模**，同时捕捉短期和长期依赖：

$ \begin{aligned} \text{Short-term}: & \quad \text{Attention}(\text{frames}_{t-2:t+2}) \\ \text{Mid-term}: & \quad \text{Attention}(\text{frames}_{t-8:t+8:2}) \\ \text{Long-term}: & \quad \text{Attention}(\text{frames}_{0:T:4}) \end{aligned} $

然后融合三个尺度的特征：

$ z_t = \alpha \cdot z_t^{\text{short}} + \beta \cdot z_t^{\text{mid}} + \gamma \cdot z_t^{\text{long}} $

**物理约束损失（Physics-Aware Loss）**：

为了让运动更合理，引入了物理约束：

$ \mathcal{L}_{\text{physics}} = \underbrace{\|\nabla_t z_t - \nabla_t z_{t-1}\|^2}_{\text{加速度平滑}} + \underbrace{\|\nabla_x z_t\|^2}_{\text{空间连续性}} $

第一项确保加速度连续（不会突然加速/减速），第二项确保空间上的平滑。

这些改进显著提升了时空一致性，包括人物 ID 保持、动作流畅度和背景稳定性。

### 改进3：MoE（Mixture of Experts）架构

**Wan 1.x 的模型瓶颈**：

-   单一模型要处理各种类型的视频（人物、风景、动物、动画…）
-   模型容量受限，难以同时精通所有领域
-   增大模型尺寸会导致计算量爆炸

**Wan 2.x 的 MoE 解决方案**：

引入 **Mixture of Experts（专家混合）** 架构，让不同的”专家”网络处理不同类型的内容。

### MoE 的基本原理

核心思想：不是所有参数都参与每次计算，而是根据输入**动态选择**少量专家。

**传统 FFN（Feed-Forward Network）**：

$ \text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 \cdot x) $

所有参数都参与计算。

  

**MoE FFN**：

$ \text{MoE}(x) = \sum_{i=1}^{N} G(x)_i \cdot E_i(x) $

  

其中：

-   $E_i(x)$：第 $i$ 个专家网络（每个专家都是一个 FFN）
-   $G(x)_i$：门控网络（Gating Network），决定每个专家的权重
-   $N$：专家总数（如 8 或 16）

**门控机制（Gating）**：

门控网络决定哪些专家被激活：

```text
class MoELayer(nn.Module):
    def __init__(self, dim, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k  # 每次只激活 top-k 个专家
        
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            ) for _ in range(num_experts)
        ])
        
        # 门控网络（决定用哪些专家）
        self.gate = nn.Linear(dim, num_experts)
        
    def forward(self, x):
        # x: (B, T, H, W, C)
        batch_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])  # (B*T*H*W, C)
        
        # 计算门控分数
        gate_logits = self.gate(x_flat)  # (B*T*H*W, num_experts)
        
        # Top-K 选择
        top_k_gates, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_gates, dim=-1)  # 归一化
        
        # 初始化输出
        output = torch.zeros_like(x_flat)
        
        # 只计算被选中的专家
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            gate_weight = top_k_gates[:, i:i+1]
            
            # 将 token 分配给对应的专家
            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[mask] += gate_weight[mask] * expert_output
        
        return output.reshape(*batch_shape, -1)
```

**负载均衡损失（Load Balancing Loss）**：

为了防止所有 token 都选择同一个专家（导致其他专家不学习），引入负载均衡：

$ \mathcal{L}_{\text{balance}} = \alpha \cdot \sum_{i=1}^{N} f_i \cdot P_i $

其中：

-   $f_i$：分配给专家 $i$ 的 token 比例
-   $P_i$：专家 $i$ 的平均门控概率

这个损失鼓励负载均匀分布。

### Wan 2.x 中的专家分工

在视频生成中，不同专家自然地学会了处理不同类型的内容：

**专家分工可视化**：

```text
专家 0：人物面部细节（眼睛、嘴巴、表情）
专家 1：人体动作（姿态、运动）
专家 2：自然场景（天空、树木、水）
专家 3：建筑结构（线条、几何）
专家 4：纹理细节（布料、皮肤、材质）
专家 5：光影效果（高光、阴影、反射）
专家 6：动态效果（水流、烟雾、火焰）
专家 7：抽象/风格化内容（动画、艺术风格）
```

这种分工是**自动涌现**的，不需要人工标注！

### MoE 的优势

**1\. 更大的模型容量，相同的计算量**

举例说明：

-   传统 Dense 模型：10B 参数，每次前向传播用全部 10B 参数
-   MoE 模型：80B 总参数（8 个专家×10B），但每次只激活 top-2 = 20B 参数实际计算

虽然 MoE 模型总参数多得多，但**激活参数**（实际计算的）可以控制在合理范围。

MoE 的关键优势：用相同的计算量（激活参数数），但通过更大的总参数量获得更强的模型容量。

**2\. 专业化能力**

每个专家专注于特定领域，比”通才”模型更精通：

-   处理人脸的专家比通用模型更擅长面部细节
-   处理风景的专家更理解自然规律

**3\. 可扩展性**

增加专家数量，模型总容量线性增长，但激活参数比例下降：

```text
4 专家 (top-2 激活) → 激活率 50%
8 专家 (top-2 激活) → 激活率 25%
16 专家 (top-2 激活) → 激活率 12.5%
```

### MoE 的挑战和解决方案

**挑战1：通信开销**

在分布式训练中，token 需要在设备间传输给对应的专家。

**解决方案：Expert Parallel + Data Parallel 混合**

```text
# 专家分布策略
# 假设 8 个专家，4 张卡
# 每张卡放 2 个专家

GPU 0: [Expert 0, Expert 1]
GPU 1: [Expert 2, Expert 3]
GPU 2: [Expert 4, Expert 5]
GPU 3: [Expert 6, Expert 7]

# 使用 All-to-All 通信
# 只传输被激活的 token，减少通信量
```

**挑战2：专家崩溃（Expert Collapse）**

某些专家从不被选中，变成”死专家”。

**解决方案：**

1.  初始化时给每个专家不同的偏置
2.  动态调整门控网络，增加探索性
3.  使用辅助损失鼓励多样性

**挑战3：推理时的显存**

虽然只激活部分专家，但所有专家参数都要加载到显存。

**解决方案：动态加载（对超大模型）**

```text
class DynamicMoE(nn.Module):
    def __init__(self, experts_on_cpu=True):
        super().__init__()
        self.experts_on_cpu = experts_on_cpu
        
    def forward(self, x):
        # 计算门控
        top_k_indices = self.gate(x)
        
        # 只加载需要的专家到 GPU
        if self.experts_on_cpu:
            active_experts = []
            for idx in top_k_indices.unique():
                expert = self.experts[idx].cuda()  # 临时加载
                active_experts.append(expert)
            
            # 计算
            output = self._compute(x, active_experts)
            
            # 卸载回 CPU
            for expert in active_experts:
                expert.cpu()
        else:
            output = self._compute(x, self.experts)
        
        return output
```

### 实际性能提升

MoE 架构在各个视频类型上都带来显著的质量提升，包括人物、风景、动物和动画风格。

在效率方面，MoE 的推理延迟与 Dense 模型相当甚至略优，但显存占用会增加一些，这是为了加载更多专家参数。

### 与其他技术的结合

MoE 可以和其他改进叠加：

```text
Wan 2.x = Ring Attention（长视频）
         + MoE（更强能力）
         + Progressive Distillation（更快速度）
         + Multi-Scale Modeling（更好一致性）
         + Multi-Modal Control（更强控制）
```

这些技术不是互斥的，可以组合使用！

**组合效果**：

| 配置 | 质量 | 速度 | 显存 | 适用场景 |
| --- | --- | --- | --- | --- |
| 基础版（Dense） | ★★★ | ★★★★ | ★★★★ | 快速原型 |
| +MoE | ★★★★ | ★★★ | ★★★ | 追求质量 |
| +Ring Attn | ★★★★ | ★★ | ★★★ | 长视频 |
| +Distillation | ★★★ | ★★★★★ | ★★★★ | 实时应用 |
| 全部组合 | ★★★★★ | ★★ | ★★ | 研究/高端应用 |

### 改进4：更快的推理速度

**Wan 1.x 的速度瓶颈**：

-   需要较多的采样步数，每步都要过一遍完整的 DiT
-   生成时间较长

**Wan 2.x 的加速策略：Progressive Distillation**

使用**渐进式蒸馏**，将多步模型逐步蒸馏成少步模型：

**蒸馏策略**：逐步减半

$ \mathcal{L}_{\text{distill}} = \mathbb{E}\left[\|z_{t/2}^{\text{student}} - z_{t/2}^{\text{teacher}}\|^2\right] $

  

学生模型用 1 步模拟教师模型的 2 步。

通过多轮蒸馏，最终得到的少步模型在保持接近质量的同时大幅加速。

**Latent Consistency Model（LCM）**：

另一个加速方法是使用一致性模型，直接从噪声跳到结果：

$ f_\theta(z_t, t) = z_0 \quad \text{（一步到位！）} $

训练时确保一致性：

$ \mathcal{L}_{\text{consistency}} = \|f_\theta(z_t, t) - f_\theta(z_{t'}, t')\|^2 $

对于任意两个时间步，预测的 $z_0$ 应该一致。

蒸馏模型实现了显著加速，质量损失可控。

### 改进5：更精细的控制能力

**Wan 1.x 的局限**：

-   只能用文本控制，很难精确指定内容
-   无法控制摄像机运动
-   不支持局部编辑

**Wan 2.x 的新控制方式**：

### 4.1 多模态条件输入

支持多种条件的组合：

```text
# 文本 + 参考图像 + 姿态序列
conditions = {
    'text': "一个人跳舞",
    'reference_image': first_frame,  # 指定人物外观
    'pose_sequence': pose_keypoints,  # 控制动作
    'camera_motion': 'zoom_in'       # 控制镜头
}

video = model.generate(conditions)
```

**ControlNet 集成**：

引入 ControlNet 架构，支持多种结构化控制：

```text
class VideoControlNet(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        # 复制 base model 的权重
        self.control_net = copy.deepcopy(base_model.encoder)
        # 零卷积层（初始时不影响生成）
        self.zero_convs = nn.ModuleList([
            nn.Conv3d(dim, dim, 1) for dim in dims
        ])
        # 初始化为 0
        for conv in self.zero_convs:
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)
    
    def forward(self, x, condition):
        # 提取控制信号
        control_features = self.control_net(condition)
        
        # 通过零卷积注入到主网络
        control_residuals = [
            conv(feat) for conv, feat in zip(self.zero_convs, control_features)
        ]
        
        return control_residuals
```

支持的控制类型：

-   **深度图**：控制 3D 结构
-   **边缘图**：控制物体轮廓
-   **姿态**：控制人物动作
-   **语义分割**：控制场景布局
-   **光流**：控制运动轨迹

### 4.2 摄像机控制

显式建模摄像机参数：

$ \text{Camera} = \{\underbrace{[x, y, z]}_{\text{位置}}, \underbrace{[\theta, \phi, \psi]}_{\text{角度}}, \underbrace{f}_{\text{焦距}}\} $

通过 Plücker 坐标嵌入摄像机信息：

```text
def camera_embedding(camera_params):
    """
    将摄像机参数编码为可学习的嵌入
    """
    pos = camera_params['position']      # (3,)
    rot = camera_params['rotation']      # (3,)
    focal = camera_params['focal_length'] # (1,)
    
    # Plücker 坐标
    ray_origin = pos
    ray_direction = rotation_to_direction(rot)
    moment = torch.cross(ray_origin, ray_direction)
    
    # 6D 表示
    plucker = torch.cat([ray_direction, moment])  # (6,)
    
    # 投影到高维空间
    emb = self.camera_encoder(plucker)  # (6,) -> (C,)
    
    return emb
```

这样可以生成特定镜头运动的视频：

-   **推拉（Dolly）**：镜头前后移动
-   **摇镜（Pan）**：水平旋转
-   **俯仰（Tilt）**：垂直旋转
-   **环绕（Orbit）**：绕物体旋转

### 4.3 局部编辑能力

支持视频的局部编辑，保持其他区域不变：

```text
def local_edit(video, mask, new_prompt):
    """
    只编辑 mask 区域
    """
    # 1. 编码原视频
    z0 = vae.encode(video)
    
    # 2. 加噪到中间时刻 t
    t = 250  # 不需要加太多噪声
    noise = torch.randn_like(z0)
    zt = sqrt_alpha[t] * z0 + sqrt_one_minus_alpha[t] * noise
    
    # 3. 去噪，但只在 mask 区域应用新 prompt
    for t in reversed(range(t)):
        # 预测噪声
        eps_new = model(zt, t, new_prompt)      # 新区域
        eps_old = model(zt, t, original_prompt) # 旧区域
        
        # 混合
        mask_3d = mask.unsqueeze(1)  # (B, 1, T, H, W)
        eps = mask_3d * eps_new + (1 - mask_3d) * eps_old
        
        # 去噪
        zt = scheduler.step(eps, zt, t)
    
    # 4. 解码
    edited_video = vae.decode(zt)
    
    return edited_video
```

### 改进6：更好的数据和训练策略

**数据方面**：

Wan 2.x 使用了更大规模、更高质量的数据：

-   **数据量**：大幅增加
-   **分辨率**：提升训练分辨率
-   **时长**：支持更长视频片段
-   **质量过滤**：使用 CLIP 和美学评分模型过滤低质量数据

**合成数据增强**：

使用游戏引擎生成带有完美标注的数据：

```text
# 从 Unity/Unreal 生成
synthetic_data = {
    'rgb': video_frames,
    'depth': depth_maps,
    'normal': normal_maps,
    'segmentation': semantic_masks,
    'optical_flow': flow_maps,
    'camera_params': camera_trajectory
}
```

合成数据的使用显著提升了模型对 3D 结构和物理规律的理解。

**训练策略改进**：

1.  **分辨率渐进训练**：  
    

-   从低分辨率、短视频开始
-   逐步提升到高分辨率、长视频
-   多阶段逐步增强

  

1.  **混合精度 + ZeRO 优化**：  
    

-   使用 DeepSpeed ZeRO-3 分布式训练
-   模型参数分片到多卡
-   梯度实时通信
-   优化器状态动态交换

  

1.  **课程学习（Curriculum Learning）**：  
    

-   从简单到复杂的视频
-   先训练静态场景
-   再训练简单运动
-   最后训练复杂动作

  

### 核心改进总结

Wan 2.x 相比 1.x 的主要改进：

1.  **更长视频**：通过 Ring Attention 支持更多帧
2.  **更好一致性**：多尺度时间建模和物理约束
3.  **MoE 架构**：更强的模型容量和专业化能力
4.  **更快速度**：渐进式蒸馏实现加速
5.  **更强控制**：多模态条件和摄像机控制
6.  **更好训练**：数据规模和训练策略优化

Wan 2.x 的这些改进不仅仅是量的提升，更是质的飞跃。特别是 Ring Attention 和 Progressive Distillation 这两个技术，可以说是视频生成领域的重要突破。如果你在做相关工作，强烈建议关注这些新技术！