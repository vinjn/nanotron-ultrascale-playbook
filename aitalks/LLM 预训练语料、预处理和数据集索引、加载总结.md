# LLM 预训练语料、预处理和数据集索引、加载总结

**作者：** AI闲谈

---

一、背景介绍

LLM 的模型参数量很大，其模型效果也需要巨大的语料库支撑，LLM 预训练需要的 Token 数已经从早期的 300B Token 逐渐增加到 1.4T，甚至进一步扩展到 3T 以上。本文中我们具体介绍 LLM 预训练语料库的来源，构建语料库的预处理过程以及 LLM 预训练的 Dataset 存储、混合、加载方式。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgw4vryJ18uicrImtQ0R24negMLbRkJt8b7zWkG4iaXyPVLVeKrDZUsic2vaJlMH4L6xzy8icE1ib1f3aQ/640?wx_fmt=png&from=appmsg&randomid=be8a9ncc)

## 二、常见语料库

虽然不同 LLM 的模型大小不同，预训练的 Token 数也各不一样，但是其原始的语料都大同小异，主要有几种类型：CommonCrawl、Wikipedia、Books、Code、ArXiv、Reddit links 等。

### 2.1 CommonCrawl

CommonCrawl 是一个免费、开放的网络爬虫数据集，旨在提供大规模的网页抓取数据，使研究人员、开发者和数据科学家能够访问和分析互联网上的信息。该数据集由 Common Crawl Foundation 维护，该基金会是一个非营利性组织，致力于促进网络信息的开放共享。

CommonCrawl 数据集非常大，并且在不断地更新中，具体可参考 Common Crawl - Overview，其中最新的 CC-MAIN-2023-50 共包含 3.35B 个网页，压缩后的数据超过 130TB。具体如下图所示：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgw4vryJ18uicrImtQ0R24nerj1hhVKQYLNEzM1POXHBqY0NAvlCazmicq7ljUwwJRGhEXxZM4mZWZA/640?wx_fmt=png&from=appmsg&randomid=0ill9tmk)

由于 CommonCrawl 数据集过于庞大，并且包含很多噪声，处理的成本很高，因此也有其他研究者提供了相应处理过的子集，比如 C4（Colossal Clean Crawled Corpus），可以参考 GitHub - google-research/text-to-text-transfer-transformer。

### 2.2 Wikipedia

Wikipedia 是一个由全球志愿者维护的在线百科全书项目。其包含多种语言，涉及的领域也非常广，并且质量非常高。比如如下图所示，“Large language model” 页面有 29 种语言，并且分了各个部分进行介绍：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgw4vryJ18uicrImtQ0R24neweUYon3XtVY3SJSQZoFcvZSjTvas5onPX9RxboIf2daoZQOMKuW4icg/640?wx_fmt=png&from=appmsg&randomid=eykxi5av)

### 2.3 Books

书籍是另一种高质量的语料库，与其他语料库相比，其涵盖的语言、领域更加广泛，内容也更加正式。总体来说，使用书籍作为语料库预训练 LLM 可以获得如下好处：

- 广泛的知识涵盖：书籍包含很多领域，比如科学、历史、文学以及技术等。书籍能使 LLM 接触丰富多样的知识，有助于提高其对各种主题的理解和表达能力。
- 丰富的语言表达：书籍中通常包含丰富而复杂的语言表达，包括各种风格、修辞和专业术语。通过学习书籍中的语言使用方式，LLM 可以更好地捕捉到语境、上下文和语法结构，提高其生成自然语言的能力。
- 专业的领域知识：一些书籍涉及特定领域的深度知识，如科学、法律、医学等。在 LLM 的训练中使用这些书籍可以使模型更好地理解和生成与这些领域相关的内容。
- 多样性的文本结构：书籍中的文本结构多种多样，包括章节、段落、脚注等。通过训练模型处理这些不同层次和结构的文本，有助于提高其对复杂文档和长文本的理解和处理能力。
- 知识结构和推理能力：书籍中的内容通常有一定的逻辑和知识结构，通过训练模型学习这些结构，可以提高其在理解和生成逻辑推理、连贯性论述方面的能力。
- 语言多样性：书籍中使用的语言可能涵盖多种方言、俚语和文学风格，这有助于训练模型更好地理解和生成多样化的语言表达。

### 2.4 Code

当前很多 LLM 预训练语料中也会加入 Code，比如来自 Github、Gitlab 或者编程问答网站（比如 StackOverflow）的语料，因为其不仅对 LLM 理解编程语言，代码注释和生成代码很有帮助，也有研究表明其对 LLM 的推理能力至关重要。

### 2.5 ArXiv

ArXiv（https://arxiv.org/） 是一个包含各个学科领域的预印本（Preprint）平台，涵盖数学、物理、计算机科学等多个学科，包含大量领域特定的术语和语言，比如数学符号、专业术语等。在预训练语料中加入 ArXiv 中的论文可以使 LLM 接触到广泛的学术知识、提高对不同学科的理解能力。

### 2.6 Stack Exchange

Stack Exchange （https://stackexchange.com/）是一个高质量问答网站，涵盖从计算机科学到化学等多个领域。

## 三、数据预处理

### 3.1 概述

收集到预料数据往往不会直接用于预训练，因为其通常比较大，并且包含很多冗余和噪声，需要进一步的过滤清理，此外，有些数据（尤其抓取的网页数据）中还可能包含一些敏感信息，比如个人的身份信息、家庭信息以及其他色情或者敏感信息，都需要进一步的处理。

如下图 Fig. 7 （来自 [2303.18223] A Survey of Large Language Models）所示，常见的数据处理包含质量过滤（Quality Filtering）、去重（De-deplication）、隐私擦除（Privacy Reduction）、Tokenization、数据混合等：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgw4vryJ18uicrImtQ0R24neeERW07wMciaUoAWFJDQQBUzAfMSN9Uvvj4Ak9qpXo62XrebobPHJpLg/640?wx_fmt=png&from=appmsg&randomid=etxa9494)

### 3.2 LLaMA-1

LLaMA 是 Meta 发布的 LLM，如下图所示为 LLaMA-1 中预训练语料的来源、混合比例等统计信息，经 Tokenizer （采用 BPE 算法）后生成 1.4T Token：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgw4vryJ18uicrImtQ0R24nerMTDNib4JBALHMVNENqe70taFic5PvFkgRWHbV8n2fVUgNQFm2wEyRGA/640?wx_fmt=png&from=appmsg&randomid=2xh9782o)

针对不同的数据采用了不同的预处理策略：

- English CommonCrawl：使用 CCNet Pipeline 处理了 2017 年到 2020 年的 5 个 CommonCrawl 数据。预处理包括：按行进行重复数据删除、基于 fastText 语言分类器删除非英语数据、使用 ngram 语言模型删除低质数据。
- C4：虽然 C4 数据集来自 CommonCrawl，但是作者发现使用不同预处理的 CommonCrawl 数据集可以提升性能，因此依旧使用了 C4。同时也应用了重复数据删除、语言识别相关操作。
- Github：来自 Google 的 BigQuery，只保留符合 Apache、BSD 和 MIT 许可的项目。此外，根据行的长度或字母和数字的比例采用启发式方法过滤低质量文件。此外，也采用正则表达式删除标题等内容。最后，还在文件级按匹配度进行去重。
- Wikipedia：来自 2022 年 6月-8月版本，共 20 种语言。去除了超链接、注解和其它格式化样式。
- Books：包含两部分数据，来自 Gutenberg 项目的书籍，以及来自 The Pile 的 Books3 部分。按照书籍粒度进行去重，删除超过 90% 重复内容的书籍。
- ArXiv：作者以 Latex 格式处理 ArXiv 论文，并删除第一节之前的所有内容以及参考文献，同时删除 .tex 文件中的注释、用户自己编写的定义和宏，以提高论文的一致性。
- Stack Exchange：作者仅保留 Stack Exchange 中 28 个最大的主题，并从文本中删除 HTML 标签，按分数（从高到低）对答案进行排序。

### 3.3 RefinedWeb

RefinedWeb 是阿布扎比的 TII 基于 CommonCrawl 构建的语料库，其数据集经过严格的过滤和去重，总共包含 5T Token，不过只开源了 600B Token 的子集。同时还发布了 Falcon-7B 和 Falcon-40B，分别基于 1.5 T Token 和 1 T Token 训练而来，其 80% 以上数据来自 RefinedWeb。

如下图所示为 RefinedWeb 的构建过程，其包含几种主要的预处理过程：

- URL filtering：目的是为了过滤掉欺诈、色情、暴力和赌博等网站。作者首先收集了一个 4.6M 的黑名单网站列表，然后构建了一个 URL 打分机制，以避免黑名单中的误报，比如流行的博客、医疗、法律等页面。此外，后续的预训练中也要与其他优质语料库混合，因此也过滤了高质量的语料来源，比如 Wikipedia 和 arXiv 等。
- Text extraction：提取页面中的主要内容，忽略目录、页眉、页脚和广告等。
- Language identification：使用 CCNet 中的 fastText 语言分类器，可以将文档分类为 176 种语言，作者删除了排名靠前的语言中得分低于 0.65 的文档。此阶段清理了一半左右的文档。
- Repetition removal：作者基于启发式方法删除了重复的行、段落等内容。
- Document-wise filtering：文档中包含了很多机器生成的垃圾邮件等内容，其不适于语言建模，作者通过总长度、字符和单词比率等因素删除了异常页面。这个阶段需要针对每种语言调整。
- Line-wise corrections：许多文档的行中存在无关的交错，比如社交媒体的点赞、导航按钮等。因此，作者设计了一个行校正器，对行进行修正，同时如果修改超过文档的 5%，则删除整个文档。
- Fuzzy deduplication：作者在文档级别采用了 MinHash 来删除相似的文档。
- Exact deduplication：作者在序列级别采用精确字符串匹配的方式进一步对文档进行去重，可以删除特定的内容，比如特定的免责声明或通知等。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgw4vryJ18uicrImtQ0R24nen2Rjib1jC1QQyy0kQI5Zswiao1GtHRPQY2mzqtkOxE9iaDt07T6XVtlAg/640?wx_fmt=png&from=appmsg&randomid=hs4si4td)

### 3.4 Baichuan 2

Baichuan 2 是百川发布的 LLM，其构建了中英语料数据集，预训练语料同样包含网页、书籍、论文和代码。大致的数据分布如下图 Figure 1 所示（图片来自 [2309.10305] Baichuan 2: Open Large-scale Language Models）：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgw4vryJ18uicrImtQ0R24nexNBHv7Dk4aFPGK0TWTsIOic6FfESIcSdFzEPNujFZVpIibZzNBoXSlXw/640?wx_fmt=png&from=appmsg&randomid=smft5bxh)

对于数据处理，作者聚焦在数据频率和质量上，数据频率主要依赖聚类和去重。关于数据去重和聚类，Baichuan 2 采用基于 LSH 特征和稠密 Embedding 特征的方案。根据聚类，可以对单个文档、段落和句子进行重复数据删除以及打分，然后这些分数也可以用于预训练中的数据采样。

其整个数据预处理过程及各阶段清理的数据占比如下图 Figure 2 所示，其中灰色部分为删除的数据比例：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgw4vryJ18uicrImtQ0R24nemCyqVhGCqoP01wGicDPVPhaImJ6ml2Suc6ozfWlibibJOwtgPF4zx5IfA/640?wx_fmt=png&from=appmsg&randomid=8qnuuzfs)

Baichuan 2 的 Tokenizer 同样采用 BPE（来自 SentencePiece），Tokenizer 后包含 2.6T Token，如下图 Table 2 所示为 Baichuan 2 与其他模型的词表大小及压缩率比较：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgw4vryJ18uicrImtQ0R24ne6cEBibJLjxkOlLtX9ia0vhocwtWQ4fzAIUzSXaNz3KODp9hWolbzVsxA/640?wx_fmt=png&from=appmsg&randomid=89lkkkbk)

### 3.5 Qwen

Qwen 是阿里发布的 LLM，其预训练数据集包含网页、百科全书、书籍以及代码等，其数据集同样是多语言的，主要是英文和中文。

为了保证预训练数据的质量，作者同样采用了全面的预处理：

- 针对网页数据，先从 HTML 中提取文本并使用语言识别工具来确定语言。
- 为了增加数据多样性，同样进行了数据去重，包括归一化后的精确匹配去重和使用 MinHash 和 LSH 算法的模糊重复去重。
- 为了过滤低质数据，作者结合了基于规则和基于机器学习的方法，具体来说，使用多种模型对内容进行打分，包括语言模型、文本质量评分模型以及识别潜在攻击性和不当内容的模型。
- 为了进一步提高数据质量，作者有选择地对某些数据进行上采样，以确保模型接受更多高质量数据。
- 此外，有些研究表明，在预训练中使用多任务指令数据可以增强其 zero-shot 和 few-shot 数据，因此作者额外添加了高质量的指令数据。

其 Tokenizer 遵照 GPT-3.5 和 GPT-4，同样采用 BPE 方法，其词表大小为 152K，最终生成 3T Token。

### 3.6 Skywork

Skywork-13B 是昆仑万维的天工团队发布的 LLM，其首先构建了一个巨大的、高质量的数据集 SkyPile，超过 6T Token，并开源了一个 150B Token 的子集，其原始语料为网页数据，地址为 Skywork/SkyPile-150B · Datasets at Hugging Face。SkyPile 的数据包含多个来源，绝大部分是公开可访问的。

Skywork 的数据处理和其他模型类似，包含几个部分：

- Structural Extraction：数据集主要来源是网页，因此第一阶段的目标是提取相关内容，同时删除导航栏、特定站点的联系信息等无关文本元素，保留连续的中长文本段落。
- Distribution Filtering：LLM 预训练往往需要包含广泛的领域知识，之前的模型通常是为每个文档或网页分配类别标签，从而手动决定语料库的组成。而本文中，作者摒弃了以标签为中心的方法，核心是对文本片段之间的语义亲和性进行测试，从而识别并删除重复率高的文本块。
- Deduplication：本文中作者把 Deduplication 当作 Distribution Filtering 的一部分。
- Quality Filtering：作者同样使用 CCNet Pipeline 来执行过滤任务，以删除劣质内容和排除中文、英文以外的页面。作者训练了一个二分类器，来预测给定网页适合作为 Wikipedia 参考的可能性。这一阶段的结果会被组织成不同的质量类别，并且只保留高质量的组。

其 Tokenizer 同样采用 BPE，词表分布如下图 Table 2 所示：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgw4vryJ18uicrImtQ0R24neTsYtefekibJHcnNJiaMdkDHgd6p1WATcIclibFOMKBKNUuAhQtbQEQ6Tg/640?wx_fmt=png&from=appmsg&randomid=s7uf1rho)

Skywork-13B 模型的预训练语料包含 3.2T Token，从 SkyPile 采样而来，其预训练分为两个阶段，第一阶段使用 2T Token，分布如下图 Table 1 所示，第二阶段采样剩下的 1.2T Token：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgw4vryJ18uicrImtQ0R24nexlnT4T4VAHibL1yaCr64Omib1zGUNicDsNvu4RwgiabnldCHia0sPAuibBicw/640?wx_fmt=png&from=appmsg&randomid=ij9k1sew)

### 3.7 DeepSeek

在 CC_Cleaner：一种丝滑高效且易扩展的数据清洗流程 中也介绍了幻方 AI LLM 的详细数据集预处理流程，大体过程类似，这里不再具体介绍。

### 3.8 总结

从上可以看出，预训练语料预处理基本上涵盖几个步骤：

- 过滤：
- 按 URL 过滤：按网站剔除欺诈、成人等网站页面，同样也可以删除与其他数据集可能重复的页面，比如 Wikipedia、arXiv 等。
- 按语言过滤（fastText 语言分类）：比如只保留英文和中文。
- 按质量过滤：比如使用语言模型判断序列的困惑度，然后删除低质数据。
- 去重：
- 文档级去重：采用 MinHash 等方式删除高度重复的文档。
- 句子级去重：通过精确字符串匹配，或者基于 Embedding 的语义相似性删除重复的语句。
- 交叉去重：预训练语料可能来自不同的数据集，也需要交叉比对去重，比如从 CommonCrawl 中删除 Wikipedia、arXiv 等数据；此外，通常也需要删除与评估集重叠的数据，避免数据泄露。
- 隐私和许可：
- 隐私去除：为了避免隐私泄露，需要移除或者脱敏隐私数据，比如个人身份信息，姓名、身份证、住址等。
- 许可规范：在采集预训练语料时需要遵循相应的许可协议，避免出现侵权等问题。比如代码数据中只保留符合 Apache、BSD 和 MIT 许可的项目。
- 分词：预训练语料在输入模型之前都需要经过 Tokenizer 分词，大部分模型都采用 BPE 算法。
- 数据混合：预训练语料通常包含不同的数据集，有些质量比较高，比如书籍、Wikipedia 等，然而其数据量可能不多，此时在真正训练的时候可以给予更高的采样权重。

## 四、数据存储和加载

目前很多 LLM 预训练会采用 NVIDIA 的 Megatron-LM 项目或者是 Microsoft 基于此改进的 DeepSpeed-Megatron 项目，其预训练数据集的存储格式和加载方式是一致的，此处我们以此为例进行介绍。

### 4.1 原始 Dataset 结构

实际的预训练语料在训练之前都会先经过 Tokenizer 分词，转换为 Binary 数据集（还没有 shuffle 和采样）。分词后的数据都以 Token ID 的方式存储，数据的大小基本等价于 Token 数乘以每个 Token ID 的字节数。

- 如果词表大小比较小，小于 65536，则可以用 Uint16 表示，存储占磁盘大小基本等于 2*Token 数。
- 很多中文 LLM 需要包含中文词表，词表数往往超过这个限制，需要使用 Uint32，导致数据大小变为 4*Token 数。

同时，数据集通常包含不同来源，比如 CommonCrawl，Wikipedia，而且数据集有大有小，为了避免单个数据集过大，会将数据集切分为不同的 part，每个 part 都相当于一个新的子集，但是来自同一个数据集的不同 part 需要有相同的权重。此外，每个 part 都有 idx 和 bin 两个文件。如下所示为一些子数据集的示例：

- en-CommonCrawl-part18.idx
- en-CommonCrawl-part18.bin
- en-Wikipedia-part0.idx
- en-Wikipedia-part0.bin

其中 idx 文件对应索引文件，bin 对应实际的 Token ID，如下图所示：

- Index：包含 Head 和 Buffer 两部分（实际是连续的）
- Head：存储 magic、version、dtype、len 和 doc_count
- Buffer: 存储 Bin 中 Document 的起始位置和大小
- Bin：存储实际的 Document，比如根据 points[m] 和 sizes[m] 即可以从 Bin 中获得 Document m。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgw4vryJ18uicrImtQ0R24nePyuU8iaQQD5SDqKf812atIWibJ3FM7TMiaMcBIlIfFeqheJfWWn966EQA/640?wx_fmt=png&from=appmsg&randomid=09opm7va)

需要说明的是，每个 Document 都已经 Token 化，并且已经添加过起始 Token <s> 和终止 Token </s>。

### 4.2 GPT Dataset 结构

在 Dataset Blending 混合阶段可以计算获得每个 GPT Dataset 数据需要生成多少个 Sample，根据 Sample 数目也可以进一步计算得到该 Dataset 需要过几个 Epoch。

如下图所示：

- _num_tokens：根据 Dataset 中每个 Document 的 size 累积即可获得当前 Dataset 总的 Tokens 数目。
- _num_epochs：根据需要采样的 num_samples，序列长度（seq_length），每个 Epoch 总的 Tokens 数目（tokens_per_epoch）即可以计算获得数据需要过几个 Epoch。需要说明的是，倒数第二行的（total_tokens - 1）是因为 Sample 中的 Token 个数会比 seq_length 多 1 个，但是多的这一个又会作为下一个 Sample 的起始 Token，因此总共需要的 Tokens 数目为 num_samples * seq_length + 1。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgw4vryJ18uicrImtQ0R24nekFjzgr6elz4TD5KzLjqa6hicpMmyicmDIIyM3mKoFHOhZia6nk8dOGthQ/640?wx_fmt=png&from=appmsg&randomid=ew3a04ss)

此外，Dataset 还需要有一定的 Shuffle 操作，总的来说有两个 Shuffle：

- Dataset 中的不同 Document 会进行 Shuffle，对应 doc_idx。
- Dataset 中的 Sample 也会 Shuffle，对应 shuffle_idx。

如下图所示：

- shuffle_idx：长度为 num_sample，存储的是 shuffle 过的 Sample 索引，图中的示例为 16 个 Sample，也就是实际训练的时候先读 Sample 5，然后读 Sample 1，以此类推。
- doc_idx：根据原始数据可以知道数据集中有多少个 Document，然后根据该数据集需要重复的 Epoch 数目即可以得到总共需要有多少个 Document。图中的示例假设 Dataset 中总共有 0, 1, 2, 3, 4, 5 这 6 个 Document，对应的 Epoch 为 4，需要说明的是，最后一个 Epoch 需要特殊处理，后续再介绍。
- sample_idx：长度为 num_sample + 1，因为其中存储的是 Sample 的起始位置和终止位置，也就是该 Sample 对应了 doc_idx 中的哪些 Document。
- 比如 shuffle_idx 中的绿色 4 这个 Sample 由 sample_idx 中第 4 个位置和第 5 个位置确定。
- sample_idx 的第 4 个位置的 idx 对应起始的 Document 索引的位置，也就是绿色箭头指向的 doc_idx 中的 3，而 offset 为 2，则表明要从 Document 3 的第 2+1=3 个 Token 开始。
- sample_idx 的第 5 个位置的 idx 对应终止的 Document 索引的位置，也就是绿色箭头指向的 doc_idx 中的 0，而 offset 为 3，则表明要从 Document 0 的第 3 个 Token 终止，但是每个 Sample 中在结束位置需要有一个额外的 Token，因此实际是从第 4 个 Token 终止。
- sample：根据上述的起始 Document idx 和 offset 以及终止 Document idx 和 offset 即可以获取到最终的 Sample，其中最后一个 * Token 也会被下一个 Sample 复用。因为 GPT 训练的目的是让每一个位置预测下一个位置的 Token，因此 input 和 target 正好交错 1 个 Token，这也是为什么 Sample 中的 Token 数目要比 seq_length 多 1。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgw4vryJ18uicrImtQ0R24neIiarOXj1nFf60sUUXC0AaSPQ0uD0qhUZUDYMicGqxqnG0MFracHDPRKw/640?wx_fmt=png&from=appmsg&randomid=kbfiqx41)

如下图所示为 GPTDataset 中获取一个 Sample 的示例：

- 首先，从 shuffle_dix 中获取 shuffle 过的 sample index
- 然后，从 sample_idx 中获取第一个和最后一个 Document 的 idx 的位置和 offset
- 最后，根据 idx 的起始和终止位置可以获得当前 Sample 由哪些 Document 组成，当然，其中的第一个和最后一个 Document 还需要根据 offset 截断。需要说明的是，有些 Document 比较长，因此有可能存在一个 Sample 来自同一个 Document，也就是 doc_index_f 等于 doc_iindex_l。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgw4vryJ18uicrImtQ0R24ne3cJRILYvAdp1CQwM5Q5e1NZT0QdMZMLc4eNdC9HDqxKsk7rlicvreWw/640?wx_fmt=png&from=appmsg&randomid=wh8rc9ji)

如下所示，如果第一步不取 shuffle_idx，也就是直接打印真实的第 0,1,2 个 Sample，可以看出 Sample 的长度为 4097（seq_length==4096），并且每个 Sample 中的最后一个和下一个 Sample 的第一个重叠，比如 1670 和 10870。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgw4vryJ18uicrImtQ0R24nehyicXZzn16siaictia4C2fIFpn9YXLx4n9tICcVkUsia9KmfxfjMBticVs2A/640?wx_fmt=png&from=appmsg&randomid=uzocckg6)

如下图所示为构建 doc_idx 的过程，可以看出其最后一个 Epoch 进行了特殊处理，这是因为根据 num_sample 算出来的 Epoch 很可能不是整数，也就是必然有些 Document 重复多，有些重复少，但需要尽可能地保证所有 Document 的采样概率相同。

比如，Document [0, 1, 2, 3] 重复 2.5 次：

- 如果直接全局 shuffle 再截断，比如从 [2, 0, 3, 2, 1, 3, 0, 3, 2, 0, 1, 1]，截断后为 [2, 0, 3, 2, 1, 3, 0, 3, 2, 0]，这样 Document 0,2,3 都出现了 3 次，而 Document 1 只出现了 1 次。
- 如果最后一个 Epoch 独立 Shuffle，比如 [2, 0, 3, 3, 1, 0 ,2, 1, 0, 3, 2, 1]，此时截断后为 [2, 0, 3, 3, 1, 0 ,2, 1, 0, 3]，可以保证采样次数最多差 1。
- 目前还不确定为什么不每个 Epoch 都独立 shuffle 拼接后再截断。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgw4vryJ18uicrImtQ0R24neCR7ic0iblapqFw7oE6kCtsmPib5fokG8b2Xrs3xYoXpHicnJYH0EwD7Hfg/640?wx_fmt=png&from=appmsg&randomid=7xqpvo54)

实际上获得的 doc_idx 并没有经过截断，还是完整的 Epoch，因此在 shuffle_idx（也就是 shuffle Samples）时也需要特殊处理最后一个 Epoch：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgw4vryJ18uicrImtQ0R24neKTKO4Nv0T7otarmECicX9BtU0SpEGLB7oL6FpWOhBRI6kicN63KJ1GAw/640?wx_fmt=png&from=appmsg&randomid=bl76dhmi)

### 4.3 GPT Dataset Blending

LLM 在训练数据集中往往包含多个子数据集，实际训练的时候每个数据集会使用不同的权重。比如说有如下三个数据集（数据集在之前的步骤中已经 shuffle）：

- A：100 Samples，采样权重 0.3
- B：50 Samples，采样权重 0.2
- C：400 Samples，采样权重 0.5

假设需要训练 1000 个 Samples，则相当于：

- A：使用 300 个 Samples，也就是数据过 3 轮
- B：使用 200 个 Samples，也就是数据过 4 轮
- C：使用 500 个 Samples，也就是数据过 1.25 轮

构建混合数据集 index 的目标是构建训练需要的 1000 个 Samples 的索引，需要知道：

- Sample 来自哪个 Dataset：dataset_index
- 对应 Dataset 的第几个 Sample：dataset_sample_index

对应计算过程如下：

- 遍历每个 Sample 位置
- 根据当前位置之前已经采样过的 Sample 计算对应 Dataset 的权重，比如 A 为 0.34（+0.04），B 为 0.18（-0.02），C 为 0.45（-0.05），表明已采样的 Sample 中 Dataset C 占比与设定目标相差更多，因此此处从 Dataset C 中采样
- C 中样本计数 +1（也就是如果最后把 dataset_sample_index 中来自 Dataset C 的数据拿出来，则按顺序应该是 [0, 1, 2, …, 498, 499]）
- 更新 dataset_index 和 dataset_sample_index

需要说明的是，上述的 Sample 可以理解为拼接好的满足 max_seq 的序列，来自同一个 Dataset，但可能来自不同的句子。

具体代码位于 https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/main/megatron/data/helpers.cpp#L36-L97，如下所示：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgw4vryJ18uicrImtQ0R24neiboS1DiasmIsMT41t4LHKdbcFQuXfbWibTDzXdYlAA2R9sU9PSr1oQrZg/640?wx_fmt=png&from=appmsg&randomid=8zw5b6mu)

此外，原始 Megatron-DeepSpeed 中的 dataset_index 存储的是 uint8 类型，也就是最多只能有 256 个 Dataset，但是实际上当前 LLM 预训练的 Dataset 可能很多，比如有 1000 多个，此时有两种方案：

1. 修改代码，dataset_index 存储改为 uint16 或 uint32。
2. 将 Dataset 合并到 256 个以内，但是需要保证合并的 Dataset 的 Weight 相同，并且在 shuffle 前合并，不然不等价。

### 4.4 数据加载

如下图所示，BlendableDataset 为实际 LLM 预训练使用的 Dataset，其在初始化阶段完成索引构建（可以 Cache），训练中直接遍历相应的 Sample 即可（返回数据包含子数据集索引及在子数据集中的位置索引）：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/zhVlwj96tTgw4vryJ18uicrImtQ0R24neUKBRjIbcyA1J7oqhavZUTutOM2xsXSibsVD6nZqTummL0zj802bPQUQ/640?wx_fmt=png&from=appmsg&randomid=shfv1nwv)

## 五、参考链接

1. https://commoncrawl.org/overview
2. https://data.commoncrawl.org/crawl-data/CC-MAIN-2023-50/index.html
3. https://arxiv.org/abs/2303.18223
4. https://www.high-flyer.cn/en/blog/cc_cleaner/
5. https://arxiv.org/abs/2309.10305
6. https://huggingface.co/datasets/Skywork/SkyPile-150B
7. http://arxiv.org/abs/2310.19341
8. https://github.com/NVIDIA/Megatron-LM
9. https://github.com/microsoft/Megatron-DeepSpeed
10. https://lifearchitect.ai/whats-in-my-ai/

