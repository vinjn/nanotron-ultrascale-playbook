# 大模型分布式训中all_gather、reduce_scatter、all_reduce使用总结

**Author:** xhchen

**Date:** 2024-07-01

**Link:** https://zhuanlan.zhihu.com/p/706341870

​

目录

收起

常用原语

DP

TP

ColumnLinear

RowLinear

Zero3

forward

backward

## 常用原语

[all\_gather](https://zhida.zhihu.com/search?content_id=245095857&content_type=Article&match_order=1&q=all_gather&zhida_source=entity)：收集所有数据块并分发到所有rank上。

[reduce\_scatter](https://zhida.zhihu.com/search?content_id=245095857&content_type=Article&match_order=1&q=reduce_scatter&zhida_source=entity)：reduce将每个rank的数据块与其他rank的数据块进行归约操作，scatter再将数据块分发到指定rank。

[all\_reduce](https://zhida.zhihu.com/search?content_id=245095857&content_type=Article&match_order=1&q=all_reduce&zhida_source=entity)：reduce\_scatter + all\_gather

## DP

现在[DDP](https://zhida.zhihu.com/search?content_id=245095857&content_type=Article&match_order=1&q=DDP&zhida_source=entity)是比较通用的数据并行方式，它利用了[ring-all\_reduce](https://zhida.zhihu.com/search?content_id=245095857&content_type=Article&match_order=1&q=ring-all_reduce&zhida_source=entity)做梯度同步。

ring-all\_reduce主要包括reduce\_scatter + all\_gather两个部分。

1、reduce\_scatter：相邻的两个rank(a,b)利用reduce进行梯度累加(a+b)，再利用scatter将结果发送给指定的rank(b)。

2、all\_gather：相邻的两个rank(a,b)利用all\_gather进行梯度同步(a=b)，最后所有rank都会得到相同的并且完整的梯度。

## TP

主要针对[Megatron](https://zhida.zhihu.com/search?content_id=245095857&content_type=Article&match_order=1&q=Megatron&zhida_source=entity)中先[ColumnLinear](https://zhida.zhihu.com/search?content_id=245095857&content_type=Article&match_order=1&q=ColumnLinear&zhida_source=entity)再[RowLinear](https://zhida.zhihu.com/search?content_id=245095857&content_type=Article&match_order=1&q=RowLinear&zhida_source=entity)的切分方式。

### ColumnLinear

1、all\_reduce：在backward的时候通过all\_reduce同步梯度；forward的时候不需要进行通信。

### RowLinear

1、all\_reduce：在forward的时候通过all\_reduce同步计算结果；backward的时候不需要进行通信。

## [Zero3](https://zhida.zhihu.com/search?content_id=245095857&content_type=Article&match_order=1&q=Zero3&zhida_source=entity)

### forward

1、all\_gather：通过all\_gather收集所有rank上的模型参数切片，为了聚合参数，以数据并行的方式进行前向传播。

### backward

1、all\_gather：通过all\_gather收集所有rank上的模型参数切片。

2、reduce\_scatter：通过reduce\_scatter同步不同的rank之间的梯度。（和DDP一样）