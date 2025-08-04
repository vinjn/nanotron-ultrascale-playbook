# 一张图理解DeepseekMoE

**Author:** 想飞的石头

**Date:** 2025-03-16

**Link:** https://zhuanlan.zhihu.com/p/30599337461

一张图解释[Deepseek混合专家](https://zhida.zhihu.com/search?content_id=255142350&content_type=Article&match_order=1&q=Deepseek%E6%B7%B7%E5%90%88%E4%B8%93%E5%AE%B6&zhida_source=entity)（[Mixture of Experts](https://zhida.zhihu.com/search?content_id=255142350&content_type=Article&match_order=1&q=Mixture+of+Experts&zhida_source=entity),MOE）模型的架构。

1\. 输入和输出：

• \`h'\_t\`是第\`t\`个token的输出。

• \`u\_t\`是第\`t\`个token的输入。

2\. 共享专家（Shared Experts）：

• 有\`Ns\`个共享专家，这些专家对每个token都会进行处理

• 公式：\`h'\_t = u\_t + ∑\_{i=1}^{Ns} FFN\_i^{(s)}\`，其中\`FFN\_i^{(s)}\`是第\`i\`个共享专家的[前馈神经网络](https://zhida.zhihu.com/search?content_id=255142350&content_type=Article&match_order=1&q=%E5%89%8D%E9%A6%88%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C&zhida_source=entity)。

3\. 常规专家（Regular Experts）：

• 有\`Nr\`个常规专家，这些专家会根据[路由分数](https://zhida.zhihu.com/search?content_id=255142350&content_type=Article&match_order=1&q=%E8%B7%AF%E7%94%B1%E5%88%86%E6%95%B0&zhida_source=entity)动态选择。

• 公式：\`h'\_t = u\_t + ∑\_{i=1}^{Nr} g\_{i,t} FFN\_i^{(r)}(u\_t)\`，其中\`g\_{i,t}\`是第\`i\`个专家的[选择概率](https://zhida.zhihu.com/search?content_id=255142350&content_type=Article&match_order=1&q=%E9%80%89%E6%8B%A9%E6%A6%82%E7%8E%87&zhida_source=entity)，\`FFN\_i^{(r)}\`是第\`i\`个常规专家的前馈神经网络。

4\. 选择概率（Selection Probability）：

• \`g'\_{i,t}\`是第\`i\`个专家的原始选择分数。

• 公式：\`g'\_{i,t} = g'\_{i,t} / ∑\_{j=1}^{Nr} g'\_{j,t}\`，这是在归一化之前的原始选择分数。

• 确保概率之和为1，以形成有效的概率分布。

  

5\. 选择专家（Selecting Experts）：

• \`g'\_{i,t}\`根据分数选择前\`Kr\`个专家。

• 公式：\`g'\_{i,t} = { Si\_{i,t}, Si\_{i,t} ∈ Topk(S\_{i,t} | 1 ≤ j ≤ Nr), Kr }\`，其中\`Si\_{i,t}\`是第\`i\`个专家的选择指示。

• 如果\`Si\_{i,t}\`不在前\`Kr\`个专家中，则\`g'\_{i,t} = 0\`，否则\`g'\_{i,t} = Si\_{i,t}\`。

6\. 计算选择指示（Computing Selection Indicator）：

• \`S\_{i,t} = Sigmoid(u\_t^T e\_i)\`，其中\`u\_t\`是token的表示，\`e\_i\`是第\`i\`个专家的中心向量。

• 点积\`u\_t^T e\_i\`用于计算亲和分数，确定token与专家的对齐程度。

7\. 亲和分数（Affinity Score）：

• \`Sigmoid\`函数确保分数在\[0,1\]范围内。

• 亲和分数用于确定token与专家的对齐程度。

  

![](images/v2-aa0fa9c5e9c745c6b2fd0e91fe200219_1440w_e4f9e9aa7300.jpg)