# vllm中的attenton计算

**Author:** running

**Date:** 2025-06-26

**Link:** https://zhuanlan.zhihu.com/p/1921944574869350147

[attention计算](https://zhida.zhihu.com/search?content_id=259628175&content_type=Article&match_order=1&q=attention%E8%AE%A1%E7%AE%97&zhida_source=entity)部分

核心的实现在[vllm](https://zhida.zhihu.com/search?content_id=259628175&content_type=Article&match_order=1&q=vllm&zhida_source=entity)\\v1\\attention\\backends\\flash\_attn.py中

此方法会在prefill阶段和decoder阶段进行复用。

![](images/v2-324e4f5ad3c1d4e1c23d6afcae46aec3_1440w_f9cc16069bf5.jpg)

标注：

```python
flash_attn_varlen_func(
    q=query[:num_actual_tokens],        # 所有请求的所有token concatenated
    k=key_cache,                        # 更新历史KV cache
    v=value_cache,
    out=output[:num_actual_tokens],
    cu_seqlens_q=cu_seqlens_q,         # 关键！告诉FlashAttention每个请求的边界
    max_seqlen_q=max_seqlen_q,
    seqused_k=seqused_k,               # 关键！每个请求要使用的历史KV长度
    max_seqlen_k=max_seqlen_k,
    causal=True,                       # 因果掩码
    # ...
)

# query tensor: [9, num_heads, head_size]
# 索引:  0     1      2     3    4   5    6     7      8
query = ["Hello","world","how","are","I","am","fine","Thank","you"]

# cu_seqlens_q（累积序列长度）:
cu_seqlens_q = [0, 4, 7, 9]  # 每个请求的结束位置
#               ^  ^  ^  ^
#               |  |  |  +-- 请求C结束
#               |  |  +----- 请求B结束  
#               |  +-------- 请求A结束
#               +----------- 开始位置

# seqused_k（每个请求在KV cache中的有效长度）:
seqused_k = [4, 3, 2]  # 请求A有4个token，请求B有3个，请求C有2个
```

大概的数据流：

```python
# === 输入数据 ===
# 3个请求的token
requests = {
    "req_A": ["Hello", "world", "how"],      # 3个token，prefill
    "req_B": ["I", "am"],                    # 2个token，prefill  
    "req_C": ["fine"]                        # 1个token，decode
}

# === 拼接后的数据 ===
# 所有token按顺序拼接
concatenated_tokens = ["Hello", "world", "how", "I", "am", "fine"]
query_tensor = tensor([6, num_heads, head_size])  # 6个token

# === 元数据 ===
cu_seqlens_q = [0, 3, 5, 6]  # 累积长度：[开始, req_A结束, req_B结束, req_C结束]
seqused_k = [3, 2, 1]        # 每个请求的KV cache长度

# === Slot Mapping ===
# 每个token在KV cache中的物理位置
slot_mapping = [
    0,    # "Hello" -> block_0, offset_0
    1,    # "world" -> block_0, offset_1
    2,    # "how"   -> block_0, offset_2
    48,   # "I"     -> block_3, offset_0  (3*16 + 0)
    49,   # "am"    -> block_3, offset_1
    80,   # "fine"  -> block_5, offset_0  (5*16 + 0)
]

# === Block Table ===
block_table = [
    [0, -1, -1],   # 请求A使用block 0
    [3, -1, -1],   # 请求B使用block 3
    [5, -1, -1],   # 请求C使用block 5
]
```

attention计算时可能的伪代码：

```python
# FlashAttention内部逻辑（简化版）
for req_idx in range(num_requests):
    # 获取当前请求的query（只有新token）
    q_start = cu_seqlens_q[req_idx]
    q_end = cu_seqlens_q[req_idx + 1]  
    current_q = query[q_start:q_end]     # 当前新token的query
    
    # 获取该请求的所有历史KV（包括刚写入的新KV）
    kv_len = seqused_k[req_idx]          # 要使用的总KV长度
    req_blocks = block_table[req_idx]    # 该请求使用的block
    
    # 从KV cache中取出该请求的所有历史KV
    all_keys = key_cache[req_blocks][:kv_len]      # 所有历史key + 新key
    all_values = value_cache[req_blocks][:kv_len]  # 所有历史value + 新value
    
    # 计算attention：新query attend to 所有历史KV
    attention_output[q_start:q_end] = compute_attention(
        current_q,        # 新token的query
        all_keys,         # 该请求的所有历史key（包括新写入的）
        all_values        # 该请求的所有历史value（包括新写入的）
    )
```