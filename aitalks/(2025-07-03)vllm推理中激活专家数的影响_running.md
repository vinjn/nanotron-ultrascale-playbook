# vllm推理中激活专家数的影响

**Author:** running

**Date:** 2025-07-03

**Link:** https://zhuanlan.zhihu.com/p/1923705431261414887

这里有个核心方法：invoke\_fused\_moe\_kernel（），其负责执行专家网络的矩阵乘法运算，在[vllm](https://zhida.zhihu.com/search?content_id=259812722&content_type=Article&match_order=1&q=vllm&zhida_source=entity)\\model\_executor\\layers\\fused\_moe\\fused\_moe.py文件中。（注意，下文中提到的块儿，chunk，block等概念不同于sheduler，这里的概念适用于MOE做计算。）

  

在进行MOE计算时，对需要计算的数据进行分块儿，因为矩阵的计算都是按照固定的block块大小进行的。（感觉，之所有MOE模型在计算时有这一步的处理，是因为在计算attention时虽然也有但是在[flash attention](https://zhida.zhihu.com/search?content_id=259812722&content_type=Article&match_order=1&q=flash+attention&zhida_source=entity)或flash infer中实现了，但是那个不能直接用于MOE）。

```python
@triton.jit
def fused_moe_kernel(
    # ...
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # ...
):
    # Map program ids to blocks
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)  # 需要整除
```

除了分块以外，要给当前的计算分配GPU资源，这里应该是使用trition实现的：这个grid决定了GPU上启动多少个并行线程块。

```python
# 计算网格大小
EM = sorted_token_ids.shape[0]
if A.shape[0] < config["BLOCK_SIZE_M"]:
    EM = min(sorted_token_ids.shape[0], A.shape[0] * top_k * config['BLOCK_SIZE_M'])

# grid 是一个 lambda 函数，接受 META 参数并返回网格维度
grid = lambda META: (
    triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(B.shape[1], META['BLOCK_SIZE_N']),
)

```

EM = M \* top\_k，网格大小与 top\_k 成正比，增加 top\_k 直接增加并行块数量，提升并行效率。

![](https://pic4.zhimg.com/v2-87586d78dfab23a8c197b7ecb6828731_1440w.jpg)

trition修饰器用于读取trition的配置

![](https://pic3.zhimg.com/v2-d15af1d52e61399dfce0b44c9d4b8b86_1440w.jpg)

加入\[grid\]执行trition的配置

这里使用了trition的修饰器。GPU上并行的网格数量应该是：

num\_blocks\_m = triton.cdiv(EM, BLOCK\_SIZE\_M) # M 维度的块数

num\_blocks\_n = triton.cdiv(B.shape\[1\], BLOCK\_SIZE\_N) # N 维度的块数

total\_blocks = num\_blocks\_m \* num\_blocks\_n # 总块数

在vllm给定了限制，毕竟GPU也是资源有限，限定的方式是对chunk块儿的大小进行限制：

```python
def fused_experts_impl(...):
    num_tokens, _ = hidden_states.shape
    
    # 关键：分块大小限制
    CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE  # 默认 32768
    M = min(num_tokens, CHUNK_SIZE)
    
    # 按块处理，避免网格过大
    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens)
        )
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        
        # 为当前块调用 kernel
        invoke_fused_moe_kernel(...)
```

对于每个块儿来说，网格大小是：

```python
# M 被限制在 CHUNK_SIZE (32768) 以内
M = min(num_tokens, CHUNK_SIZE)  # 最大 32768
EM = sorted_token_ids.shape[0]   # ≈ M * top_k

# 网格大小
grid_size = triton.cdiv(EM, BLOCK_SIZE_M) * triton.cdiv(B.shape[1], BLOCK_SIZE_N)
```

并行度提升的代码：

```python
# 第331-339行：GPU线程块的分配逻辑
@triton.jit
def fused_moe_kernel(...):
    # 获取当前线程块ID
    pid = tl.program_id(axis=0)  # 0 到 6143
    
    # 计算2D坐标
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)      # 96
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)       # 64
    
    # 分组处理以提高L2缓存利用率
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # 根据pid_m找到对应的专家
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)



# 第355行：专家选择逻辑
off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # 专家不在当前rank，写入零
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N,
                              offs_token, token_mask, BLOCK_SIZE_M,
                              BLOCK_SIZE_N, compute_type)
        return
```

-   **激活专家数=1**: 所有线程块都访问同一个专家的权重
-   **激活专家数=6**: 线程块分散访问6个专家的权重，GPU计算单元分散处理6个专家，并行度更高

  

  

  

  

  

优化点：

给每个专家分配到的token数量都是block\_size的倍数，为每个专家预留最多(block\_size - 1)个填充位置

在vllm\\model\_executor\\layers\\fused\_moe\\moe\_align\_block\_size.py中，有一个对齐操作

```python
def moe_align_block_size(
    topk_ids: torch.Tensor,  # shape: [M, top_k]
    block_size: int,
    num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
    pad_sorted_ids: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

"""
Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
"""
```

-   **将 token 按专家重新排序**：topk\_ids 被展开为 `[M * top_k]` 的一维张量
-   **填充到块大小的倍数**：确保每个专家处理的 token 数量是 block\_size 的倍数(这个block\_size跟scheduler是两个东西，这个是用来配置矩阵运算的块大小的vllm\\model\_executor\\layers\\fused\_moe\\fused\_moe.py->def try\_get\_optimal\_moe\_config()方法)
-   **返回排序后的 token IDs**：sorted\_token\_ids 的长度变为 M \* top\_k + padding。

这个padding其实是一种空间浪费，能不能组合？