# vllm中deepseek模型的运行流程

**Author:** running

**Date:** 2025-07-02

**Link:** https://zhuanlan.zhihu.com/p/1924041534766056006

1.推理时的整体入口,先初始化worker和modelrunner，执行时通过execute\_model（）执行

```python
class Worker(LocalOrDistributedWorkerBase):
    def __init__(self, vllm_config: VllmConfig, ...):
        # 根据模型类型选择 ModelRunner
        ModelRunnerClass: Type[GPUModelRunnerBase] = ModelRunner
        if model_config.runner_type == "pooling":
            ModelRunnerClass = PoolingModelRunner
        elif self.model_config.is_encoder_decoder:
            ModelRunnerClass = EncoderDecoderModelRunner
        self.model_runner: GPUModelRunnerBase = ModelRunnerClass(...)

# vllm/worker/worker_base.py - execute_model()
def execute_model(self, execute_model_req: Optional[ExecuteModelRequest] = None):
    output = self.model_runner.execute_model(
        model_input=model_input,
        kv_caches=self.kv_cache[worker_input.virtual_engine],
        intermediate_tensors=intermediate_tensors,
        num_steps=num_steps,
        **kwargs,
    )
#vllm\v1\worker\gpu_model_runner.py 中也会进行类似的操作，来执行推理，model_runner是GPU上实际运行的代码
```

2.对于 [DeepSeek-R1](https://zhida.zhihu.com/search?content_id=259851865&content_type=Article&match_order=1&q=DeepSeek-R1&zhida_source=entity)模型的加载，主要涉及vllm\\model\_executor\\models\\deepseek\_v2.py

```python
def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        self.model = DeepseekV2Model(vllm_config=vllm_config, ...)
        
class DeepseekV2Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # 创建各个 Decoder Layer
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV2DecoderLayer(...),
            prefix=f"{prefix}.layers")
##MOE层的构建
class DeepseekV2DecoderLayer(nn.Module):
    def __init__(self, config, ...):
        # 根据层索引决定是否使用 MOE
        if (config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0):
            self.mlp = DeepseekV2MoE(config=config, ...)
        else:
            self.mlp = DeepseekV2MLP(...)
# MOE模型组件（类）
class DeepseekV2MoE(nn.Module):
    # 在 deepseek_v2.py 第103-147行
   def __init__(self, config, quant_config, prefix):
       # 1. 创建门控网络
       self.gate = ReplicatedLinear(config.hidden_size, config.n_routed_experts, bias=False)
    
       # 2. 创建专家并行的 FusedMoE
       self.experts = FusedMoE(
           num_experts=config.n_routed_experts,
           top_k=config.num_experts_per_tok,
           hidden_size=config.hidden_size,
           intermediate_size=config.moe_intermediate_size,
           use_grouped_topk=True,  # DeepSeek 使用分组 top-k
           num_expert_group=config.n_group,
           topk_group=config.topk_group,
           scoring_func=config.scoring_func,
           e_score_correction_bias=self.gate.e_score_correction_bias
       )
    
       # 3. 可选的共享专家
       if config.n_shared_experts is not None:
           self.shared_experts = DeepseekV2MLP(...)
```

FusedMoe的初始化

```python
# 在 layer.py 第384-491行
def __init__(self, num_experts, top_k, hidden_size, intermediate_size, ...):
    # 1. 设置并行配置（TP/EP/DP）
    self.tp_size = get_tensor_model_parallel_world_size()
    self.global_num_experts = num_experts
    self.local_num_experts = num_experts  # 如果没有 EP
    
    # 2. 创建量化方法（无量化情况下使用 UnquantizedFusedMoEMethod）
    self.quant_method = UnquantizedFusedMoEMethod()
    
    # 3. 创建专家权重矩阵
    self.quant_method.create_weights(layer=self, num_experts=self.local_num_experts, ...)

# 在 layer.py 第75-95行
def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype):
    # 创建融合的 gate_up_proj 权重 (w1 + w3)
    w13_weight = torch.nn.Parameter(torch.empty(
        num_experts, 2 * intermediate_size_per_partition, hidden_size, dtype=params_dtype))
    
    # 创建 down_proj 权重 (w2)
    w2_weight = torch.nn.Parameter(torch.empty(
        num_experts, hidden_size, intermediate_size_per_partition, dtype=params_dtype))
```

  

3.模型的前向传播

```python
# DeepseekV2ForCausalLM.forward
def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
    hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
    return hidden_states

# DeepseekV2Model.forward  
def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds=None):
    # Embedding
    hidden_states = self.get_input_embeddings(input_ids)
    
    # 逐层处理
    for layer in self.layers[self.start_layer:self.end_layer]:
        hidden_states, residual = layer(positions, hidden_states, residual)
    
    return hidden_states

#DeepseekV2DecoderLayer.forward 中：
def forward(self, positions, hidden_states, residual):
    # Self Attention
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    
    hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
    
    # MoE/MLP 处理
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)  # 这里调用 MoE
    
    return hidden_states, residual
```

4.主要的Moe前向传播流程

```python
#DeepseekV2MoE.forward 中
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    num_tokens, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    
    # 共享专家计算 (如果有)
    if self.n_shared_experts is not None:
        shared_output = self.shared_experts(hidden_states)
    
    # Router 计算：决定激活哪些专家
    router_logits, _ = self.gate(hidden_states)  # [num_tokens, n_experts]
    
    # 专家计算：核心 MOE 计算
    final_hidden_states = self.experts(
        hidden_states=hidden_states,
        router_logits=router_logits) * self.routed_scaling_factor
    
    # 合并共享专家输出
    if shared_output is not None:
        final_hidden_states = final_hidden_states + shared_output
    
    # 张量并行 all-reduce
    if self.tp_size > 1:
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
    
    return final_hidden_states.view(num_tokens, hidden_dim)
```

Fused\_moe执行：FusedMoE.forward\->FusedMoE.forward\_impl

```python
#vllm\model_executor\layers\fused_moe\layer.py中的FusedMoE.forward 
def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
    if self.use_direct_call:
        return self.forward_impl(hidden_states, router_logits)
    else:
        return torch.ops.vllm.moe_forward(hidden_states, router_logits, self.layer_name)

def forward_impl(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
    # 数据并行处理 (如果有)
    if self.dp_size > 1:
        hidden_states = self.naive_multicast(hidden_states, cu_tokens_across_dp_cpu)
        router_logits = self.naive_multicast(router_logits, cu_tokens_across_dp_cpu)
    
    # 核心 MOE 计算
    final_hidden_states = self.quant_method.apply(  ##如果没有量化的话这里的方法类会是无量化的执行方法
        layer=self,
        x=hidden_states,
        router_logits=router_logits,
        top_k=self.top_k,
        renormalize=self.renormalize,
        use_grouped_topk=self.use_grouped_topk,  # DeepSeek 使用 grouped_topk
        ...)
    
    return final_hidden_states
```

进行专家选择（[UnquantizedFusedMoEMethod.apply](file:///d%3A/project/vllm/vllm/model_executor/layers/fused_moe/layer.py#76%2C7)\->[forward\_cuda](file:///d%3A/project/vllm/vllm/model_executor/layers/fused_moe/layer.py#176%2C9)）：

```python
# 在 layer.py 第188-215行
def forward_cuda(self, layer, x, router_logits, top_k, renormalize, use_grouped_topk, ...):
    # 1. 专家选择
    topk_weights, topk_ids = FusedMoE.select_experts(
        hidden_states=x, router_logits=router_logits,
        use_grouped_topk=use_grouped_topk, top_k=top_k, renormalize=renormalize,
        topk_group=topk_group, num_expert_group=num_expert_group,
        custom_routing_function=custom_routing_function,
        scoring_func=scoring_func, e_score_correction_bias=e_score_correction_bias)
    
    # 2. 融合专家计算
    return fused_experts(
        hidden_states=x, w1=layer.w13_weight, w2=layer.w2_weight,
        topk_weights=topk_weights, topk_ids=topk_ids, inplace=True,
        activation=activation, apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts, expert_map=expert_map)

##专家选择方法：FusedMoE.select_experts
# 在 layer.py 第787-827行
@staticmethod
def select_experts(hidden_states, router_logits, top_k, use_grouped_topk, renormalize, ...):
    # DeepSeek 使用分组 top-k
    if use_grouped_topk:
        topk_weights, topk_ids = grouped_topk(
            hidden_states=hidden_states, gating_output=router_logits,
            topk=top_k, renormalize=renormalize,
            num_expert_group=num_expert_group, topk_group=topk_group,
            scoring_func=scoring_func, e_score_correction_bias=e_score_correction_bias)
    else:
        # 标准 top-k
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states, gating_output=router_logits,
            topk=top_k, renormalize=renormalize)
    
    return topk_weights, topk_ids

##分组Top-K方法grouped_topk
# 在 fused_moe.py 第918-980行
@torch.compile(dynamic=True)
def grouped_topk(hidden_states, gating_output, topk, renormalize, num_expert_group, topk_group, scoring_func, e_score_correction_bias):
    # 1. 计算评分（softmax 或 sigmoid）
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    
    # 2. 应用偏置修正（如果有）
    if e_score_correction_bias is not None:
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
    
    # 3. 分组选择：先选 topk_group 个专家组
    group_scores = scores.view(num_token, num_expert_group, -1).max(dim=-1).values
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    
    # 4. 在选中的组内选择 topk 个专家
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(...).reshape(num_token, -1)
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
    
    # 5. 最终 top-k 选择
    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)[1]
        topk_weights = original_scores.gather(1, topk_ids)  # 使用原始评分作为权重
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    
    # 6. 重新归一化权重
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)
```

5.专家计算

```python
# 在 fused_moe.py 第1297-1362行
def fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, inplace, activation, ...):
    # 调用 dispatch_fused_experts_func 选择合适的实现
    return dispatch_fused_experts_func(inplace)(
        hidden_states=hidden_states, w1=w1, w2=w2,
        topk_weights=topk_weights, topk_ids=topk_ids,
        activation=activation, apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8, use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16, use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant, global_num_experts=global_num_experts,
        expert_map=expert_map, w1_scale=w1_scale, w2_scale=w2_scale,
        w1_zp=w1_zp, w2_zp=w2_zp, a1_scale=a1_scale, a2_scale=a2_scale,
        block_shape=block_shape)
```

实际实现：

```python
# 在 fused_moe.py 第1009-1175行（这是核心计算逻辑）
def fused_experts_impl(hidden_states, w1, w2, topk_weights, topk_ids, inplace, ...):
    # 1. 参数检查和初始化
    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape  # 专家数、输出维度、输入维度
    K = w2.shape[1]     # w2 的输入维度
    top_k_num = topk_ids.shape[1]
    
    # 2. 分块处理大 batch
    CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
    M = min(num_tokens, CHUNK_SIZE)
    
    # 3. 获取配置
    config = get_config_func(M)
    
    # 4. 分配缓存张量
    cache13 = torch.empty(M * top_k_num * max(N, K), device=hidden_states.device, dtype=hidden_states.dtype)
    intermediate_cache1 = cache13[:M * top_k_num * N].view(M, top_k_num, N)
    intermediate_cache2 = torch.empty((M * top_k_num, N // 2), device=hidden_states.device, dtype=hidden_states.dtype)
    intermediate_cache3 = cache13[:M * top_k_num * K].view(M, top_k_num, K)
    
    # 5. 按块处理
    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE, min((chunk + 1) * CHUNK_SIZE, num_tokens))
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        
        # 6. 准备当前块的输入（量化等）
        qcurr_hidden_states, qa1_scale = moe_kernel_prepare_input(...)
        
        # 7. 对齐块大小并排序专家
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            curr_topk_ids, config['BLOCK_SIZE_M'], global_num_experts, expert_map)
        
        # 8. 第一次矩阵乘法：A × W1（gate_up_proj）
        invoke_fused_moe_kernel(
            A=qcurr_hidden_states.view(-1, qcurr_hidden_states.shape[-1]),
            B=w1, C=intermediate_cache1.view(-1, intermediate_cache1.shape[-1]),
            # ... 其他参数
        )
        
        # 9. 激活函数（SiLU）处理
        ops.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))
        
        # 10. 第二次矩阵乘法：activated × W2（down_proj）
        invoke_fused_moe_kernel(
            A=intermediate_cache2, B=w2,
            C=intermediate_cache3.view(-1, intermediate_cache3.shape[-1]),
            # ... 其他参数
        )
        
        # 11. 应用 top-k 权重并聚合结果
        ops.topk_softmax(...)  # 应用路由权重
        
        # 12. 写回输出
        out_hidden_states[begin_chunk_idx:end_chunk_idx] = ...
    
    return out_hidden_states
```

这里有一个核心的kernel调用（`invoke_fused_moe_kernel`）：

```python
# 在 fused_moe.py 第471-626行
def invoke_fused_moe_kernel(A, B, C, A_scale, B_scale, B_zp, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, ...):
    # 1. 计算网格大小
    M = A.shape[0]
    EM = sorted_token_ids.shape[0]
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        # 小 batch 优化
        EM = min(sorted_token_ids.shape[0], A.shape[0] * top_k * config['BLOCK_SIZE_M'])
    
    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(B.shape[1], META['BLOCK_SIZE_N']),)
    
    # 2. 选择 kernel 类型
    if (use_int8_w8a16 or use_int4_w4a16) and block_shape is not None:
        # 使用量化 kernel
        if use_moe_wna16_cuda:
            ops.moe_wna16_gemm(...)  # CUDA kernel
        else:
            fused_moe_kernel_gptq_awq[grid](...)  # Triton kernel
    else:
        # 使用标准 kernel
        fused_moe_kernel[grid](...)
```

Triton Kernel 执行（`fused_moe_kernel`）：

```python
# 在 fused_moe.py 第280-470行
@triton.jit
def fused_moe_kernel(a_ptr, b_ptr, c_ptr, ...):
    # 1. 获取程序 ID 和计算块坐标
    pid = tl.program_id(axis=0)
    # ... 计算 pid_m, pid_n
    
    # 2. 检查专家有效性
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        write_zeros_to_output(...)  # 写零到无效专家的输出
        return
    
    # 3. 计算指针偏移
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # 4. 主循环：K 维度分块矩阵乘法
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 加载 A 和 B 块
        a = tl.load(a_ptrs, mask=..., other=0.0)
        b = tl.load(b_ptrs, mask=..., other=0.0)
        
        # 处理量化（如果需要）
        if use_int8_w8a16:
            accumulator = (accumulator * b_scale).to(compute_type)
        elif use_fp8_w8a8 or use_int8_w8a8:
            accumulator = (accumulator * a_scale * b_scale).to(compute_type)
        
        # 矩阵乘法累积
        accumulator += tl.dot(a, b)
        
        # 前进指针
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # 5. 应用路由权重
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
    
    # 6. 写回结果
    tl.store(c_ptrs, accumulator, mask=c_mask)
```