# 【DeepEP】使用Cursor+Mermaid阅读代码（四）low_latency dispatch kernel流程图

**Author:** shifang

**Date:** 2025-06-28

**Link:** https://zhuanlan.zhihu.com/p/1922625463332877523

## 使用[Cursor](https://zhida.zhihu.com/search?content_id=259688709&content_type=Article&match_order=1&q=Cursor&zhida_source=entity)生成用uml描述的流程图

```python3
@startuml
skinparam titleFontSize 30
skinparam titleFontStyle bold
title \n low_latency dispatch kernel流程图 \n

participant "Host" as H
participant "GPU Kernel" as K
participant "Warp Groups" as WG
participant "RDMA Buffer" as RB
participant "Shared Memory" as SM
participant "Remote Node" as RN
participant "Packed Buffer" as PB

== 初始化阶段 ==
H -> K: 启动dispatch kernel
K -> K: 初始化线程/块信息
K -> SM: 分配共享内存

note over K, RN: low_latency 模式没有 notify_dispatch 的过程，即不会先进行一次通信来确定 GPU 之间互相发送 token 的数量。\nHost function 会为每个 expert 提前准备能容纳 num_max_dispatch_tokens_per_rank * num_ranks 个 token 的 buffer。\n因此 low_latency 模式的内存开销是比较高的。
== 发送阶段 (LOW_LATENCY_SEND_PHASE) ==
group 主要工作线程 (warp_id < num_warps - 1)
    loop 对每个token
        K -> K: 数据格式转换(FP8/BF16)，并保存到send buffer      
        
        alt 有目标专家 (dst_expert_idx >= 0)
            K -> K: 计算目标rank和本地专家索引
            K -> RB: 准备发送缓冲区
            
            alt P2P不可用 (dst_p2p_ptr == 0)
                K -> RN: IBGDA异步发送
                note over K, RN: 调用IBGDA进行发送到指定的slot, 这样可以实现在AdaptiveRouting开启时发送不用保序。
            else 
                K -> RN: 直接P2P发送
            end
            K -> K: 增加完成计数器
        end
    end
end

group 统计线程 (warp_id == num_warps - 1)   
    K -> K: 统计每个专家的token数量
    K -> WG: warp内reduce统计结果
    K -> SM: 存储专家统计信息
end

group 发送专家计数 (sub_warp_id == 0 and lane_id == 0)
    K -> RN: 发送专家token计数
end

== 接收阶段 (LOW_LATENCY_RECV_PHASE) ==
group 接收处理
    K -> K: 设置接收缓冲区
    
    alt sub_warp_id == 1
        loop 等待token到达
            K -> RN: 检查rdma_recv_count
        end
        K -> PB: 更新packed_recv_count
    end
        
    loop 对每个接收的token
        K -> PB: 复制 source info 和 data
        
        alt 使用FP8
            K -> PB: 复制FP8 scales
        end
    end
end

== 完成 ==
K -> H: 返回结果

note over K, RN
DeepEP使用低延迟通信模式
- 支持FP8和BF16格式
- 使用IBGDA和P2P通信
- 异步发送和接收
end note
@enduml
```

  

## 使用在线工具绘制流程图

![](https://pic4.zhimg.com/v2-a41a22d1d212fc94f32dd1c7327e8fdd_1440w.jpg)