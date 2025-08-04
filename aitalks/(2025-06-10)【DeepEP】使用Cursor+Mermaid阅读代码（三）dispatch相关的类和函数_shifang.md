# 【DeepEP】使用Cursor+Mermaid阅读代码（三）dispatch相关的类和函数

**Author:** shifang

**Date:** 2025-06-10

**Link:** https://zhuanlan.zhihu.com/p/1915799730899886719

​

目录

收起

提示词

生成的UML

svg图

这边文章重点关注dispatch相关的类和函数。

## 提示词

```text
分析这几个文件中的关键类和函数，并画出UML类图
DeepEP/deep_ep/buffer.py
DeepEP/csrc/deep_ep.cpp
DeepEP/csrc/kernels/intranode.cu
DeepEP/csrc/kernels/internode.cu
DeepEP/csrc/kernels/internode_ll.cu
DeepEP/csrc/kernels/buffer.cuh

层级关系是:
buffer.py 中的类或函数调用 deep_ep.cpp
deep_ep.cpp 中的类或函数调用 intranode.cu, internode.cu,internode_ll.cu 中的类或函数.
intranode.cu, internode.cu,internode_ll.cu 中的类或函数 调用 buffer.cuh 中的类或者函数.

关键类或者函数是：
buffer.py 中的关键类是 Buffer, buffer.py 中的关键函数是 Buffer.dispatch
deep_ep.cpp 中的关键函数是 Buffer::intranode_dispatch, Buffer::internode_dispatch, Buffer::low_latency_dispatch
intranode.cu 中的关键函数是 intranode::notify_dispatch, intranode::dispatch
internode.cu 中的关键函数是 internode::cached_notify, internode::dispatch
internode_ll.cu 中的关键函数是 internode_ll::dispatch
buffer.cuh 中的关键类是 Buffer, SymBuffer, AsymBuffer.
Buffer 用于 intranode.cu 中的 rdma_channel 通讯
SymBuffer 用于 internode.cu 中的 rdma_channel 通讯
AsymBuffer 用于 internode.cu 中的 nvl_channel 通讯
```

## 生成的UML

Cursor自动生成之后，又进行了一些手动编辑。

```text
classDiagram
    class Buffer_py {
        +dispatch()
    }
    
    class Buffer_cpp {
        +Intranode_dispatch()
        +Internode_dispatch()
        +low_latency_dispatch()
    }
    
    class Intranode.cu {
        +notify_dispatch()
        +dispatch()
    }
    
    class Internode.cu {
        +cached_notify()
        +dispatch()
    }
    
    class Internode_ll.cu {
        +dispatch()
    }
    
    class Buffer_cuh {
        +attributes
        +methods
    }
    
    class SymBuffer_cuh {
        +attributes
        +methods
    }
    
    class ASymBuffer_cuh {
        +attributes
        +methods
    }
    
    Buffer_py --> Buffer_cpp : calls
    Buffer_cpp --> Intranode.cu : calls
    Buffer_cpp --> Internode.cu : calls
    Buffer_cpp --> Internode_ll.cu : calls
    Intranode.cu --> Buffer_cuh : uses for rdma_channel
    Internode.cu --> SymBuffer_cuh : uses for rdma_channel
    Internode.cu --> ASymBuffer_cuh : uses for nvl_channel
    
    note for Buffer_py "Python interface in buffer.py"
    note for Buffer_cpp "C++ implementation in deep_ep.cpp"
    

```

### svg图

![](https://pic4.zhimg.com/v2-99eb88f1fb199753d47ee83b5b1a22b5_1440w.jpg)