# pytorch 源码分析（4）softmax （1）

**Author:** 阿嚏

**Date:** 2025-06-16

**Link:** https://zhuanlan.zhihu.com/p/1917970073152385299

前文中，[pytorch](https://zhida.zhihu.com/search?content_id=259150717&content_type=Article&match_order=1&q=pytorch&zhida_source=entity)都是处理element-wise问题的，相对比较简单，但不是所有的问题都是element-wise问题,例如softmax,这种算子很难放到一个统一的框架，只能有自己的代码特殊处理。

一般大的工程，总是有一些生成代码的机制，看代码时不是很方便，这里涉及到了pytorch的[dispatch机制](https://zhida.zhihu.com/search?content_id=259150717&content_type=Article&match_order=1&q=dispatch%E6%9C%BA%E5%88%B6&zhida_source=entity)，有机会可以再深入分析一下

```text
- func: _softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  dispatch:
    CPU: softmax_cpu_out
    CUDA: softmax_cuda_out
    MPS: softmax_mps_out
```

softmax\_cuda\_out 主要就是调用host\_softmax,这个函数非常长，我们分段看下

```text
template<template<typename, typename, typename> class Epilogue, bool is_log_softmax>
Tensor host_softmax(const Tensor & input_, const int64_t dim_, const bool half_to_float, const Tensor& output){
  if (half_to_float) {
    TORCH_CHECK(input_.scalar_type() == ScalarType::Half, "conversion is supported for Half type only");
  }
  auto input = input_.contiguous();
  static_assert(std::is_same<acc_type<at::Half, true>, float>::value, "accscalar_t for half should be float");
  if (input.dim() == 0) input = input.view(1);
  int64_t dim = maybe_wrap_dim(dim_, input.dim());
  TORCH_CHECK(dim >=0 && dim < input.dim(), "dim must be non-negative and less than input dimensions");
  int64_t outer_size = 1;
  int64_t dim_size = input.size(dim);

  if (input.numel() > 0) {
    int64_t inner_size = 1;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    for (int64_t i = 0; i < dim; ++i)
      outer_size *= input.size(i);
    for (int64_t i = dim + 1; i < input.dim(); ++i)
      inner_size *= input.size(i);
    // This kernel spawns a block per each element in the batch.
    // XXX: it assumes that inner_size == 1
```

首先有三个变量，inner\_size dim\_size outer\_size ，注意在softmax中使用时，是有一个dim概念在的，这个很抽象，因为大家生活在三维世界，而脑子能理解二维也已经很不错了，用脑子理解一个四维的tensor，甚至更高维的tensor，那肯定有点困难。

先来一个三维tensor，shape(2,3,4)

如果dim=1, 那么dim\_size=3 outer\_size=2 inner\_size=4

这里的outer\_size意义有点像batch\_size,每个batch的计算是相对独立的。

而inner\_size是内部softmax的重复次数，这个后面再看。

先看一种简单的情况，inner\_size = 1，这种情况很常见，毕竟这世界上最火的模型就是[attention计算](https://zhida.zhihu.com/search?content_id=259150717&content_type=Article&match_order=1&q=attention%E8%AE%A1%E7%AE%97&zhida_source=entity)。

```text
    if (inner_size == 1) {
      dim3 grid(outer_size);
      AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "host_softmax", [&] {
        using accscalar_t = acc_type<scalar_t, true>;
        if (!half_to_float) {
          auto output_ptr = output.mutable_data_ptr<scalar_t>();
          auto input_ptr = input.const_data_ptr<scalar_t>();
          if (dim_size <= 1024 && dim_size*sizeof(scalar_t) <= 4096) {
            int64_t remaining = outer_size;
            int64_t chunk_size = (1L << 30L) / dim_size;
            while(remaining > 0) {
              dispatch_softmax_forward<scalar_t, scalar_t, accscalar_t, is_log_softmax, false>(
                output_ptr, input_ptr, dim_size, dim_size, std::min<int64_t>(remaining, chunk_size), nullptr/* not masked */);
              input_ptr += chunk_size * dim_size;
              output_ptr += chunk_size * dim_size;
              remaining -= chunk_size;
            }
          }
```

这里有一些经验值，当dim\_size 比较小时，走到这里，注意这里还有一个chunk\_size，防止outer\_size过大，计算不过来。

下面我们看一下dispatch\_softmax\_forward

首先需要算出一个第一个大于softmax\_elements（dim\_size）的2的整次方，这时我们发现深度学习中参数总是2的整次方不仅是玄学，也是为了计算方便，要不然没准哪里就有雷。。

计算block和thread的步骤仍然逻辑比较乱，首先关注batch\_count，之前也说过，outer\_size类似batch size，这里命名都一致了，只不过其实这更该看作chunk\_size。那么如果总共处理batch\_count个数据，需要多少个block？自然是需要计算出每一个block可以计算多少数据，也就是batches\_per\_block，最后blocks = (batch\_count + batches\_per\_block - 1) / batches\_per\_block;

threads\_per\_block 是一个固定值128，这大概是因为目前架构的gpu一个sm中有128个cuda core。

在线程维度，被划分成x y轴，一个是[warp\_size](https://zhida.zhihu.com/search?content_id=259150717&content_type=Article&match_order=1&q=warp_size&zhida_source=entity) 一个是warps\_per\_block = (threads\_per\_block / warp\_size)，总共还是128线程。

warp\_size这块有点怪怪的，不一定是固定值32，也可能更小 warp\_size = (next\_power\_of\_two < warp\_size) ? next\_power\_of\_two : warp\_size;

如果dim\_size比较小，也就是next\_power\_of\_two <= 128，可以考虑让一个warp处理2个batch，毕竟活太少也不能让warp闲着。

计算好这些信息，就可以调用softmax\_warp\_forward了，这里这么奇怪的调用，是为了让模板编译时特化，省的一上来就全特化了。

```text
template<typename input_t, typename output_t, typename acc_t, bool is_log_softmax, bool is_masked>
void dispatch_softmax_forward(output_t *dst, const input_t *src, int softmax_elements, int softmax_elements_stride, int batch_count, const bool *mask = nullptr, int chunk_size = -1, bool is_transformer_mask = false)
{
    TORCH_INTERNAL_ASSERT( softmax_elements >= 0 && softmax_elements <= 1024 );
    if (softmax_elements == 0) {
        return;
    } else {
        int log2_elements = log2_ceil(softmax_elements);
        const int next_power_of_two = 1 << log2_elements;

        // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_forward.
        int warp_size = at::cuda::warp_size();
        warp_size = (next_power_of_two < warp_size) ? next_power_of_two : warp_size;

        // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_forward.
        int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

        // use 128 threads per block to maximize gpu utilization
        constexpr int threads_per_block = 128;

        int warps_per_block = (threads_per_block / warp_size);
        int batches_per_block = warps_per_block * batches_per_warp;
        int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
        dim3 threads(warp_size, warps_per_block, 1);
        // Launch code would be more elegant if C++ supported FOR CONSTEXPR
        switch (log2_elements) {
            #define LAUNCH_SOFTMAX_WARP_FORWARD(L2E) case L2E:                    \
            softmax_warp_forward<input_t, output_t, acc_t, L2E, is_log_softmax, is_masked>   \
                <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst,   \
                    src, batch_count, softmax_elements_stride, softmax_elements, mask, chunk_size, is_transformer_mask); \
            C10_CUDA_KERNEL_LAUNCH_CHECK();                                       \
            break;

            LAUNCH_SOFTMAX_WARP_FORWARD(0);  // 1
            LAUNCH_SOFTMAX_WARP_FORWARD(1);  // 2
            LAUNCH_SOFTMAX_WARP_FORWARD(2);  // 4
            LAUNCH_SOFTMAX_WARP_FORWARD(3);  // 8
            LAUNCH_SOFTMAX_WARP_FORWARD(4);  // 16
            LAUNCH_SOFTMAX_WARP_FORWARD(5);  // 32
            LAUNCH_SOFTMAX_WARP_FORWARD(6);  // 64
            LAUNCH_SOFTMAX_WARP_FORWARD(7);  // 128
            LAUNCH_SOFTMAX_WARP_FORWARD(8);  // 256
            LAUNCH_SOFTMAX_WARP_FORWARD(9);  // 512
            LAUNCH_SOFTMAX_WARP_FORWARD(10); ; // 1024
            default:
                break;
        }
    }
}
```

下面终于进入到了kernel部分，代码比较长，分开来看

```text
template <typename input_t, typename output_t, typename acc_t, int log2_elements, bool is_log_softmax, bool is_masked>
__global__ void softmax_warp_forward(output_t *dst, const input_t *src, int batch_size, int stride, int element_count, const bool *mask = nullptr, const int head_chunk_size = -1, bool is_transformer_mask = false)
{
    // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and warp_size of method warp_softmax_forward_kernel.
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int WARP_SIZE = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
    constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
    constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;
```

第一步就是经典的计算idx，由于有grid block两个维度，而且很多时候可以人为规定一些事情，实际上这里并不好计算，一般来说，可以将其看作一个列优先矩阵。

那么，如果将grid看作一个列优先矩阵，其stride就为(1,gridDim.x, gridDim.x\*gridDim.y)

grid需要block的idx（一直觉得这么设计很乱），对于坐标（blockIdx.x, blockIdx.y, blockIdx.z） 其idx就为

```text
idx_for_block = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z* gridDim.x*gridDim.y
```

同理，对于block级别：

```text
idx_for_thread = threadIdx.x +  threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y
```

对于一个block来说，其有blockDim.x \* blockDim.y \* blockDim.z个thread，所以最终的索引为

```text
idx = idx_for_block * blockDim.x * blockDim.y * blockDim.z + idx_for_thread 
= (blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z* gridDim.x*gridDim.y) * blockDim.x * blockDim.y * blockDim.z 
     + threadIdx.x +  threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y
```

当然，这么写也过于复杂了，一般来说，如果一个维度没有用到，那么那个维度的dim就是1，那个维度的idx就是0，可以去掉。softmax\_warp\_forward中grid只用到了x维度，block用到了x,y维，

这样，gridDim.y gridDim.z blockDim.z都是1，blockIdx.y blockIdx.z threadIdx.z都是0

那么公式就变为了

```text
blockIdx.x * blockDim.x * blockDim.y + threadIdx.x +  threadIdx.y * blockDim.x
```

但是这个只是一般用法，假定了都是列优先，实际上是比较灵活的，可以自己定义维度的用途。

这里的softmax kernel代码就是这样，观察此代码

```text
    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;
```

为什么和之前分析的不一样？回忆一下之前的设定，grid就一维，大小为(batch\_count + batches\_per\_block - 1) / batches\_per\_block;

每个block处理batches\_per\_block个batch，block有x y两维，一个是warp\_size 一个是warps\_per\_block = (threads\_per\_block / warp\_size)，总共还是128线程。

而warp\_size是负责处理1个或两个batch，这个值被定义为WARP\_BATCH。

所以，一个block可以处理的batch数量，是blockDim.y \* blockIdx.x \* WARP\_BATCH

而这个block中的线程，其负责处理的batch的相对坐标，就是 threadIdx.y \* WARP\_BATCH

所以其绝对坐标，就是 (blockDim.y \* blockIdx.x + threadIdx.y) \* WARP\_BATCH;

代码接着就是一个经典的处理尾部数据的操作，因为要把数据均匀分给block，但是最后一部分数据可能因为没整除开，比较少，所以得特殊处理一下。

```text
    int local_batches = batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;
```

接着再根据Idx找到数据指针。local\_idx 是指的当前线程在其warp中的idx 注意，这里的src dst，对于每个线程看到的值，已经不同了，每个线程看到的是其私有值

```text
    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x;
    int idx_offset = first_batch * stride + local_idx;

    src += idx_offset;
    dst += idx_offset;
```

下面将数据从显存读入寄存器，而且两层循环让逻辑更加清晰，第一层是warp\_batch, 2或者1 以2为例，不一定整除，所以batch\_element\_count可以是0

对数据的索引，仍然有些抽象，大体还是有一个大索引，一个local索引

由于一个warp可以负责多个batch, 所以当前负责batch的大索引是i\*element\_count

其实这个完整的应该写成这样 src\[first\_batch \* stride + threadIdx.x + it\*WARP\_SIZE \]

其中WARP\_ITERATIONS = next\_power\_of\_two / WARP\_SIZE;

这时候大体就可以看出来，这个任务是如何划分给各个线程的了，举一个例子，如果next\_power\_of\_two=128，WARP\_SIZE=32，那么WARP\_ITERATIONS=4。

first\_batch上面分析过了，我们就假设这个block与threadIdx.y就是处理0

threadIdx.x 在0~31之间，it在0~3之间。

那么对于threadIdx.x=0, 其负责0，32，64，96这几个坐标的数据

32个线程，共负责0~127这几个坐标的数据 正好把一个next\_power\_of\_two处理完。

所以elements是一个线程私有数据，0维是batch索引，每个batch占用1个1维，1维按照warp size切分开

```text
    acc_t elements[WARP_BATCH][WARP_ITERATIONS];
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < batch_element_count) {
                elements[i][it] = src[i*element_count+it*WARP_SIZE];
            } else {
                elements[i][it] = -std::numeric_limits<acc_t>::infinity();
            }
        }
    }
```

下面看如果计算最大值，最大值在softmax中也比较常见，主要是防止数据溢出或者太多不准确。

这时候有小伙伴就说了，计算最大值还不简单 max()搞定。但是如果想要极致的性能，就需要一点点理清数据的所在位置，因为寄存器中值，线程是无法互相看见的。那如何同步数据，一个是通过smem甚至gmem转一手就可以，也可以用warp之间的同步指令，这就是为什么做优化还需要知道gpu体系结构

```text
    // compute max_value
    acc_t max_value[WARP_BATCH];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        bool is_meaningful_max = false;
        max_value[i] = elements[i][0];
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            if (is_masked) {
                int idx = it*WARP_SIZE;
                if ((idx + local_idx) < batch_element_count) {
                    if (!is_transformer_mask) {
                        idx += i*element_count;
                    }
                    if (!mask[idx]) {
                        max_value[i] = (is_meaningful_max && max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
                        is_meaningful_max = true;
                    }
                }
            } else {
                max_value[i] = max_value[i] > elements[i][it] ? max_value[i] : elements[i][it];
            }
        }
        if (is_masked) {
            if (!is_meaningful_max) {
                max_value[i] = -std::numeric_limits<acc_t>::infinity();
            }
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Max>(max_value);
```

在分析上面代码之前，先看一下warp reduce操作，reduce就是sum max等操作，也就是所有数据一起出一个结果，个人认为，底层中通讯比计算困难。

这里又涉及到了\_\_shfl\_xor\_sync命令，简单来说，这条命令的作用就是把其他线程的value，return给当前线程。

那么，如何区分线程？我们知道cuda有threadIdx的概念，不过这里用的不是这个，而是线程在warp中的编号0~31，大体可以理解为idx%32就可以了

那么，当前的线程号^laneMask，就是目标线程，通过蝶式规约，就可以让值慢慢的传递。

这里举一个例子，假设warp size是8（搞小一点好看，32太大了不好分析）

```text
template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if !defined(USE_ROCM)
    return __shfl_xor_sync(mask, value, laneMask, width);
#else
    return __shfl_xor(value, laneMask, width);
#endif
}


template <typename acc_t, int WARP_BATCH, int WARP_SIZE, template<typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce(acc_t* sum) {
    ReduceOp<acc_t> r;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;  i < WARP_BATCH;  ++i) {
            acc_t b = WARP_SHFL_XOR(sum[i], offset, WARP_SIZE);
            sum[i] = r(sum[i], b);
        }
    }
}
```

先写一个简单的伪代码

```text
for num in range(8):
    print("{} ".format(num),end="")
print()
mask=4
while(mask>0):
    for num in range(8):
        xor_result = num ^ mask
        print("{} ".format(xor_result),end="")
    print()
    mask //=2
```

输出结果

```text
0 1 2 3 4 5 6 7 
4 5 6 7 0 1 2 3 
2 3 0 1 6 7 4 5 
1 0 3 2 5 4 7 6
```

我们聚焦于0线程，0第一步获取4的值，第二布获取2的值，注意2在第一步获取了6的值，第三步0获取1的值，而1在第一步获取了5的值，第二部获取了3的值，3在第一步获取了7的值，所以最终，0，1，2，3，4，5，6，7，这八个值就全在一起了。如果把这数据连成线的话，据说会出现一只美丽的蝴蝶，所以叫蝶式规约，不过本人从来没看出来过，哪里像了，这不拉丝山药吗。

在看上面代码，一开始很简单，就是遍历找最大值，这个是线程内部寻找最大值。

max\_value\[i\] = max\_value\[i\] > elements\[i\]\[it\] ? max\_value\[i\] : elements\[i\]\[it\];

找到后，在通过warp线程之间规约一下，那么每个线程的max\_value就是真正的最大值了。

  

下面就是算exp后总和，和上面逻辑差不多，这里主要记录一下log\_softmax, 就是在softmax基础上加个Log，防止溢出，而且计算方便，从算法原理上没有太大区别

![](https://pic2.zhimg.com/v2-49d85c504cdee2999a30d16f3454bc67_1440w.jpg)

https://www.zhihu.com/question/358069078

```text
    acc_t sum[WARP_BATCH] { 0.0f };
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            if (!is_masked) {
                if (is_log_softmax) {
                    sum[i] += std::exp(elements[i][it] - max_value[i]);
                } else {
                    elements[i][it] = std::exp(elements[i][it] - max_value[i]);
                    sum[i] += elements[i][it];
                }
            } else {
                int idx = it*WARP_SIZE;
                bool valid = (idx + local_idx) < batch_element_count;
                if (!is_transformer_mask) {
                    idx += i*element_count;
                }
                if (valid) {
                    if (!mask[idx]) {
                        if (is_log_softmax) {
                            sum[i] += std::exp(elements[i][it] - max_value[i]);
                        } else {
                            elements[i][it] = std::exp(elements[i][it] - max_value[i]);
                            sum[i] += elements[i][it];
                        }
                    } else {
                        if (!is_log_softmax) {
                            // Masked values are treated as -infinity, and std::exp(-infinity) is 0.
                            elements[i][it] = 0;
                        }
                    }
                } else {
                    if (!is_log_softmax) {
                        elements[i][it] = 0.;
                    }
                }
            }
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);
```

最后再将数据存储回去就可以了，可以看出，softmax主要难点 一个在于将任务划分给block thread，通过很多idx找到对应的数值，一个在于线程间通信reduce

```text
    // store result
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        if (i >= local_batches)
            break;
        if (is_log_softmax) sum[i] = std::log(sum[i]);
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                if (is_log_softmax) {
                    dst[i*element_count+it*WARP_SIZE] = elements[i][it] - max_value[i] - sum[i];
                } else if (sum[i] == 0) {
                    dst[i*element_count+it*WARP_SIZE] = std::numeric_limits<acc_t>::quiet_NaN();
                } else {
                    dst[i*element_count+it*WARP_SIZE] = elements[i][it] / sum[i];
                }
            } else {
                break;
            }
        }
    }
```

本篇文章分析了简单情况下的softmax计算，下面我们继续分析，复杂情况的softmax计算及优化