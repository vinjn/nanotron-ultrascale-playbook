# pytorch 源码分析 （5）softmax （2）

**Author:** 阿嚏

**Date:** 2025-06-17

**Link:** https://zhuanlan.zhihu.com/p/1918256956612777526

接上文 [pytorch 源码分析（4）softmax （1）](https://zhuanlan.zhihu.com/p/1917970073152385299) 本文继续分析softmax的其他情况。

本篇分析同样为inner\_size == 1，但是dim\_size比较大的情况

首先是[ILP](https://zhida.zhihu.com/search?content_id=259180939&content_type=Article&match_order=1&q=ILP&zhida_source=entity)，nvidia gpu一个线程最多可以一次性读写128bit，也就是4个float，这里ILP是一个线程一次性处理的数量。SoftMaxForward\_getBlockSize代码逻辑很简单，简单来说，就是block = min(dim\_size,1024)

这里有两个类型，一个是accscalar\_t，scalar\_t，其实上文也出现了，scalar\_t是input tensor的类型，accscalar\_t是在计算过程中（例如sum max）中间变量的类型，中间变量精度高一点有助于结果精度高，但是会占用更多的资源。

smem\_reduction\_sz 每个warp只需要存储一个值就可以，通过剩余的smem大小，计算出来max\_elements\_per\_smem 从而得知是否有足够的smem可以使用。

同时，使用smem还需要对齐到16byte，我记得tma是有这个要求的

```text
            constexpr int ILP = sizeof(float4) / sizeof(scalar_t);
            dim3 block = SoftMaxForward_getBlockSize(dim_size);
            size_t smem_reduction_sz = block.x / C10_WARP_SIZE * sizeof(accscalar_t);
            auto max_elements_per_smem = (at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock -
              smem_reduction_sz) / sizeof(scalar_t);

            bool can_use_smem = dim_size < max_elements_per_smem;
            can_use_smem &= !(reinterpret_cast<const uintptr_t>(input_ptr) % ALIGN_BYTES);
            can_use_smem &= (!(reinterpret_cast<uintptr_t>(output_ptr) % ALIGN_BYTES));
            can_use_smem &= !(dim_size % ILP);

            if (can_use_smem) {
              size_t smem_sz = dim_size * sizeof(scalar_t) + smem_reduction_sz;
              cunn_SoftMaxForwardSmem<ILP, scalar_t, accscalar_t, scalar_t, Epilogue>
                <<<grid, block, smem_sz, stream>>>(output_ptr, input_ptr, dim_size);
            } else {
              cunn_SoftMaxForward<ILP, scalar_t, accscalar_t, scalar_t, Epilogue>
                <<<grid, block, smem_reduction_sz, stream>>>(output_ptr, input_ptr, dim_size);
            }

            C10_CUDA_KERNEL_LAUNCH_CHECK();
```

先看下cunn\_SoftMaxForwardSmem，当共享内存充足时，如何做，这部分代码也比较长，分段来看

看第一部分，这时就可以看到smem的用途，分为两部分，第一部分用来存储input，大小为 classes \* sizeof(scalar\_t) 第二部分是一些规约操作的结果，类型为accscalar\_t 注意这里再次出现了LoadT 向量化读写

```text
template <int ILP, typename scalar_t, typename accscalar_t, typename outscalar_t,
  template <typename, typename, typename> class Epilogue, typename index_t = int32_t>
__global__ void
cunn_SoftMaxForwardSmem(outscalar_t *output, const scalar_t *input, index_t classes)
{
  // Each thread block processes a sample in the batch
  input += static_cast<int64_t>(blockIdx.x) * classes;
  output += static_cast<int64_t>(blockIdx.x) * classes;

  accscalar_t threadMax = -at::numeric_limits<accscalar_t>::max();
  accscalar_t threadExp = static_cast<accscalar_t>(0);

  // The first smem segment is used to cache input values and the last
  // segment is used for thread block reductions
  extern __shared__ unsigned char smem[];
  auto smem_input_cache = reinterpret_cast<scalar_t*>(smem);
  auto smem_reduction_cache = reinterpret_cast<accscalar_t*>(smem +
    classes * sizeof(scalar_t));

  using LoadT = at::native::memory::aligned_vector<scalar_t, ILP>;
  const LoadT* const input_vec_ptr = reinterpret_cast<const LoadT*>(input);
  LoadT* const smem_input_cache_vec_ptr = reinterpret_cast<LoadT*>(smem_input_cache);
```

开始向量化读取数据到smem中，这里读取的是Input。

举一个例子，如果dim\_size=8192, 那么block=1024 = blockDim.x

注意，input 先是+ blockIdx.x \* dim\_size

对于blockIdx.x = 0 threadIdx.x = 0, 其负责0,1024 但是有向量化读写，所以其实是0-3，4096-4099

对于blockIdx.x = 0 threadIdx.x = 1 其负责 1，1025 向量化读写，其实是4-7 1025\*4-1025\* 4+3

对于blockIdx.x = 1 threadIdx.x = 0 其负责2048，3072 依次类推

依旧是大索引，小索引，将所有数据处理好。

在遍历读入Input的时候，顺便把最大值记录一下，这里threadMax初始化为-at::numeric\_limits<accscalar\_t>::max();个人觉得并不好，因为-max不一定是min哈哈。

注意到，数据先是到了crnt\_vec再进入smem，这里其实是老款gpu架构，数据只能从寄存器倒一手，新款gpu用tma可以之间从显存到smem，这是一个小优化点。

最后，threadMax 就是这个线程处理的数据的最大值，注意并不是全部数据，只是这个线程经手的数据。

```text
  // Download inputs to shared memory while doing the first step
  // in max calculation
  MaxFloat<scalar_t, accscalar_t> maxFunc;
  for (index_t offset = threadIdx.x; offset * ILP < classes; offset += blockDim.x) {
    LoadT crnt_vec = input_vec_ptr[offset];
    smem_input_cache_vec_ptr[offset] = crnt_vec;

    #pragma unroll
    for (int i = 0; i < ILP; ++i) {
      threadMax = maxFunc(threadMax, crnt_vec.val[i]);
    }
  }
```

接着是一个block级别的reduce,warp级别的reduce上文已经讲过了，快，但是可以reduce的线程有限，一般只有32个线程，但是block级别可以让这个block的线程都进行reduce。

block是在一个sm运行的，不会跨sm运行，所以通过sm中的smem进行数据同步自然是首选选择

我们深入看一下是如何做的，首先，对于每个tid，也就是blockIdx.x，其有一个local id 也就是在自己所在warp中的线程id 0-31,也有一个warp id，也就是其所在warp的id ,这两者分别被简写为lid wid。

第一步先进行warp级别的规约，然后由每个warp的第一个线程，将数据写到shared的对应位置。

然后有一个奇妙的逻辑，由第一个warp 32个线程，将读入smem中的值，读入到自己的val，再次进行依次规约，也就相当于最多32\*32=1024个数据进行了规约，再回头看一下SoftMaxForward\_getBlockSize，就知道这1024值是怎么来的原因了。同时，也知道了smem\_reduction\_sz 是怎么来的，这就是一个cache，跟程序运行时分配块内存没太大区别

```text
 template <typename T, class ReduceOp, typename B = Block1D>
__inline__ __device__ T
BlockReduce(T val, const ReduceOp& op, const T& identity_element, T* shared) {
  const int tid = B::Tid();
  const int lid = tid % C10_WARP_SIZE;
  const int wid = tid / C10_WARP_SIZE;
  val = WarpReduce(val, op);
  __syncthreads(); // prevent races when BlockReduces are called in a row.
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (tid < B::Warps()) ? shared[lid] : identity_element;
  if (wid == 0) {
    val = WarpReduce(val, op);
  }
  return val;
}

// Performs a thread block reduction with a given functor but uses
// warp shuffles as the first step in the reduction
template <template<typename> class Reduction, typename T>
__device__ __forceinline__
T blockReduceWarp(T* smem_cache, T value, const Reduction<T>& op, T defaultVal)
{
  T result = cuda_utils::BlockReduce<T, Reduction<T>>(value, op, defaultVal, smem_cache);
  if (threadIdx.x == 0) {
    smem_cache[0] = result;
  }
  __syncthreads();
  return smem_cache[0];
}


accscalar_t max_k = blockReduceWarp<Max, accscalar_t>(smem_reduction_cache, threadMax,
    Max<accscalar_t>(), -at::numeric_limits<accscalar_t>::max());
```

下面开始进行softmax的计算，这块反而是最简单的，就是单纯的sum + std::exp(v - max\_k);

```text
template<typename T, typename AccumT>
struct SumExpFloat
{
  __device__ __forceinline__ SumExpFloat(AccumT v)
    : max_k(v) {}

  __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
    return sum + std::exp(v - max_k);
  }

  const AccumT max_k;
};

  SumExpFloat<scalar_t, accscalar_t> sumExpFunc(max_k);
  for (index_t offset = threadIdx.x; offset * ILP < classes; offset += blockDim.x) {
    LoadT crnt_vec = smem_input_cache_vec_ptr[offset];

    #pragma unroll
    for (int i = 0; i < ILP; ++i) {
      threadExp = sumExpFunc(threadExp, crnt_vec.val[i]);
    }
  }
```

  

计算总和和上面计算最大值是一样的，注意的是，同样使用了smem\_reduction\_cache，这个cache可以重复利用。

```text
  accscalar_t sumAll = blockReduceWarp<Add, accscalar_t>(smem_reduction_cache, threadExp,
    Add<accscalar_t>(), static_cast<accscalar_t>(0));
```

  

最后计算一下数据，再将数据写回就可以了，其实softmax计算的难点，就是计算max 和 sum这两个reduce操作，本身计算反而不复杂。

  

```text
 template<typename T, typename AccumT, typename OutT>
struct SoftMaxForwardEpilogue {
  __device__ __forceinline__ SoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
    : max_input(max_input)
    , sum(sum) {}

  __device__ __forceinline__ OutT operator()(T input) const {
    return static_cast<OutT>(std::exp(input - max_input) / sum);
  }

  const AccumT max_input;
  const AccumT sum;
}; 

  Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(max_k, sumAll);

  // Use vectorized stores to save the output
  using StoreT = at::native::memory::aligned_vector<outscalar_t, ILP>;
  StoreT* output_vec_ptr = reinterpret_cast<StoreT*>(output);
  for (index_t offset = threadIdx.x; offset * ILP < classes; offset += blockDim.x) {
    LoadT crnt_vec = smem_input_cache_vec_ptr[offset];
    StoreT out_vec;

    #pragma unroll
    for (int i = 0; i < ILP; ++i) {
      out_vec.val[i] = epilogue(crnt_vec.val[i]);
    }

    output_vec_ptr[offset] = out_vec;
  }
```

  

在很早之前就体会过，排序算法，本身很简单，但要是内存不够，那就难了，那么如果smem不够，softmax还怎么计算呢。

首先，smem\_reduction\_sz 还是 block.x / C10\_WARP\_SIZE \* sizeof(accscalar\_t)，这个值是很小的，必须得够，也就是必须有个cache用来reduce计算。下面我们看一下cunn\_SoftMaxForward，参数基本和cunn\_SoftMaxForwardSmem一致。

```text
            if (can_use_smem) {
              size_t smem_sz = dim_size * sizeof(scalar_t) + smem_reduction_sz;
              cunn_SoftMaxForwardSmem<ILP, scalar_t, accscalar_t, scalar_t, Epilogue>
                <<<grid, block, smem_sz, stream>>>(output_ptr, input_ptr, dim_size);
            } else {
              cunn_SoftMaxForward<ILP, scalar_t, accscalar_t, scalar_t, Epilogue>
                <<<grid, block, smem_reduction_sz, stream>>>(output_ptr, input_ptr, dim_size);
            }
```

cunn\_SoftMaxForward和cunn\_SoftMaxForwardSmem还是非常像的，主要区别在这个ilpReduce

```text
template <int ILP, typename scalar_t, typename accscalar_t, typename outscalar_t, template <typename, typename, typename> class Epilogue>
__global__ void
cunn_SoftMaxForward(outscalar_t *output, const scalar_t *input, int classes)
{
  extern __shared__ unsigned char smem[];
  auto sdata = reinterpret_cast<accscalar_t*>(smem);

  // forward pointers to batch[blockIdx.x]
  // each block handles a sample in the mini-batch
  input += static_cast<int64_t>(blockIdx.x) * classes;
  output += static_cast<int64_t>(blockIdx.x) * classes;

  const int shift = ((uint64_t)input) % ALIGN_BYTES / sizeof(scalar_t);
  const int output_shift = ((uint64_t)output) % ALIGN_BYTES / sizeof(outscalar_t);

  // find the max
  accscalar_t threadMax = ilpReduce<MaxFloat, ILP, scalar_t, accscalar_t>(
    shift, input, classes, MaxFloat<scalar_t, accscalar_t>(), -at::numeric_limits<accscalar_t>::max());
  accscalar_t max_k = blockReduceWarp<Max, accscalar_t>(sdata, threadMax,
    Max<accscalar_t>(), -at::numeric_limits<accscalar_t>::max());

  // reduce all values
  accscalar_t threadExp = ilpReduce<SumExpFloat, ILP, scalar_t, accscalar_t>(
    shift, input, classes, SumExpFloat<scalar_t, accscalar_t>(max_k), static_cast<accscalar_t>(0));
  accscalar_t sumAll = blockReduceWarp<Add, accscalar_t>(sdata, threadExp,
    Add<accscalar_t>(), static_cast<accscalar_t>(0));

  Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(max_k, sumAll);

  if (shift == output_shift) {
    WriteFpropResultsVectorized<ILP, scalar_t, accscalar_t, outscalar_t, Epilogue>(classes, shift, input, output, epilogue);
  } else {
    WriteFpropResults<ILP, scalar_t, accscalar_t, outscalar_t, Epilogue>(classes, input, output, epilogue);
  }
}
```

略过对齐的逻辑，假设shift为0，其实逻辑和smem没有什么不同，只是数据从显存读入寄存器计算，就结束了

只不过因为加了向量化读写，一些尾部数据需要特殊处理一下，就可以了。但是注意到，由于没有smem这一层，数据多次的访问显存，效率肯定低。

```text
template <template<typename, typename> class Reduction, int ILP, typename T, typename AccumT, typename index_t=int>
__device__ __forceinline__ AccumT
ilpReduce(index_t shift,
          const T* data,
          index_t size,
          const Reduction<T, AccumT>& r,
          AccumT defaultVal)
{
  using LoadT = at::native::memory::aligned_vector<T, ILP>;
  AccumT threadVal = defaultVal;
  index_t offset = threadIdx.x;

  // shift and do 1
  if(shift > 0){
    data -= shift;
    size += shift;
    if(threadIdx.x >= shift){
      threadVal = r(threadVal, data[offset]);
    }
    size -= blockDim.x;
    data += blockDim.x;
  }
  index_t last = size % (ILP * blockDim.x);

  T v[ILP];
  LoadT* value = reinterpret_cast<LoadT*>(&v);

  for (; offset * ILP < (size - last); offset += blockDim.x) {
    *value = reinterpret_cast<const LoadT*>(data)[offset];

    #pragma unroll
    for (int j = 0; j < ILP; ++j) {
      threadVal = r(threadVal, v[j]);
    }
  }

  offset = size - last + threadIdx.x;
  // Epilogue
  for (; offset < size; offset += blockDim.x)
    threadVal = r(threadVal, data[offset]);

  return threadVal;
}
```

本文分析完毕了inner\_size == 1情况下的softmax计算，后面有机会会继续分析inner\_size不为1的情况