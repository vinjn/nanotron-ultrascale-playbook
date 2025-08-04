# pytorch 源码阅读 （6）layernorm

**Author:** 阿嚏

**Date:** 2025-06-17

**Link:** https://zhuanlan.zhihu.com/p/1918352104982045230

[batchnorm](https://zhida.zhihu.com/search?content_id=259200372&content_type=Article&match_order=1&q=batchnorm&zhida_source=entity)和[layernorm](https://zhida.zhihu.com/search?content_id=259200372&content_type=Article&match_order=1&q=layernorm&zhida_source=entity)都是做归一化，只是维度不同，之前也说过，作为三维生物人的脑子很难理解高维，还是用二维好理解：现在有一个(batch,hidden\_dim) / (2，4)

```text
batch1  0 1 2 3
batch2  4 5 6 7
```

那么batchnorm 就是0和4计算均值方差，1 5计算均值方差，。。。

layernorm是 0 1 2 3计算均值方差，。。。

话不多说，直接上代码，核心是调用[LayerNormKernelImpl](https://zhida.zhihu.com/search?content_id=259200372&content_type=Article&match_order=1&q=LayerNormKernelImpl&zhida_source=entity)Internal，pytorch的dispatch机制有时间需要分析一下。

其中参数XY是输入输出，gamma beta是缩放偏移，M N就是上面说的(batch,hidden\_dim)，mean rstd是将计算得到的均值方差填入，注意rstd是标准差的倒数（本文没咋区分方差和标准差，按语义理解就行。。。）

```text
void LayerNormKernelImpl(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    double eps,
    Tensor* Y,
    Tensor* mean,
    Tensor* rstd) {
  TORCH_DCHECK_EQ(X.numel(), M * N);
  DCHECK(!gamma.defined() || gamma.numel() == N);
  DCHECK(!beta.defined() || beta.numel() == N);
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, X.scalar_type(),
      "LayerNormKernelImpl", [&]() {
    LayerNormKernelImplInternal<scalar_t>(
        X, gamma, beta, M, N, eps, Y, mean, rstd);
  });
}
```

LayerNormKernelImplInternal的第一步就是准备各种指针，也就是data\_ptr数据指针。

```text
template <typename T, typename T_ACC>
void LayerNormKernelImplInternal(
    const Tensor& X,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    T_ACC eps,
    Tensor* Y,
    Tensor* mean,
    Tensor* rstd) {
  // assumes input, gamma and beta are of proper shape, this was checked in _check_layer_norm_inputs
  // assumes all tensors are contiguous
  TORCH_CHECK(M <= at::cuda::getCurrentDeviceProperties()->maxGridSize[0], "M should be less than maximum CUDA grid size, \
  file a support request to support bigger batches");
  const T* X_data = X.const_data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.const_data_ptr<T>() : nullptr;
  const T* beta_data = beta.defined() ? beta.const_data_ptr<T>() : nullptr;
  T* Y_data = Y->data_ptr<T>();
  T_ACC* mean_data = mean->data_ptr<T_ACC>();
  T_ACC* rstd_data = rstd->data_ptr<T_ACC>();
```

接着是看看能不能向量化，向量化在底层kernel也是比较常见的优化了

```text
  // check if can take fast path - all tensors are properly aligned, N is less than 2^24 (to use float count),
  // N is multiple of vec_size (so that all rows are aligned if tensor is aligned)
  constexpr int num_vec_elems = vec_size;
  constexpr int alignment = num_vec_elems * sizeof(T);
  bool can_vec_X = can_vectorize(X_data, alignment);
  bool can_vec_Y = can_vectorize(Y_data, alignment);
  bool can_vec_gamma = gamma.defined() ? can_vectorize(gamma_data, alignment) : true;
  bool can_vec_beta = beta.defined() ? can_vectorize(beta_data, alignment) : true;
```

如果类型合适，而且满足向量化，N也比较小，那么就可以使用launch\_vectorized\_layer\_norm\_kernel

```text
  if ((std::is_same<T, float>::value || std::is_same<T, at::Half>::value || std::is_same<T, at::BFloat16>::value) &&
  N <= static_cast<int64_t>(1ULL << std::numeric_limits<float>::digits) && N % num_vec_elems == 0 &&
  can_vec_X && can_vec_Y && can_vec_gamma && can_vec_beta) {
    launch_vectorized_layer_norm_kernel(static_cast<int>(N), M, eps, X_data, gamma_data, beta_data, Y_data, mean_data, rstd_data);
```

而其实launch\_vectorized\_layer\_norm\_kernel也只是个封装，负责启动真正Kernel, block数量设置为M,也就是每个block负责一个batch，线程数量按照（32，4）设置好。nshared值需要进一步分析。

```text
template <typename T, typename T_ACC>
void launch_vectorized_layer_norm_kernel(
  int N,
  int64_t M,
  T_ACC eps,
  const T* X_data,
  const T* gamma_data,
  const T* beta_data,
  T* Y_data,
  T_ACC* mean_data,
  T_ACC* rstd_data
) {
    //constexpr int alignment = 16; //currently unused to make sure float and half results are bw accurate
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int warp_size = at::cuda::warp_size();
    const dim3 threads(warp_size, num_threads() / warp_size, 1);
    const dim3 blocks(M);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(threads.y % 2 == 0 || threads.y == 1);
    int nshared = threads.y > 1 ? threads.y * 3/2 *sizeof(T_ACC) : 0;
    vectorized_layer_norm_kernel<<<blocks, threads, nshared, stream>>>(N, eps, X_data,
    gamma_data, beta_data, mean_data, rstd_data, Y_data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
```

vectorized\_layer\_norm\_kernel只是调用vectorized\_layer\_norm\_kernel\_impl，直接分析vectorized\_layer\_norm\_kernel\_impl，这部分较长，逐步分析。

首先第一步是compute\_stats，如果有很多数据，如何算其均值方差？有点类似与softmax online计算，就是先算一部分，计算均值方差，等新数据来，再进行一定改动，这叫做Welford算法。

```text
template <typename T, typename T_ACC,
typename std::enable_if<!std::is_same<T, double>::value, int>::type = 0>
__device__ __inline__ void vectorized_layer_norm_kernel_impl(
  const int N,
  T_ACC eps,
  const  T* __restrict__ X,
  const  T* gamma,
  const  T* beta,
  T_ACC* mean,
  T_ACC* rstd,
  T* Y){
    extern __shared__ float s_data[]; //if we made smem WelfordDataLN type, there would be bank conflicts,
    //as one thread would have to write 3 consecutive floats
    auto i1 = blockIdx.x;
    const T * block_row = X + i1 * N;
    WelfordDataLN wd = compute_stats(block_row, N, s_data);
```

WelfordDataLN数据很简单，就是均值，二阶中心矩，还是当前元素数量

```text
struct WelfordDataLN{
  float mean;
  float sigma2;
  float count;
  C10_HOST_DEVICE WelfordDataLN(): mean(0.f), sigma2(0.f), count(0.f){}
  C10_HOST_DEVICE WelfordDataLN(float mean, float sigma2, float count): mean(mean), sigma2(sigma2), count(count) {}
};
```

那么有新数据来了，如何计算呢？count好说+1，原来的总和是curr\_sum.mean\*curr\_sum.count，现在是curr\_sum.mean\*curr\_sum.count + val,

那么新的平均值就是（curr\_sum.mean\*curr\_sum.count + val） / （curr\_sum.count+1）

\= （curr\_sum.mean\*（curr\_sum.count + 1） + val - curr\_sum.mean） / （curr\_sum.count+1）

\= curr\_sum.mean + （ val - curr\_sum.mean）/（curr\_sum.count+1）

方差的推导要复杂很多，感兴趣可以参考[Welford算法小记](https://zhuanlan.zhihu.com/p/408474710)

```text
template<typename U> __device__
WelfordDataLN cuWelfordOnlineSum(
  const U val,
  const WelfordDataLN& curr_sum)
{
  U delta = val - curr_sum.mean;
  U new_count = curr_sum.count + 1.f;
  U new_mean = curr_sum.mean + delta * (1.f/new_count); //proper division is slow, this is less accurate but noticeably faster
  return {new_mean, curr_sum.sigma2 + delta * (val - new_mean), new_count};
}
```

这样对于每个线程，都可以获得其处理数据的均值方差，但是，也只是线程级别的，那么，线程之间还需要通讯的，cuWelfordOnlineSum是在一个现有的WelfordDataLN上更新一个值，而现在的需求是多个WelfordDataLN融合成一个，这就是cuWelfordCombine了

下面代码就是warp内部的规约操作：

```text
    // intra-warp reduction
    for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
        WelfordDataLN wdB{WARP_SHFL_DOWN(wd.mean, offset),
        WARP_SHFL_DOWN(wd.sigma2, offset), WARP_SHFL_DOWN(wd.count, offset)};
        wd = cuWelfordCombine(wd, wdB);
    }
```

但是光warp内部规约还不够，还需要Block级别的规约操作，这时候就用到smem了，我们看pytorch是如何做的：

首先如果blockDim.y==1，说明只有一个warp，也就不用再规约了。

接着，就是需要用到了smem buf，原理上和warp内规约没有太大不同，只是这里是后半段warp需要将数据存储到smem中，共计使用3\*0.5=1.5倍smem，呼应上文nshared，而前半段warp再负责cuWelfordCombine，最后返回结果。

```text
    if (blockDim.y > 1) {
      float * meansigmabuf = buf;
      float * countbuf = buf + blockDim.y;
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_y = threadIdx.y - offset;
          meansigmabuf[2*wrt_y] = wd.mean;
          meansigmabuf[2*wrt_y+1] = wd.sigma2;
          countbuf[wrt_y] = wd.count;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          WelfordDataLN wdB{meansigmabuf[2*threadIdx.y],
                          meansigmabuf[2*threadIdx.y+1],
                          countbuf[threadIdx.y]};
          wd = cuWelfordCombine(wd, wdB);
        }
        __syncthreads();
      }
      if (threadIdx.x == 0 && threadIdx.y ==0) {
        meansigmabuf[0] = wd.mean;
        meansigmabuf[1] = wd.sigma2/float(N);
      }
      __syncthreads();
      return WelfordDataLN{meansigmabuf[0], meansigmabuf[1],0.f};

    } else {
      return WelfordDataLN{WARP_SHFL(wd.mean,0), WARP_SHFL(wd.sigma2,0)/float(N), 0.f};
    }
```

compute\_stats基本分析完了，就是在计算均值方差，下面是一些计算前的准备。

一个是准备向量化读写的指针，一个是一些索引。

const T \* block\_row = X + i1 \* N; 每个block处理N个数据，所以block\_row就是处理数据的指针。

numx thrx 在[pytorch 源码分析（4）softmax （1）](https://zhuanlan.zhihu.com/p/1917970073152385299)这篇文章分析过类似的，这里就不分析啦

rstd\_val 计算标准差 的倒数，加个eps防止成0

```text
    using vec_t = aligned_vector<T, vec_size>;
    const vec_t * X_vec = reinterpret_cast<const vec_t*>(block_row);
    const vec_t * gamma_vec = (gamma != nullptr) ? reinterpret_cast<const vec_t*>(gamma) : nullptr;
    const vec_t * beta_vec = (beta != nullptr) ? reinterpret_cast<const vec_t*>(beta) : nullptr;
    vec_t * Y_vec = reinterpret_cast<vec_t*>(Y + i1 * N);

    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const int n_vec_to_read = N/vec_size;

    T_ACC rstd_val = c10::cuda::compat::rsqrt(wd.sigma2 + eps);
```

终于到了最后的运算，发现很多kernel都是前期准备时间比较长，到最后运算反而简单了。

基本就是在计算rstd\_val \* (data.val\[ii\]) - wd.mean) 只是有时候需要通过gamma\_vec beta\_vec 加个缩放偏移什么的。忽略不计。

```text
    // No tail, N is guaranteed to be multiple of vec size
    for (int i = thrx; i < n_vec_to_read; i += numx) {
      vec_t data = X_vec[i];
      vec_t out;

      // Computation is performed in T_ACC, X is cast to T_ACC and result is implicitly cast to T
      if (gamma_vec != nullptr && beta_vec != nullptr) {
        #pragma unroll
        for (int ii=0; ii < vec_size; ii++){
          out.val[ii] = static_cast<T_ACC>(gamma_vec[i].val[ii]) * (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean))
            + static_cast<T_ACC>(beta_vec[i].val[ii]);
        }
      } else if (gamma_vec != nullptr) {
        #pragma unroll
        for (int ii=0; ii < vec_size; ii++){
          out.val[ii] = static_cast<T_ACC>(gamma_vec[i].val[ii]) * (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean));
        }
      } else if (beta_vec != nullptr) {
        #pragma unroll
        for (int ii=0; ii < vec_size; ii++){
          out.val[ii] = (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean)) + static_cast<T_ACC>(beta_vec[i].val[ii]);
        }
      } else {
        #pragma unroll
        for (int ii=0; ii < vec_size; ii++){
          out.val[ii] = rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean);
        }
      }
      Y_vec[i] = out;
    }
    if (thrx == 0) {
      mean[i1] = wd.mean;
      rstd[i1] = rstd_val;
    }
```

上面我们分析完了layernorm的vec kernel，下面分析无法向量化的情况

```text
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  RowwiseMomentsCUDAKernel<T, T_ACC>
      <<<M, cuda_utils::kCUDABlockReduceNumThreads, 0, cuda_stream>>>(
          N, eps, X_data, mean_data, rstd_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  LayerNormForwardCUDAKernel<T, T_ACC><<<M, kCUDANumThreads, 0, cuda_stream>>>(
      N, X_data, mean_data, rstd_data, gamma_data, beta_data, Y_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
```

第一个碰见的函数是RowwiseMomentsCUDAKernel 其实和上面的代码没有什么不同，只是因为没有向量化了，所以按照索引直接访问就可以了。Welford BlockReduce之前也分析过，这里就不分析了

后面的LayerNormForwardCUDAKernel逻辑更简单，同样只是计算rstd\_val \* (data.val\[ii\]) - wd.mean)，此处略过。

```text
template <typename T, typename T_ACC>
__global__ void RowwiseMomentsCUDAKernel(
    int64_t N,
    T_ACC eps,
    const T* X,
    T_ACC* mean,
    T_ACC* rstd) {
  using WelfordType = WelfordData<T_ACC, int64_t>;
  using WelfordOp =
      WelfordOps<T_ACC, T_ACC, int64_t, thrust::pair<T_ACC, T_ACC>>;

  __shared__
      typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::
          type val_shared[C10_WARP_SIZE];
  WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);

  const int64_t i = blockIdx.x;
  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);

  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    val = welford_op.reduce(val, static_cast<T_ACC>(X[index]), index);
  }
  val = cuda_utils::BlockReduce(
      val,
      welford_op,
      /*identity_element=*/WelfordType(0, 0, 0, 0),
      val_shared_ptr);

  if (threadIdx.x == 0) {
    T_ACC m1;
    T_ACC m2;
    thrust::tie(m2, m1) = welford_op.project(val);
    mean[i] = m1;
    rstd[i] = c10::cuda::compat::rsqrt(m2 + eps);
  }
}
```