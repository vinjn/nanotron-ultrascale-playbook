# pytorch 源码分析（3）gpu_kernel （2）

**Author:** 阿嚏

**Date:** 2025-06-19

**Link:** https://zhuanlan.zhihu.com/p/1917874917367611994

上篇文章[pytorch 源码分析（2）gpu\_kernel （1）](https://zhuanlan.zhihu.com/p/1916873712545792503)分析了output与function返回值类型相同，而且是连续的情况，这篇文章继续分析其他情况。

首先是数据连续但是返回值不同的时候，最大的区别就是增加了类型转换。

值得注意的是，之前类型相同，取值只需要data\[offset\]就可以了，但是这里需要通过[strides](https://zhida.zhihu.com/search?content_id=259128059&content_type=Article&match_order=1&q=strides&zhida_source=entity)进行地址具体的计算，毕竟类型不同，其余和之前没有太大区别。

```text
    at::detail::Array<ScalarType, ntensors> dtypes;
    auto inner_strides = iter.get_inner_strides();
    at::detail::Array<int, ntensors> strides;
    for (int i = 0; i < ntensors; i++) {
      dtypes[i] = iter.dtype(i);
      strides[i] = inner_strides[i];
    }
    launch_legacy_kernel<512, 1>(numel, [=]GPU_LAMBDA(int idx) {
      void* out = data[0] + strides[0] * idx;
      arg0_t result = invoke(f, &data.data[1], &strides.data[1], &dtypes.data[1], idx);
      c10::cast_and_store<arg0_t>(dtypes[0], out, result);
    });
```

但是如果数据不连续，事情就会变得复杂一些，先看一下整体代码

```text
    at::detail::Array<ScalarType, ntensors> dtypes;
    for (int i = 0; i < ntensors; i++) {
      dtypes[i] = iter.dtype(i);
    }
    auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
    launch_legacy_kernel<128, 4>(numel, [=] GPU_LAMBDA(int idx) {
      auto offsets = offset_calc.get(idx);
      void* out = data[0] + offsets[0];
      arg0_t result = invoke(f, &data.data[1], &offsets.data[1], &dtypes.data[1], 1);
      c10::cast_and_store<arg0_t>(dtypes[0], out, result);
    });
```

疑难点就在于这个 make\_offset\_calculator，在分析offset之前，先看一下idx的使用，基本和上一篇文章分析的一致

blockIdx.x是外层idx，每次跨度为128\*4=512。

threadIdx.x是内层idx，每次跨度为128

如果两者都为0，那么处理的就是0，128，256，384这四个值，注意这里就没有向量化了。

```text
template <int nt, int vt, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel(int N, func_t f) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int idx = nv * blockIdx.x + tid;
#pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      f(idx);
      idx += nt;
    }
  }
}
```

下面开始分析offset，其实其大体目标，就是根据类型步长，计算合适的偏移。

```text
template<int N, bool signed_strides = false>
static OffsetCalculator<N, uint32_t, signed_strides> make_offset_calculator(const at::TensorIteratorBase& iter) {
  TORCH_INTERNAL_ASSERT(N <= iter.ntensors());
  std::array<const int64_t*, N> strides;
  for (int i = 0; i < N; i++) {
    strides[i] = iter.strides(i).data();
  }
  return OffsetCalculator<N, uint32_t, signed_strides>(iter.ndim(), iter.shape().data(), strides.data());
}
```

一开始主要就是为了传递strides，这一部分没什么，会调用OffsetCalculator另一个构造函数

注意 strides 和 strides\_两维表示正好是反的，不知道为什么要这样。

```text
  // if element_sizes is nullptr, then the strides will be in bytes, otherwise
  // the strides will be in # of elements.
  OffsetCalculator(int dims, const int64_t* sizes, const int64_t* const* strides, const int64_t* element_sizes=nullptr) : dims(dims) {
    TORCH_CHECK(dims <= MAX_DIMS, "tensor has too many (>", MAX_DIMS, ") dims");
    for (int i=0; i < dims; i++){
      sizes_[i] = at::cuda::detail::IntDivider<index_t>(sizes[i]);
      for (int arg = 0; arg < NARGS; arg++) {
        int64_t element_size = (element_sizes == nullptr ? 1LL : element_sizes[arg]);
        strides_[i][arg] = strides[arg][i] / element_size;
      }
    }
  }
```

关键是这个get方法。

```text
  C10_HOST_DEVICE offset_type get(index_t linear_idx) const {
    offset_type offsets;
    #pragma unroll
    for (int arg = 0; arg < NARGS; arg++) {
      offsets[arg] = 0;
    }

    #pragma unroll
    for (int dim = 0; dim < MAX_DIMS; ++dim) {
      if (dim == dims) {
        break;
      }
      auto divmod = sizes_[dim].divmod(linear_idx);
      linear_idx = divmod.div;

      #pragma unroll
      for (int arg = 0; arg < NARGS; arg++) {
        offsets[arg] += divmod.mod * strides_[dim][arg];
      }

    }
    return offsets;
  }
```

打个比方，如果一个shape(3,4), strides (1,3)的tensor, 寻找idx=7的偏移。

linear\_idx =7

dim = 0 7/3 = 2 7%3 =1

linear\_idx = 2 offsets += 1\*1

dim=1 2/4 = 0 2%4 = 2

linear\_idx = 0 offset += 2\*3 = 1+6 = 7

其实这就是看坐标（1，2）所在的值，为7

```text
0 3 6 9
1 4 7 10
2 5 8 11
```

  

如果strides是(4,1) idx=7

linear\_idx =7

dim = 0 7/3 = 2 7%3 =1

linear\_idx = 2 offsets += 1\*4

dim=1 2/4 = 0 2%4 = 2

linear\_idx = 0 offset += 2\*1 = 4+2=6

同样是看坐标（1，2）的值，这个值为6

```text
0 1 2 3
4 5 6 7
8 9 10 11
```

这里的idx，是给block thread分配任务的逻辑坐标，然后再根据tensor真实的物理分布，转为物理坐标，进行读取数据计算等操作。从上面的例子也可以看出，这个idx，是对于列优先矩阵而言的，如果是列优先矩阵，idx连续，那么存储也会连续，但是对于行优先矩阵就不是这样了。

所以要在一开始时，让tensor步长小的维度排在前面，这样同一个block的不同线程访问的数据是连续的，有利于访存。