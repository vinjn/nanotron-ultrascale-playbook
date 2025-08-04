# pytorch 源码分析（2）gpu_kernel （1）

**Author:** 阿嚏

**Date:** 2025-06-19

**Link:** https://zhuanlan.zhihu.com/p/1916873712545792503

上一篇文章[pytorch源码分析（1）TensorIterator](https://zhuanlan.zhihu.com/p/1916807898912257830)中，我们着重分析了gpu\_kernel的参数[TensorIteratorBase](https://zhida.zhihu.com/search?content_id=259039230&content_type=Article&match_order=1&q=TensorIteratorBase&zhida_source=entity)，至于其还有第二个参数，完全可以简单理解为就是个函数指针，所以现在可以开始分析gpu\_kernel部分了。

gpu\_kernel函数body其实毕竟小，核心是去调用[gpu\_kernel\_impl](https://zhida.zhihu.com/search?content_id=259039230&content_type=Article&match_order=1&q=gpu_kernel_impl&zhida_source=entity)做真正的工作，从代码简单分析，应该是如果数据过大，32位索引无法表示，就拆分一下。

```text
template <typename func_t>
void gpu_kernel(TensorIteratorBase& iter, const func_t& f) {

  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(
      iter.device(arg).is_cuda(),
      "argument ", arg, ": expected a CUDA device but found ", iter.device(arg));
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      gpu_kernel(sub_iter, f);
    }
    return;
  }

  gpu_kernel_impl(iter, f);
}
```

这里涉及到一个iter的numel，实现如下：这个shape\_上一篇分析过，是广播后的shape\_ 如果一个元素也没有，自然就直接跳过了。

```text
int64_t TensorIteratorBase::numel() const {
  int64_t numel = 1;
  for (int64_t size : shape_) {
    numel *= size;
  }
  return numel;
}
```

gpu\_kernel\_impl的函数体比较长，我们就拆分来看：

```text
template <typename func_t>
void gpu_kernel_impl(TensorIteratorBase& iter, const func_t& f) {
  if (!needs_dynamic_casting<func_t>::check(iter)) {
    return gpu_kernel_impl_nocast(iter, f);
  }
```

这里是检查传入的函数的返回类型和Output的返回类型是否一直，要是一致，说明不需要做类型转换，就简单处理：

```text
template <typename func_t>
void gpu_kernel_impl_nocast(TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  constexpr int ntensors = traits::arity + 1;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);
  TORCH_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));

  at::detail::Array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();

  bool contiguous = iter.is_contiguous();

  if (contiguous) {
    return launch_vectorized_kernel(numel, f, data);
  }
  auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
  constexpr int unroll_factor = sizeof(arg0_t) >= 4 ? 2 : 4;
  launch_legacy_kernel<128, unroll_factor>(numel, [=] GPU_LAMBDA(int idx) {
    auto offsets = offset_calc.get(idx);
    arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
    *out = invoke(f, &data.data[1], &offsets.data[1], 1);
  });
}
```

traits 这种用法个人是比较反感的，其实就是参数，只不过是类型参数，arity是这个函数的输入参数的数量，+1是因为还有Output。接着将所有tensor的data指针存储起来，然后开始判断是否连续，如果连续的话，使用向量化kernel。

```text
  int ndim() const {
    return static_cast<int>(shape_.size());
  }

  bool has_contiguous_first_dim() const {
    if (ndim() == 0) {
      return true;
    }

    int num_tensors = ntensors();
    for (const auto i : c10::irange(num_tensors)) {
      if (strides(i)[0] != element_size(i)) {
        return false;
      }
    }
    return true;
  }

bool TensorIteratorBase::is_contiguous() const {
  if (numel() == 1) {
    return true;
  }
  if (ndim() != 1) {
    return false;
  }
  return has_contiguous_first_dim();
}
```

可以看出来，其对连续定义的要求比较高，要求tensor都是1维的，并且数据存储是连续密集的, 可是平时使用的tensor，往往是高维的居多，这是因为，tensor之前有过合并维度的操作，可以参考上文[pytorch源码分析（1）Tensor](https://zhuanlan.zhihu.com/p/1916807898912257830)，如果是一个正常的默认tensor，大概率是连续的。

这里先假设数据连续。

```text
// this function assume trivial 1d and no dynamic casting
template <typename func_t, typename array_t>
static inline void launch_vectorized_kernel(
    int64_t N,
    const func_t& f,
    array_t data) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  using traits = function_traits<func_t>;
  int64_t grid = (N + block_work_size() - 1) / block_work_size();
  auto stream = at::cuda::getCurrentCUDAStream();
  int vec_size = memory::can_vectorize_up_to<func_t>(data);

  switch (vec_size) {
    case 4:
      vectorized_elementwise_kernel<4, func_t, array_t>
          <<<grid, num_threads(), 0, stream>>>(N, f, data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 2:
      vectorized_elementwise_kernel<2, func_t, array_t>
          <<<grid, num_threads(), 0, stream>>>(N, f, data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 1: {
      auto input_calc = TrivialOffsetCalculator<traits::arity>();
      auto output_calc = TrivialOffsetCalculator<1>();
      auto loader = memory::LoadWithoutCast();
      auto storer = memory::StoreWithoutCast();
      unrolled_elementwise_kernel<func_t, array_t>
          <<<grid, num_threads(), 0, stream>>>(
              N, f, data, input_calc, output_calc, loader, storer);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(false, "Unexpected vectorization size");
  }
}
```

N是所有的待处理数据数量，这个N是对于单个input tensor来说的，由于广播，每个tensor的大小都一样。block\_work\_size是一个block可以处理的数据数量，其由thread\_work\_size每个线程处理的数据数量乘以num\_threads线程数量得来，这里我们假设thread\_work\_size是4，num\_threads是128，那么block\_work\_size就是512，grid是处理总的N,需要多少个block。假设N是1025（故意比1024多个1），那么grid就是3，代表3个block，前两个block都处理512个数据，而最后一个block只处理一个数据。

  

这里有一个向量化读写，也就是将多个数据合并起来进行读写，这样可以增快访存。

在vectorized\_elementwise\_kernel中也有两种情况，也就是当前的block是否会进行一个完整的数据处理，我们关注完整部分。

```text
template <int vec_size, typename func_t, typename array_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void vectorized_elementwise_kernel(int N, func_t f, array_t data) {
  using traits = function_traits<func_t>;
  int remaining = N - block_work_size() * blockIdx.x;

  if (remaining < block_work_size()) { // if this block handles the reminder,
                                       // just do a naive unrolled loop
    auto input_calc = TrivialOffsetCalculator<traits::arity>();
    auto output_calc = TrivialOffsetCalculator<1>();
    auto loader = memory::LoadWithoutCast();
    auto storer = memory::StoreWithoutCast();
    auto policy = memory::policies::unroll<
        array_t,
        decltype(input_calc),
        decltype(output_calc),
        memory::LoadWithoutCast,
        memory::StoreWithoutCast>(
        data, remaining, input_calc, output_calc, loader, storer);
    elementwise_kernel_helper(f, policy);
  } else { // if this block has a full `block_work_size` data to handle, use
           // vectorized memory access
    elementwise_kernel_helper(
        f, memory::policies::vectorized<vec_size, array_t>(data));
  }
}
```

这边逻辑比较简单了，即读取数据，处理数据，存储数据，apply就是调用gpu\_kernel最开始传入的那个函数，因为函数参数是变长的，所以这么设计。

```text
template<typename func_t, typename policy_t>
__device__ inline void elementwise_kernel_helper(func_t f, policy_t policy) {
  using traits = function_traits<func_t>;
  using return_t = typename traits::result_type;
  using args_t = typename traits::ArgsTuple;

  int idx = blockIdx.x;

  return_t results[thread_work_size()];
  args_t args[thread_work_size()];

  // load
  policy.load(args, idx);

  // compute
  #pragma unroll
  for (int i = 0; i < thread_work_size(); i++) {
    if (policy.check_inbounds(i)) {
      results[i] = c10::guts::apply(f, args[i]);
    }
  }

  // store
  policy.store(results, idx);
}
```

那么向量化就隐含在这个 policy中了，这一部分实现主要在aten/src/ATen/native/cuda/MemoryAccess.cuh中，方便讲解，采取自底向上的路径讲解

先看最底层是如何向量化的，其实就是多个数据数组组装成一个[aligned\_vector](https://zhida.zhihu.com/search?content_id=259039230&content_type=Article&match_order=1&q=aligned_vector&zhida_source=entity)，然后转换类型，一次性将这些数据读取过来。

```text
template <int vec_size, typename scalar_t>
__device__ aligned_vector<scalar_t, vec_size> load_vector(const scalar_t *base_ptr, uint32_t offset) {
  using vec_t = aligned_vector<scalar_t, vec_size>;
  auto *from = reinterpret_cast<const vec_t *>(base_ptr);
  return from[offset];
}
```

每个线程比如需要处理4条数据（thread\_work\_size=4）如果向量化一次读取2（vec\_size），那么就需要循环两次，注意这里的to，是个函数，其实是为了向上面定义的args赋值，args属于局部变量，也就是寄存器上的值

在loop\_size外循环中，每次load vec\_size数据，然后再把这些值依次给到对应的位置

```text
  static constexpr int loop_size = thread_work_size() / vec_size;
  

template<typename accessor_t, typename scalar_t>
  __device__ inline void load_single_arg(accessor_t to, scalar_t *from) {
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < loop_size; i++) {
      int index = thread_idx + i * num_threads();
      auto v = load_vector<vec_size>(from, index);
      #pragma unroll
      for (int j = 0; j < vec_size; j++) {
        to(vec_size * i + j) = v.val[j];
      }
    }
  }

template<int arg_index>
struct vectorized_load_helper {
  template <typename args_t, typename policy_t>
  static __device__ void apply(policy_t &self, args_t *args, int idx) {
    using arg_t = std::tuple_element_t<arg_index, args_t>;
    // `data` hold the data_ptr for tensors [output, input0, input1, ...], so we
    // need a +1 offset to get the input
    auto ptr = reinterpret_cast<arg_t *>(self.data[arg_index + 1]) + block_work_size() * idx;
    auto args_accessor = [&args] __device__ (int thread_unroll_idx) -> arg_t & { return std::get<arg_index>(args[thread_unroll_idx]); };
    self.load_single_arg(args_accessor, ptr);
  }
};

  template<typename args_t>
  __device__ inline void load(args_t *args, int idx) {
    constexpr int arity = std::tuple_size<args_t>::value;
    detail::static_unroll<detail::vectorized_load_helper, arity>::with_args(*this, args, idx);
  }
```

static\_unroll 就是借助模板循环展开，对于每个Input, 都执行一下vectorized\_load\_helper，只是arg\_index不一样，用来选取不同的input tensor

关于各种idx，有必要总结一下，以128个线程，每个线程处理4个值，也就是每个Block处理512个值，假如总共处理32768个值，那么也就是64个block工作。

假设vec\_size是2，也就是向量化一次性读取两个值，那么loop\_size就是2，代表着一个线程为了完成处理4个值的目标，需要处理2个vec\_size

先看外循环坐标 int index = thread\_idx + i \* num\_threads();

对于0号线程，其index 是0，128

对于1号线程，其index是1，129

...

对于128号线程，其index是127，255

那么索引最多到255，一个block不是处理256个数据吗，注意load\_vector函数，将类型转为了const vec\_t \*，之前如果数组是from\[512\]转成后就是from\[256\]了

所以，对于0号线程，其真实处理的数据，是0，1，256，257

对于1号线程，其真实处理的数据，是2，3，258，259

...

对于128号线程，其真实处理的数据，是254，255，510，511

同时注意到，在loop\_size循环中，第一轮线程处理的数据，是0-255，第二轮处理的数据，是256-511，同样也为连续的，这样设计可以优化访存。

  

我们注意到，一个vec读取到v中后，还有一层内循环，将数据再次读入到to中，这个内循环比较好理解，就是将向量化的数据再拆开，依次填入到其该去的位置。

但是这个to，可能没那么好理解，是一个lambda函数，

```text
    auto args_accessor = [&args] __device__ (int thread_unroll_idx) -> arg_t & { return std::get<arg_index>(args[thread_unroll_idx]); };
```

回顾一下，args的定义为 args\_t args\[thread\_work\_size()\];

而类型为 using args\_t = typename traits::ArgsTuple;

这个traits::ArgsTuple就是一个function参数的元组，对于float add函数，其实就是tuple(float,float)这样

那么，其中的arg\_index，就是来选择，是前一个float,还是后一个float。也就是是前一个参数，还是后一个参数。arg\_index 这个值由static\_unroll设定，这就是一个模板展开。

注意到，args定义的时候，还有一个thread\_work\_size()，这个值我们假设的是4，vec\_size \* i + j其实也是0-3，也就是thread\_work\_size的索引。这样，两层索引，就可以将数据读入到args对应正确的位置上去了。

  

看一下中间的函数执行：从字面也可以看出来只是执行函数而已，不过涉及到很多c++语法,这一部分有机会再分析[C++编程系列笔记（3）——std::forward与完美转发详解](https://zhuanlan.zhihu.com/p/645328162)

```text
template <class F, class Tuple>
C10_HOST_DEVICE inline constexpr decltype(auto) apply(F&& f, Tuple&& t) {
  return std::apply(std::forward<F>(f), std::forward<Tuple>(t));
}
```

  

store基本和load相反，这里就不过多分析了

```text
  template<typename scalar_t>
  __device__ inline void store(scalar_t *from, int idx) {
    using vec_t = aligned_vector<scalar_t, vec_size>;
    scalar_t *to = reinterpret_cast<scalar_t *>(data[0]) + block_work_size() * idx;
    vec_t *to_ = reinterpret_cast<vec_t *>(to);
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < loop_size; i++) {
      int index = thread_idx + i * num_threads();
      vec_t v;
      for (int j = 0; j < vec_size; j++) {
        v.val[j] = from[vec_size * i + j];
      }
      to_[index] = v;
    }
  }
```