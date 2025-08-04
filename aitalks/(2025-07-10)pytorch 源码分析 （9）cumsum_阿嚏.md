# pytorch 源码分析 （9）cumsum

**Author:** 阿嚏

**Date:** 2025-07-10

**Link:** https://zhuanlan.zhihu.com/p/1926651261836563036

算子大体可以分为两种，一种是gpu\_kernel那种element wise的，也就是每个元素的计算互不干扰, 一种是非element wise的（我在说什么废话），这种算子往往比较复杂，像softmax需要reduce操作，算是其中比较基础的，还有比较难的，例如sort，排序算法作为最经典的算法在gpu上也有许多优化，不过本文暂时不讲这位大将，本文讲的是，前缀和算法，又被称为scan。

对于cpu来说，写一个前缀和程序大家都比较熟悉了，一个for循环搞定，但是gpu这种并行化场景，想写的性能高就得复杂点了。

在pytorch中,这个算子叫[cumsum](https://zhida.zhihu.com/search?content_id=260166326&content_type=Article&match_order=1&q=cumsum&zhida_source=entity): 使用方式加个dim就行了。

```text
>>> a
tensor([1, 2, 1, 2, 1, 2])
>>> a.cumsum(0)
tensor([1, 3, 4, 6, 7, 9])
```

pytorch 的dispatch机制先跳过，先看一下最终的工作部分

```text
void launch_cumsum_cuda_kernel(const TensorBase& result, const TensorBase& self, int64_t dim) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      self.scalar_type(), "cumsum_cuda",
      [&]() {
        scalar_t init = 0;
        scan_dim<scalar_t>(
            self,
            result,
            dim,
            init,
            std::plus<scalar_t>());
      });
}
```

pytorch很多底层的计算，都是调用的[cub库](https://zhida.zhihu.com/search?content_id=260166326&content_type=Article&match_order=1&q=cub%E5%BA%93&zhida_source=entity)。不过对于llm中，很多情况都是计算(batch,dim\_size), 也就是scan\_innermost\_dim这种情况，我们先看这个。

```text
template<typename scalar_t, typename BinaryFunction>
void scan_dim(const TensorBase& self, const TensorBase& result,
     int64_t dim, scalar_t init, BinaryFunction binary_op) {
  int ndim = self.dim();
  auto self_ = self.expect_contiguous();
  TORCH_INTERNAL_ASSERT(result.is_contiguous());

  if (self.numel() == self.size(dim)) {
    cuda::cub::inclusive_scan(self_->const_data_ptr<scalar_t>(), result.mutable_data_ptr<scalar_t>(), binary_op, self.numel());
  } else if (dim == ndim - 1) {
    scan_innermost_dim<scalar_t>(*self_, result, init, binary_op);
  } else {
    scan_outer_dim<scalar_t>(*self_, result, dim, init, binary_op);
  }
}
```

scan\_innermost\_dim的主要作用就是启动tensor\_kernel\_scan\_innermost\_dim这个真正的内核。

之前碰到的大多block grid都是一维的，事实上哪怕多维问题也可以用一维处理，不过这里用到了一个两维，需要看一下get\_log\_num\_threads\_x\_inner\_scan这个函数。

```text
template <typename scalar_t, class BinaryFunction>
void scan_innermost_dim(const TensorBase& self, const TensorBase& result,
                        scalar_t init, BinaryFunction binary_op) {
  int64_t ndim = self.dim();
  // Treat all outer dimensions as a single dimension.
  int64_t row_size = self.size(ndim - 1);
  int64_t num_rows = self.numel() / row_size;

  // assuming max_num_threads per block is 512
  const uint32_t num_threads = 512;
  const uint32_t log_num_threads_x = get_log_num_threads_x_inner_scan<uint32_t>(num_rows, row_size);
  const uint32_t num_threads_x = (1 << log_num_threads_x);
  const uint32_t num_threads_y = num_threads / num_threads_x;
  dim3 threads(num_threads_x, num_threads_y);
  int64_t maxGridDim = at::cuda::getCurrentDeviceProperties()->maxGridSize[0];
  dim3 grid(std::min(maxGridDim, ceil_div(num_rows, int64_t{threads.y})));

  check_fits_in_unsigned(num_rows, "Number of rows (self.numel()/self.size(self.dim()-1))");
  check_fits_in_unsigned(row_size, "row_size");

  tensor_kernel_scan_innermost_dim<scalar_t><<<grid, threads, num_threads * 2 * sizeof(scalar_t),
                                               at::cuda::getCurrentCUDAStream()>>>(
    result.mutable_data_ptr<scalar_t>(), self.const_data_ptr<scalar_t>(),
    num_rows, row_size, log_num_threads_x, init, binary_op);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
```

这个函数的思想是，希望x-threads 和 y-threads的比值与row\_size 和num\_rows的比值差不多，而且要求线程总数为512。其实按我说想自动找到合适的block就是瞎搞，只能是多试试。

grid被设置为std::min(maxGridDim, ceil\_div(num\_rows, int64\_t{threads.y}))，也就是block y维度上一次性处理多个row

```text
template <typename integer>
constexpr inline integer get_log_num_threads_x_inner_scan(integer num_rows, integer row_size) {
  integer log_num_threads_x = 0;
  integer log_num_threads_y = 0;
  while (((integer)1 << log_num_threads_x) < row_size) {
    ++log_num_threads_x;
  }
  while (((integer)1 << log_num_threads_y) < num_rows) {
    ++log_num_threads_y;
  }
  // we want to keep the ratio between the x-threads and y-threads about the same as
  // the ratio between the row_size and num_rows, but the total number of threads in
  // a block should be about 512
  integer diff = log_num_threads_x - log_num_threads_y;
  // 9 is from log2(512)
  log_num_threads_x = ((integer)9 + diff) / (integer)2;
  // I found that in having larger log_num_threads_x can give significant speed up in some cases,
  // but detrimental in another case, so just keep the lower bound to be log2(16) == 4 to make it
  // similar to the previous implementation
  // Keeping the upper bound to be log2(512) == 9 as the maximum number of threads in a block.
  log_num_threads_x = std::min(std::max((integer)4, log_num_threads_x), (integer)9);
  return log_num_threads_x;
}
```

tensor\_kernel\_scan\_innermost\_dim 主要是处理一下smem，smem总大小为num\_threads \* 2 \* sizeof(scalar\_t)，其中thread y维度是在num\_rows维度用的，每一个row，其smem对应位置为sbuf2 + num\_threads\_x \* 2 \* threadIdx.y

```text
template <
    typename T,
    class BinaryFunction>
__global__ void tensor_kernel_scan_innermost_dim(
    T* tgt_,
    const T* src_,
    const uint32_t num_rows,
    const uint32_t row_size,
    const uint32_t log_num_threads_x,
    T init,
    BinaryFunction binary_op) {
  alignas(sizeof(double)) extern __shared__ char sbuf[];
  T* sbuf2 = reinterpret_cast<T*>(sbuf);
  const uint32_t num_threads_x = 1 << log_num_threads_x;
  T* row_buf = reinterpret_cast<T*>(sbuf2 + num_threads_x * 2 * threadIdx.y);

  tensor_kernel_scan_innermost_dim_impl<T>(
      row_buf, tgt_, src_, num_rows, row_size, log_num_threads_x, init, binary_op);
}
```

tensor\_kernel\_scan\_innermost\_dim\_impl函数是本文的核心重点，用的是[Sklansky算法](https://zhida.zhihu.com/search?content_id=260166326&content_type=Article&match_order=1&q=Sklansky%E7%AE%97%E6%B3%95&zhida_source=entity)，详细可以参考

[https://research.nvidia.com/publication/single-pass-parallel-prefix-scan-decoupled-look-back](https://link.zhihu.com/?target=https%3A//research.nvidia.com/publication/single-pass-parallel-prefix-scan-decoupled-look-back)

先看外层循环，不得不说多了一维立马不大好直观看了，block\_row 其实就是当前block负责的row。

blockDim.y对应的是num\_threads\_y ，也就是一个block负责的row数量。

举一个例子 数据是（8，32）， block(2,4)，也就是每个block负责4组数据，那么一共就有2个block，grid(2)

blockIdx.x 是grid的block的索引（我一直觉得这里是取名失败了），也就是0或者1

blockDim.y 是4，所以block\_row取值为 0 4 (一个block负责4个row)

grid 是std::min(maxGridDim, ceil\_div(num\_rows, int64\_t{threads.y}))，这是8/4=2 比maxGridDim小ok，要是不ok，就+8 +8 这样向下拓展，直到处理完所有的row

  

row = block\_row + threadIdx.y 就确定了当前线程处理的row了，对于累加初始值是0，累乘的话是1

row\_src row\_tgt 就是定位到这一行的数据指针（tensor的行优先真的省了很多事）

```text
  for (uint32_t block_row = blockIdx.x * blockDim.y;
       block_row < num_rows;
       block_row += blockDim.y * gridDim.x) {
    uint32_t row = block_row + threadIdx.y;
    T block_total = init;

    const T *row_src = src_ + row * row_size;
    T *row_tgt = tgt_ + row * row_size;
    const bool row_exists = row < num_rows;
```

大循环是循环行的，小循环自然是循环列的，这一部分是本函数重点，先说明一下，抛开y轴，每个block一次性是处理2 \* num\_threads\_x个数的，也就是将这2 \* num\_threads\_x的前缀和算出来，然后将最后一个值，也就是这些数的总和给block\_total，下次循环再累加它就好了。

打个比方，有数\[1,1,1,1,1,1,1,1\]，block(2,1) （y轴我们这边暂时不管了）

那么第一次会完成, 知道了总和为4

\[1,2,3,4,1,1,1,1\]

第二次直接将4加到第一个处理的数上，也就是

\[1,2,3,4,5,1,1,1\]

继续累加，变为

\[1,2,3,4,5,6,7,8\]，依次类推，这就是block\_total的作用。

那么，每次小迭代如何将2 \* num\_threads\_x的前缀和算出来呢？核心在于下面这个公式

uint32\_t a = ((threadIdx.x >> m) << (m + 1)) | s; // a = (threadIdx.x / s) \* (2 \* s) + s

对于线程0，a值的变化为1，2，4，8... 这个好理解，这个值就是 0 | s = s = 1<<m

而线程之间，是1 << m为一组相同值，不同组之间差值为1<<(m+1) 这是因为 idx/(1<<m)相同的这些值，有1 << m个。举个例子：

对于m=0

1 3 5 7 9 11 13 15

对于m=1

2 2 6 6 10 10 14 14

对于m=2

4 4 4 4 12 12 12 12

那么ti，就是一组之间的所有值，例如4 4 4 4，ti就对应4 5 6 7，这一组数的值管谁要呢？管a-1要，也就是上一组的结果。这是一个数学归纳法，大家只能自己体会一下了，反正就是这么神奇。。。

为了方便理解举一个例子：

假设有数\[0,1,2,3,4,5,6,7\] （数不同好理解些） block(4,1)

首先，各个线程将数据读入smem中，其中线程0读取1 5 线程2读取2 6，依次类推

然后4个线程，

\[0, 1, 2, 3, 4, 5, 6, 7\]

\[0,(0,1), 2, (2,3), 4, (4,5), 6, (6,7)\]

\[0,(0,1),(0,1,2),(0,1,2,3), 4 ,(4,5), (4,5,6), (4,5,6,7)\]

\[0,(0,1),(0,1,2),(0,1,2,3),(0,1,2,3,4),(0,1,2,3,4,5),(0,1,2,3,4,5,6),(0,1,2,3,4,5,6,7)\]

  

```text
    for (uint32_t block_col = 0; block_col < row_size; block_col += 2 * num_threads_x) {
      // Load data into shared memory (two values per thread).
      uint32_t col1 = block_col + threadIdx.x;
      uint32_t col2 = block_col + num_threads_x + threadIdx.x;
      if (row_exists) {
        if (col1 < row_size) {
          row_buf[threadIdx.x] = row_src[col1];
        } else {
          row_buf[threadIdx.x] = init;
        }

        if (col2 < row_size) {
          row_buf[num_threads_x + threadIdx.x] = row_src[col2];
        } else {
          row_buf[num_threads_x + threadIdx.x] = init;
        }

        // Add the total value of all previous blocks to the first value of this block.
        if (threadIdx.x == 0) {
          row_buf[0] = binary_op(row_buf[0], block_total);
        }
      }
      __syncthreads();

      // Parallel reduction with Sklansky method. The diagram can be seen on this paper:
      // https://research.nvidia.com/publication/single-pass-parallel-prefix-scan-decoupled-look-back
      for (uint32_t m = 0; m <= log_num_threads_x; ++m) {
        if (row_exists) {
          uint32_t s = 1 << m; // s = 2 ^ m
          uint32_t a = ((threadIdx.x >> m) << (m + 1)) | s; // a = (threadIdx.x / s) * (2 * s) + s
          uint32_t ti = a + (threadIdx.x % s);
          uint32_t si = a - 1;
          row_buf[ti] = binary_op(row_buf[ti], row_buf[si]);
        }
        __syncthreads();
      }

      // Write back to output.
      if (row_exists) {
        if (col1 < row_size) row_tgt[col1] = row_buf[threadIdx.x];
        if (col2 < row_size) row_tgt[col2] = row_buf[num_threads_x + threadIdx.x];
      }
      block_total = row_buf[2 * num_threads_x - 1];
      __syncthreads();
    }
```