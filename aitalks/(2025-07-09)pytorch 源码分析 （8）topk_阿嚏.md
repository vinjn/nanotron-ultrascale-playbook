# pytorch 源码分析 （8）topk

**Author:** 阿嚏

**Date:** 2025-07-09

**Link:** https://zhuanlan.zhihu.com/p/1926282497328650198

随着llm的兴起，[topk算子](https://zhida.zhihu.com/search?content_id=260100566&content_type=Article&match_order=1&q=topk%E7%AE%97%E5%AD%90&zhida_source=entity)使用开始变得频繁。

topk即找到最大/最小的k个值，在pytorch中，用法为:

```text
torch.topk(input, k, dim=None, largest=True, sorted=True)
```

他是如何进行工作的？其实分为很多种场景，有的时候干脆直接排序，然后再选出k个值来就好，我们就假设这是一个（batch,dim\_size）问题，有sbtopk和mbtopk两种情况。

先看sbtopk。其中[grid](https://zhida.zhihu.com/search?content_id=260100566&content_type=Article&match_order=1&q=grid&zhida_source=entity)是batch, [block](https://zhida.zhihu.com/search?content_id=260100566&content_type=Article&match_order=1&q=block&zhida_source=entity)是min(dim\_size,1024),这是一种典型的任务划分方式。其核心函数为[gatherTopK](https://zhida.zhihu.com/search?content_id=260100566&content_type=Article&match_order=1&q=gatherTopK&zhida_source=entity) 其中用到了很多[基数排序](https://zhida.zhihu.com/search?content_id=260100566&content_type=Article&match_order=1&q=%E5%9F%BA%E6%95%B0%E6%8E%92%E5%BA%8F&zhida_source=entity)的算法。一般基数排序是给整数用的，这里给浮点数，把浮点数看作整数也一样，只是对于符号位需要转化一下，可以参考：

```text
template <>
struct TopKTypeConfig<float> {
  typedef uint32_t RadixType;

  static inline __device__ RadixType convert(float v) {
    RadixType x = __float_as_int(v);
    RadixType mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;

    return (v == v) ? (x ^ mask) : 0xffffffff;
  }

  static inline __device__ float deconvert(RadixType v) {
    RadixType mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;

    return __int_as_float(v ^ mask);
  }
};
```

最重要的是[radixSelect](https://zhida.zhihu.com/search?content_id=260100566&content_type=Article&match_order=1&q=radixSelect&zhida_source=entity)这个函数，这个函数十分复杂，先看开头，关于RADIX\_SIZE是这样的，如果只是按每个bit排序，只有俩值，太多次迭代了，所以两个bit组成一组，有四个值。

```text
// Over what radix we are selecting values
constexpr int RADIX_BITS = 2; // digits are base-(2 ^ RADIX_BITS)
constexpr int RADIX_SIZE = 4; // 2 ^ RADIX_BITS
constexpr int RADIX_MASK = (RADIX_SIZE - 1);
```

代码就是从高位到低位，运行countRadixUsingMask函数。基数排序不是从低到高吗？为什么这里从高到低？因为这里只是想选择top k，如果某一位，一些数比另一些数高，那么这些数肯定比另一些数值大，这样可以快速去除不需要的数值。

```text
// Returns the top-Kth element found in the data using radix selection
template <typename scalar_t, typename bitwise_t, typename index_t>
__device__ void radixSelect(
    const scalar_t* data,
    index_t k,
    bool largest,
    index_t sliceSize,
    index_t withinSliceStride,
    int* smem,
    scalar_t* topK) {
  // Per-thread buckets into which we accumulate digit counts in our
  // radix
  int counts[RADIX_SIZE];

  // We only consider elements x such that (x & desiredMask) == desired
  // Initially, we consider all elements of the array, so the above
  // statement is true regardless of input.
  bitwise_t desired = 0;
  bitwise_t desiredMask = 0;

  // We are looking for the top kToFind-th element when iterating over
  // digits; this count gets reduced by elimination when counting
  // successive digits
  int kToFind = k;

  // We start at the most significant digit in our radix, scanning
  // through to the least significant digit
  for (int digitPos = sizeof(scalar_t) * 8 - RADIX_BITS; digitPos >= 0;
       digitPos -= RADIX_BITS) {
    // Count radix distribution for the current position and reduce
    // across all threads
    countRadixUsingMask<
        scalar_t,
        bitwise_t,
        index_t,
        int,
        RADIX_SIZE,
        RADIX_BITS>(
        counts,
        smem,
        desired,
        desiredMask,
        digitPos,
        sliceSize,
        withinSliceStride,
        data);
```

在进行下面代码分析前，先看一下countRadixUsingMask做了什么。其实这个函数的主要功能就是统计，也就是counts\[RadixSize\]，并且坐了一下高效的reduce，将所有线程的统计值合并到一起，我们看下它有哪些功能：

其实第一个学习点就在于多线程初始化smem：

```text
#pragma unroll
  for (int i = 0; i < RadixSize; ++i) {
    counts[i] = 0;
  }

  if (threadIdx.x < RadixSize) {
    smem[threadIdx.x] = 0;
  }
  __syncthreads();
```

下面出现了很多[WARP\_BALLOT](https://zhida.zhihu.com/search?content_id=260100566&content_type=Article&match_order=1&q=WARP_BALLOT&zhida_source=entity)，这个是一个1bit级别的，warp内reduce。为什么这么说，predicate只分是不是0，不是0的会在最后返回的一个mask中为1，如果再配合上\_\_popc这种统计一个值中有多少个1的函数，就相当于sum reduce了

```text
__device__ __forceinline__ unsigned int WARP_BALLOT(int predicate, unsigned int mask = 0xffffffff)
{
#if !defined(USE_ROCM)
    return __ballot_sync(mask, predicate);
#else
    return __ballot(predicate);
#endif
}
```

getLaneId是一个线程在其所在warp中的id。我们整体看一下这个函数，发现做的就是统计这个counts，并且还有reduce。

但是我们注意到，对于值还有这么一个要求 bool hasVal = ((val & desiredMask) == desired)

这就涉及到radixSelect下面的代码。

```text
template <
    typename scalar_t,
    typename bitwise_t,
    typename index_t,
    typename CountType,
    int RadixSize,
    int RadixBits>
__device__ void countRadixUsingMask(
    CountType counts[RadixSize],
    CountType* smem,
    bitwise_t desired,
    bitwise_t desiredMask,
    int radixDigitPos,
    index_t sliceSize,
    index_t withinSliceStride,
    const scalar_t* data) {
  // Clear out per-thread counts from a previous round
#pragma unroll
  for (int i = 0; i < RadixSize; ++i) {
    counts[i] = 0;
  }

  if (threadIdx.x < RadixSize) {
    smem[threadIdx.x] = 0;
  }
  __syncthreads();

  // Scan over all the data. Upon a read, the warp will accumulate
  // counts per each digit in the radix using warp voting.
#if !defined(USE_ROCM)
  // Must be called outside of loop to ensure all threads participate
  unsigned mask = WARP_BALLOT(threadIdx.x < sliceSize);
#endif
  for (index_t i = threadIdx.x; i < sliceSize;) {
    bitwise_t val =
        TopKTypeConfig<scalar_t>::convert(doLdg(&data[i * withinSliceStride]));

    bool hasVal = ((val & desiredMask) == desired);
    bitwise_t digitInRadix = at::cuda::Bitfield<bitwise_t>::getBitfield(
        val, radixDigitPos, RadixBits);

#pragma unroll
    for (uint32_t j = 0; j < RadixSize; ++j) {
      bool vote = hasVal && (digitInRadix == j);
#if defined(USE_ROCM)
      counts[j] += __popcll(WARP_BALLOT(vote));
#else
      counts[j] += __popc(WARP_BALLOT(vote, mask));
#endif
    }
    i += blockDim.x;
#if !defined(USE_ROCM)
    mask = WARP_BALLOT(i < sliceSize, mask);
#endif
  }

  // Now, for each warp, sum values
  if (at::cuda::getLaneId() == 0) {
#pragma unroll
    for (uint32_t i = 0; i < RadixSize; ++i) {
      gpuAtomicAddNoReturn(&smem[i], counts[i]);
    }
  }

  __syncthreads();

  // For each thread, read in the total counts
#pragma unroll
  for (uint32_t i = 0; i < RadixSize; ++i) {
    counts[i] = smem[i];
  }

  __syncthreads();
}
```

radixSelect 中定义了两个匿名函数[found\_unique](https://zhida.zhihu.com/search?content_id=260100566&content_type=Article&match_order=1&q=found_unique&zhida_source=entity)和[found\_non\_unique](https://zhida.zhihu.com/search?content_id=260100566&content_type=Article&match_order=1&q=found_non_unique&zhida_source=entity)，但是在看这两个函数之前，还得先看一下findPattern这个函数，这个函数的逻辑很简单，就是所有线程一起开工，找到一个满足((v & desired) == desiredMask)的唯一值。值得注意的是，要求所有线程同时工作，这个应该是为了同步。

```text
template <typename scalar_t, typename bitwise_t, typename index_t>
__device__ scalar_t findPattern(
    scalar_t* smem,
    const scalar_t* data,
    index_t sliceSize,
    index_t withinSliceStride,
    bitwise_t desired,
    bitwise_t desiredMask) {
  if (threadIdx.x < 2) {
    smem[threadIdx.x] = static_cast<scalar_t>(0);
  }
  __syncthreads();

  // All threads participate in the loop, in order to sync on the flag
  index_t numIterations =
      round_up(sliceSize, static_cast<index_t>(blockDim.x));
  for (index_t i = threadIdx.x; i < numIterations; i += blockDim.x) {
    bool inRange = (i < sliceSize);
    scalar_t v = inRange ? doLdg(&data[i * withinSliceStride])
                         : static_cast<scalar_t>(0);

    if (inRange &&
        ((TopKTypeConfig<scalar_t>::convert(v) & desiredMask) == desired)) {
      // There should not be conflicts if we are using findPattern,
      // since the result is unique
      smem[0] = static_cast<scalar_t>(1);
      smem[1] = v; // can't use val as the flag, since it could be 0
    }

    __syncthreads();

    scalar_t found = smem[0];
    scalar_t val = smem[1];

    __syncthreads();

    // Check to see if a thread found the value
    if (found != static_cast<scalar_t>(0)) {
      // all threads return this value
      return val;
    }
  }

  // should not get here
  CUDA_KERNEL_ASSERT(false);
  return static_cast<scalar_t>(0);
}
```

下面还需要看一下setBitfield和getBitfield，可以忽略bfe bfi等指令，只看没有gpu的实现，也比较清晰。getBitfield就是将值val中，pos位置开始，长度为len的bit返回，setBitfield则是对val值，pos位置开始,len长度，将toInsert放到这个位置上去。

```text
template <>
struct Bitfield<unsigned int> {
  static __device__ __host__ __forceinline__
  unsigned int getBitfield(unsigned int val, int pos, int len) {
#if !defined(__CUDA_ARCH__)
    pos &= 0xff;
    len &= 0xff;

    unsigned int m = (1u << len) - 1u;
    return (val >> pos) & m;
#else
    unsigned int ret;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
    return ret;
#endif
  }

  static __device__ __host__ __forceinline__
  unsigned int setBitfield(unsigned int val, unsigned int toInsert, int pos, int len) {
#if !defined(__CUDA_ARCH__)
    pos &= 0xff;
    len &= 0xff;

    unsigned int m = (1u << len) - 1u;
    toInsert &= m;
    toInsert <<= pos;
    m <<= pos;

    return (val & ~m) | toInsert;
#else
    unsigned int ret;
    asm("bfi.b32 %0, %1, %2, %3, %4;" :
        "=r"(ret) : "r"(toInsert), "r"(val), "r"(pos), "r"(len));
    return ret;
#endif
  }
};
```

最后看一下匿名函数found\_unique和found\_non\_unique，解读函数的同时也需要将其应用解读，为了方便，我们只看寻找k个最大值的情况，也就是if (largest)

注意i其实是在遍历radix桶，是从大往小也就是3 2 1 0，而且这个小循环里面还有一层外循环，也就是

for (int digitPos = sizeof(scalar\_t) \* 8 - RADIX\_BITS; digitPos >= 0;  
digitPos -= RADIX\_BITS)

也就是一开始从高位，高bit遍历，这第一个count，就比其他所有数大，当count<kToFind,那么kToFind -= count就是新的需要寻找的k，直到count >= kToFind，这就说明需要寻找的k，就在这些count里面。

这时候found\_non\_unique做了两件事，第一件事是将desired设置上i，因为数值在这个桶中，它的对应位置一定是i，第二件事是对应的desiredMask也设置上，表示要去寻找这个位置的值，是不是i，这时候，数值范围就被确定了。

而found\_unique则是将这个最终的值找出来，必须要满足count == 1 && kToFind == 1，为什么会这样，因为如果count 不等于1，就说明目前确定的bit位，并不足以将这个数找出来，还需要继续遍历，而kToFind 不等于1，就说明还需要继续执行kToFind -= count，最终一定会找到一个值。为什么？因为随着不断确定bit位，count值是会越来越小的，从某种角度来说，它就是前缀和，最终一定有一个1。

```text
    auto found_unique = [&](int i, int count) -> bool {
      /* All threads have the same value in counts here, so all */
      /* threads will return from the function. */
      if (count == 1 && kToFind == 1) {
        /* There is a unique answer. */
        desired = at::cuda::Bitfield<bitwise_t>::setBitfield(
            desired, i, digitPos, RADIX_BITS);
        desiredMask = at::cuda::Bitfield<bitwise_t>::setBitfield(
            desiredMask, RADIX_MASK, digitPos, RADIX_BITS);

        /* The answer is now the unique element v such that: */
        /* (v & desiredMask) == desired */
        /* However, we do not yet know what the actual element is. We */
        /* need to perform a search through the data to find the */
        /* element that matches this pattern. */
        *topK = findPattern<scalar_t, bitwise_t, index_t>(
            (scalar_t*)smem,
            data,
            sliceSize,
            withinSliceStride,
            desired,
            desiredMask);
        return true;
      }
      return false;
    };
    auto found_non_unique = [&](int i, int count) -> bool {
      if (count >= kToFind) {
        desired =
            at::cuda::Bitfield<bitwise_t>::setBitfield(
                desired, i, digitPos, RADIX_BITS);
        desiredMask = at::cuda::Bitfield<bitwise_t>::setBitfield(
            desiredMask, RADIX_MASK, digitPos, RADIX_BITS);

        /* The top-Kth element v must now be one such that: */
        /* (v & desiredMask == desired) */
        /* but we haven't narrowed it down; we must check the next */
        /* least-significant digit */
        return true;
      }
      kToFind -= count;
      return false; // continue the loop
    };

    if (largest) {
      // Process in descending order
#pragma unroll
      for (int i = RADIX_SIZE - 1; i >= 0; --i) {
        int count = counts[i];
        if (found_unique(i, count)) {
          return;
        }
        if (found_non_unique(i, count)) {
          break;
        }
      }
    }
```

可以看出这个算法非常巧妙，使用countRadixUsingMask，利用gpu的并行性快速统计count，然后在相当于单线程判断数值在哪个桶中，进而一步步缩小范围。

那么找到了第k个值还不算完，还需要进一步找到topk, 这就是下面的代码逻辑。

在此之前需要看一下exclusiveBinaryPrefixScan和inclusiveBinaryPrefixScan这两个函数，为什么还没完事？对于最终要返回的k个值，一个线程只能知道这个值是不是属于topk(hasTopK)，但是往k空间填数的时候，怎么知道填数的索引位置呢，就需要这俩函数了。

先看inclusiveBinaryPrefixScan，其中getLaneMaskLe是将小于等于自己laneid的线程全都置1，vote之前出现过，是将所有in为1的线程置为1，那么\_\_popc(getLaneMaskLe() & vote)就是，所有小于等于自己的线程，in也就是(hasTopK)的数量。carry则是hasTopK的总数量了。

下面对于每个warp，都将smem对应位置设置为carry，接着就是靠一个线程0，很朴素的计算carry的前缀和，记录到smem中。而且还需要将相对index转为绝对index。

```text
template <typename T, bool KillWARDependency, class BinaryFunction>
__device__ void inclusiveBinaryPrefixScan(T* smem, bool in, T* out, BinaryFunction binop) {
  // Within-warp, we use warp voting.
#if defined (USE_ROCM)
  unsigned long long int vote = WARP_BALLOT(in);
  T index = __popcll(getLaneMaskLe() & vote);
  T carry = __popcll(vote);
#else
  T vote = WARP_BALLOT(in);
  T index = __popc(getLaneMaskLe() & vote);
  T carry = __popc(vote);
#endif

  int warp = threadIdx.x / C10_WARP_SIZE;

  // Per each warp, write out a value
  if (getLaneId() == 0) {
    smem[warp] = carry;
  }

  __syncthreads();

  // Sum across warps in one thread. This appears to be faster than a
  // warp shuffle scan for CC 3.0+
  if (threadIdx.x == 0) {
    int current = 0;
    for (int i = 0; i < blockDim.x / C10_WARP_SIZE; ++i) {
      T v = smem[i];
      smem[i] = binop(smem[i], current);
      current = binop(current, v);
    }
  }

  __syncthreads();

  // load the carry from the preceding warp
  if (warp >= 1) {
    index = binop(index, smem[warp - 1]);
  }

  *out = index;

  if (KillWARDependency) {
    __syncthreads();
  }
}
```

exclusiveBinaryPrefixScan 相对而言就没啥了，只是单纯的索引减一，也就是不包括本身，同时将数量总和计算一下。

```text
template <typename T, bool KillWARDependency, class BinaryFunction>
__device__ void exclusiveBinaryPrefixScan(T* smem, bool in, T* out, T* carry, BinaryFunction binop) {
  inclusiveBinaryPrefixScan<T, false, BinaryFunction>(smem, in, out, binop);

  // Inclusive to exclusive
  *out -= (T) in;

  // The outgoing carry for all threads is the last warp's sum
  *carry = smem[at::ceil_div<int>(blockDim.x, C10_WARP_SIZE) - 1];

  if (KillWARDependency) {
    __syncthreads();
  }
}
```

有了上面的知识，看下面代码就比较清晰了：就是计算该写的位置的索引。

```text
    int index;
    int carry;
    at::cuda::exclusiveBinaryPrefixScan<int, true>(
        smem, hasTopK, &index, &carry, AddOp<int>());

    if (hasTopK) {
      int writeIndex = writeIndexStart + index;
      CUDA_KERNEL_ASSERT(writeIndex < outputSliceSize);

      IndexType topKOffset = writeIndex * topKWithinSliceStride;
      IndexType indexOffset = writeIndex * indicesWithinSliceStride;

      topKSliceStart[topKOffset] = v;
      indicesSliceStart[indexOffset] = i;
    }

    writeIndexStart += carry;
```

但是还没有结束，因为很多值可能等于k-th本身，比如有下面的数

\[5,4,3,3,3,3,2\]

如果想找top3的数，只能先找5，4再找3，其实这块能优化一下，只不过这里不讲了。

然而这里还没有结束，因为should\_use\_multiblock，一些情况下（主要是数比较多的时候），会使用mbtopk，

也就是多block进行计算。这块代码十分复杂，我们从头看一下

```text
  auto stream = c10::cuda::getCurrentCUDAStream();

  // configure items_per_thread based on device architecture and input size
  int items_per_thread = get_items_per_thread(numInputSlices, inputSliceSize);
  int items_per_block = items_per_thread * BLOCK_THREADS;

  using Bitwise = typename TopKTypeConfig<T>::RadixType;
  uint32_t blocks_per_slice = at::ceil_div((int64_t)inputSliceSize, (int64_t)items_per_block);
  uint32_t num_blocks = numInputSlices * blocks_per_slice;

  // temporary storage
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();

  auto kthValues_buffer = allocator.allocate(numInputSlices * sizeof(T));
  T* kthValues = reinterpret_cast<T*>(kthValues_buffer.get());

  TORCH_CHECK(blocks_per_slice <= std::numeric_limits<uint32_t>::max(), "blocks_per_slice larger than uint32 maximum is not supported");
  auto semaphores_buffer = allocator.allocate(numInputSlices * sizeof(uint32_t));
  uint32_t* semaphores = reinterpret_cast<uint32_t*>(semaphores_buffer.get());
  AT_CUDA_CHECK(cudaMemsetAsync(semaphores, 0, numInputSlices * sizeof(uint32_t), stream));

  auto ks_to_find_buffer = allocator.allocate(numInputSlices * sizeof(uint32_t));
  uint32_t* ks_to_find = reinterpret_cast<uint32_t*>(ks_to_find_buffer.get());
  uint32_t k_to_find = largest ? inputSliceSize - outputSliceSize + 1: outputSliceSize;
  fill<uint32_t><<<std::min(((int64_t)numInputSlices + 511) / 512, (int64_t)1073741824), 512, 0, stream>>>(
    ks_to_find, k_to_find, numInputSlices);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto desired_buffer = allocator.allocate(numInputSlices * sizeof(Bitwise));
  Bitwise* desired = reinterpret_cast<Bitwise*>(desired_buffer.get());

  auto counts_buffer = allocator.allocate(num_blocks * RADIX_DIGITS * sizeof(short));
  short* counts = reinterpret_cast<short*>(counts_buffer.get());
  static_assert(MAX_ITEMS_PER_THREAD * BLOCK_THREADS < std::numeric_limits<short>::max(),
    "blockwise counter too large");

#if CUB_SUPPORTS_SCAN_BY_KEY()
  auto withinKCounts_buffer = allocator.allocate(num_blocks * sizeof(uint32_t));
  uint32_t* withinKCounts = reinterpret_cast<uint32_t*>(withinKCounts_buffer.get());
  AT_CUDA_CHECK(cudaMemsetAsync(withinKCounts, 0, num_blocks * sizeof(uint32_t), stream));

  auto kthCounts_buffer = allocator.allocate(num_blocks * sizeof(uint32_t));
  uint32_t* kthCounts = reinterpret_cast<uint32_t*>(kthCounts_buffer.get());
#endif
```

先看计算items\_per\_thread 和items\_per\_block ，也就是每个线程处理多少数据和每个block处理多少数据，代码在下面，比较清晰，通过一些硬件参数算出结果。对于每一行数据，都是通过blocks\_per\_slice 个block来计算，总共就有num\_blocks 个block

```text
int get_items_per_thread(uint64_t num_slices, uint64_t slice_size) {
  // occupancy of this kernel is limited by registers per threads
  constexpr int REGS_PER_THREAD = 40; // from nsight launch statistics
  constexpr int REGS_PER_BLOCK = REGS_PER_THREAD * BLOCK_THREADS;
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  int mpc = prop->multiProcessorCount;
#if defined(USE_ROCM)
  int regs_per_mp = prop->regsPerBlock;
  int max_blocks_per_mp = 32;
#else
  int regs_per_mp = prop->regsPerMultiprocessor;
#if !defined(USE_ROCM)
  int max_blocks_per_mp = prop->maxBlocksPerMultiProcessor;
#else
  int max_blocks_per_mp = 32;
#endif
#endif
  int blocks_per_mp = std::min(regs_per_mp / REGS_PER_BLOCK, max_blocks_per_mp);
  int64_t items_per_thread = at::ceil_div((int64_t)(slice_size * num_slices), (int64_t)(mpc * blocks_per_mp * BLOCK_THREADS));
  items_per_thread = std::max(MIN_ITEMS_PER_THREAD, std::min((int)items_per_thread, MAX_ITEMS_PER_THREAD)); // clamp to (4, 64)
  return items_per_thread;
}
```

CUDACachingAllocator是一个显存分配器，我们发现其分配了很多buffer,这是用来block之间通讯用的，因为block之间只能用显存通信了。

然后mbtopk有自己的radix桶，比较大，8个bit一个桶

```text
constexpr int RADIX_BITS = 8;
constexpr int RADIX_DIGITS = 1 << RADIX_BITS; // 2 ^ RADIX_BITS
constexpr int RADIX_MASK = (RADIX_DIGITS - 1);
```

看第一个for循环，其中radixFindKthValues的参数很多都是之前申请的buffer，我们开始分析radixFindKthValues

```text
  // iterate radix bits for multiple passes
  for (int current_bit = sizeof(T) * 8 - RADIX_BITS; current_bit >= 0; current_bit -= RADIX_BITS) {
    radixFindKthValues<T, IndexType, Bitwise, Dim><<<grid, block, 0, stream>>>(
        input,
        inputSliceSize,
        ks_to_find,
        numInputSlices,
        inputWithinSliceStride,
        current_bit,
        items_per_thread,
        blocks_per_slice,
        desiredMask,
        semaphores,
        desired,
        counts,
        kthValues);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#if CUB_SUPPORTS_SCAN_BY_KEY()
    computeBlockwiseWithinKCounts<Bitwise><<<grid, RADIX_DIGITS, 0, stream>>>(
      desired, counts, blocks_per_slice, current_bit, largest, withinKCounts, num_blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
    desiredMask = at::cuda::Bitfield<Bitwise>::setBitfield(desiredMask, RADIX_MASK, current_bit, RADIX_BITS);
  }
```

radixFindKthValues比较长，一段段看，比较贴心的是，很多参数有点注释。

```text
template <typename T, typename IndexType, typename Bitwise, int Dim>
C10_LAUNCH_BOUNDS_1(BLOCK_THREADS)
__global__ void radixFindKthValues(
    at::cuda::detail::TensorInfo<const T, IndexType> input,
    uint32_t slice_size,
    uint32_t* ks_to_find,  // size: num_slices

    uint32_t num_slices,
    IndexType withinSliceStride,

    int current_bit,
    int items_per_thread,
    uint32_t blocks_per_slice,
    Bitwise desiredMask,

    // outputs
    uint32_t* semaphores,  // size: num_slices
    Bitwise* desires,      // size: num_slices
    short* counts,         // size: num_slices * blocks_per_slice * radix_digits
    T* kthValues           // size: num_slices, only write when current_bit reaches 0
  ) {

  int items_per_block = items_per_thread * BLOCK_THREADS;
  int tidx = threadIdx.x;
  uint32_t block_idx = getLinearBlockId<uint32_t>();
  uint32_t slice_idx = block_idx / blocks_per_slice;
  uint32_t blk_idx_in_slice = block_idx % blocks_per_slice;
  if (slice_idx >= num_slices) {
    return;
  }

  Bitwise desired = desires[slice_idx];
  uint32_t k_to_find = ks_to_find[slice_idx];
  IndexType slice_start_index = at::cuda::detail::IndexToOffset<const T, IndexType, Dim>::get(slice_idx, input);
  const T* data = &input.data[slice_start_index];
```

先看前面的计算，由于把一组数据分成不同的block计算，所以得知道当前block的角色，一个是slice\_idx，即处理哪一组数据，一个是blk\_idx\_in\_slice，即负责数据的哪一部分。

接着根据这些索引，找到对应的desires，ks\_to\_find，以及数据指针等。

下面是一个cub库的BlockScan，作用是类似求前缀和，后面有时间单独分析。

```text
  typedef cub::BlockScan<uint32_t, BLOCK_THREADS> BlockScan;
  static_assert(MAX_ITEMS_PER_THREAD * BLOCK_THREADS < std::numeric_limits<short>::max(),
    "blockwise counter too large");
  union __align__(16) TempStorage {
    uint32_t digit_counters[RADIX_DIGITS];
    uint32_t digit_count_cumsum[RADIX_DIGITS]; // only used if this it the last block for this slice
    typename BlockScan::TempStorage scan_storage;
  };
  __shared__ TempStorage temp_storage;

  // fill digit_counters with zeros
  if (tidx < RADIX_DIGITS) {
    temp_storage.digit_counters[tidx] = 0;
  }
  __syncthreads();
```

items\_per\_thread 计算这个线程处理多少数据，尾部分单独计算。

```text
  items_per_thread = (blk_idx_in_slice + 1 < blocks_per_slice)
      ? items_per_thread
      : at::ceil_div((int64_t)(slice_size - blk_idx_in_slice * items_per_block), (int64_t)BLOCK_THREADS);
```

下面则是每个线程处理自己items\_per\_thread 这些数据，逻辑和sbtopk差不多，匹配模式，然后直接通过atomicAdd记录到共享内存中。这是因为每个线程都处理很多数据，再通过warp同步反而会慢

```text
  for (int i = 0; i < items_per_thread; ++i) {
    // Find the start offset for this slice
    IndexType idx = blk_idx_in_slice * items_per_block + i * BLOCK_THREADS + tidx;
    if (idx < slice_size) {
      idx *= withinSliceStride;
      Bitwise val = TopKTypeConfig<T>::convert(doLdg(&data[idx]));
      bool has_val = ((val & desiredMask) == (desired & desiredMask));
      Bitwise digit = at::cuda::Bitfield<Bitwise>::getBitfield(val, current_bit, RADIX_BITS);
      if (has_val) {
        atomicAdd(&temp_storage.digit_counters[digit], 1);
      }
    }
  }
```

下面要将digit\_count 写回到显存中，\_\_threadfence是内存屏障，是为了确定写到显存的数值可见。

```text
  // load digit counter to register, one digit per thread
  static_assert(RADIX_DIGITS <= BLOCK_THREADS, "this kernel requires RADIX_DIGITS <= BLOCK_THREADS");
  uint32_t digit_count = 0;
  if (tidx < RADIX_DIGITS) {
    digit_count = temp_storage.digit_counters[tidx];
  }

  // We always write out counts regardless if blocks_per_slice == 1 because
  // it will be used to compute offsets for `gatherTopK`.
  if (tidx < RADIX_DIGITS) {
    counts[block_idx * RADIX_DIGITS + tidx] = digit_count;
  }
  // if blocks_per_slice == 1, there is no need to do cross-block reduction
  // in this case we use counts saved at registers directly
  if (blocks_per_slice > 1) {
    __threadfence(); // make sure writes are globally visible
    __syncthreads(); // make sure all writes are finished before update semaphores
  }
```

下面又是一个小技巧，得到最后一个走到这里的block

因为block很多，block先后执行顺序不确定，通过这个方式，就知道处理这批数据的最后一个执行到这里block是谁了，此时counts已经被写回显存，不是最后一个block直接返回就行。

```text
  if (tidx == 0) {
    if (blocks_per_slice == 1) {
      s_is_last_block_done = true;
    } else {
      uint32_t blocks_finished_old = atomicAdd(&semaphores[slice_idx], 1);
      s_is_last_block_done = (blocks_finished_old == blocks_per_slice - 1);
    }
  }

  __syncthreads();

  if (!s_is_last_block_done)
    return;
```

接着就算了digit\_count 这个所有Block的count总和，最后一个block的256个线程分别负责一个radix桶的值，并通过BlockScan方法，将前缀和计算出来，写到smem中。

```text
  // accumulates counters from multiple blocks
  if (tidx < RADIX_DIGITS && blocks_per_slice > 1) {
    digit_count = 0;
    for (int blk = 0; blk < blocks_per_slice; ++blk) {
      digit_count += counts[(slice_idx * blocks_per_slice + blk) * RADIX_DIGITS + tidx];
    }
  }

  // compute the block-wide inclusive prefix sum
  uint32_t digit_count_cumsum;
  BlockScan(temp_storage.scan_storage).InclusiveSum(digit_count, digit_count_cumsum);
  __syncthreads();
  // every thread also need the perfix_sum of it's left value for comparison, so save a copy in shared mem
  if (tidx < RADIX_DIGITS) {
    temp_storage.digit_count_cumsum[tidx] = digit_count_cumsum;
  }
  __syncthreads();
```

digit\_count\_cumsum\_left 是计算左边不算自己的前缀和，如果满足条件 digit\_count\_cumsum\_left < k\_to\_find && k\_to\_find <= digit\_count\_cumsum，那就说明这个kth就在这个桶中了，ks\_to\_find就需要减去digit\_count\_cumsum\_left ，继续向下搜索，如果current\_bit等于0，就说明已经找到头了（似乎可以提前停止一下, 但是提前停止的话又需要遍历一次找数，对于dim\_size比较大的场景并不好。）

```text
  if (tidx < RADIX_DIGITS) {
    uint32_t digit_count_cumsum_left = (tidx == 0) ? 0 : temp_storage.digit_count_cumsum[tidx - 1];

    // if not the last pass: update desired and ks_to_find
    // if last pass: write out the kth value
    if (digit_count_cumsum_left < k_to_find && k_to_find <= digit_count_cumsum) {
      desired = at::cuda::Bitfield<Bitwise>::setBitfield(desired, tidx, current_bit, RADIX_BITS);
      desires[slice_idx] = desired;
      if (current_bit > 0) {
        ks_to_find[slice_idx] = k_to_find - digit_count_cumsum_left;
      } else {
        kthValues[slice_idx] = TopKTypeConfig<T>::deconvert(desired);
      }
    }
  }
```

找到这个k-th后，再找剩下的k个值就简单了，可以直接调用sbtopk的 gatherTopK，只不过WithKthValues这里是true，不需要再调用radixSelect了，剩下的都一样。

但是mbtopk还有一种优化方式，基于CUB\_SUPPORTS\_SCAN\_BY\_KEY，这个后续会单开一篇文章记录一下，也已经有大佬研究过了[CUB scan 算法学习](https://zhuanlan.zhihu.com/p/596332478)

为了后面的算法，还需要一些前期准备withinKCounts记录着每个block上符合条件数据的数量

computeBlockwiseWithinKCounts其实就相当于一个block级别的reduce，但是有很多优化，第一个就是判断哪些warp 哪些thread需要工作。

对于选取最大topk，end\_of\_warp 是每个warp的最后一个线程id，比如warp 0 对应31，warp 1对应63，

那么如何判断一个线程/warp需不需要工作？

```text
  bool warp_is_active, thread_is_active;
  int warp = tidx / C10_WARP_SIZE;
  if (largest) {
    int end_of_warp = warp * C10_WARP_SIZE + C10_WARP_SIZE - 1;
    warp_is_active = end_of_warp > desired_digit;
    thread_is_active = tidx > desired_digit;
  }
```

回到radixFindKthValues函数，我们注意到desired的赋值。

```text
    if (digit_count_cumsum_left < k_to_find && k_to_find <= digit_count_cumsum) {
      desired = at::cuda::Bitfield<Bitwise>::setBitfield(desired, tidx, current_bit, RADIX_BITS);
      desires[slice_idx] = desired;
```

tidx的值其实就是对应着radix桶，都是0~255，如果满足digit\_count\_cumsum\_left < k\_to\_find && k\_to\_find <= digit\_count\_cumsum 说明数据就在这个桶中，就确定这部分desired了。

回到上面，只有tidx 比这部分大，才能确定topk值在这些线程处理范围内。

在下面代码中，进行了一次block级别的reduce,统计了这个block中，有多少值是大于kth的。

首先count初始化为0，counts的大小为 num\_slices \* blocks\_per\_slice \* radix\_digits，每个block的每个radix桶都有一个计数，这个digit\_count是这个block统计到的，在这个radix中的数据数量（这个代码来自 radixFindKthValues）。

```text
  if (tidx < RADIX_DIGITS) {
    counts[block_idx * RADIX_DIGITS + tidx] = digit_count;
  }
```

只有比k值大的count才会进行统计，显示线程将对应的radix桶中数据读入，然后进行warp reduce，每个warp将值都写入warp\_counts，再进行一次reduce就是block reduce了。

```text
  uint32_t count = 0;
  if (warp_is_active) {
    if (thread_is_active) {
      count = doLdg(counts + block_idx * RADIX_DIGITS + tidx);
    }
    for (int offset = C10_WARP_SIZE / 2; offset > 0; offset /= 2) {
      count += WARP_SHFL_DOWN(count, offset);
    }
  }

  constexpr int num_warps = RADIX_DIGITS / C10_WARP_SIZE;
  __shared__ uint32_t warp_counts[num_warps];
  if (tidx % C10_WARP_SIZE == 0) {
    warp_counts[warp] = count;
  }
  __syncthreads();
  static_assert(RADIX_DIGITS < C10_WARP_SIZE * C10_WARP_SIZE,
    "Assuming only 1 warp is needed for final reduction");
  if (warp != 0) {
    return;
  }
  count = 0;
  if (tidx < num_warps) {
    count = warp_counts[tidx];
  }
  for (int offset = num_warps / 2; offset > 0; offset /= 2) {
    count += WARP_SHFL_DOWN(count, offset);
  }
  if (tidx == 0) {
    withinKCounts[block_idx] += count;
  }
```

withinKCounts这个值比较有意思， radixFindKthValues的运行是从高位到低位，所以效果是这样

|24-31| |小于kth的 | 不确定大于kth还是小于kth | 大于kth的|

|16-23| |小于| 不确定 |大于 |

这样不断累加确定大于kth的数量，到最后就可以得到确定的，这个block中有多少数是大于kth的。

有了确定的kth，还有withinKCounts这种统计数据，就可以得到最后的topk，注意要是没有C10\_CUDA\_KERNEL\_LAUNCH\_CHECK这个，用sbtopk::gatherTopK也是一样的。

```text
#if CUB_SUPPORTS_SCAN_BY_KEY()
  computeBlockwiseKthCounts<Bitwise><<<std::min(((int64_t)numInputSlices + 255) / 256, (int64_t)1073741824), 256, 0, stream>>>(
    desired, counts, num_blocks, blocks_per_slice, kthCounts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // Do a prefix scan of withinKCounts and kthCounts using slice_idx as keys to get the starting index of each block
  using counting_iter_t = cub::CountingInputIterator<uint32_t, uint32_t>;
  using slice_idx_iter_t = cub::TransformInputIterator<uint32_t, BlockIdxToKey, counting_iter_t>;
  slice_idx_iter_t slice_idx_iter(counting_iter_t(0), BlockIdxToKey(blocks_per_slice));
  at::cuda::cub::inclusive_sum_by_key(slice_idx_iter, withinKCounts, withinKCounts, num_blocks);
  at::cuda::cub::inclusive_sum_by_key(slice_idx_iter, kthCounts, kthCounts, num_blocks);
  // copy topk values to output tensor
  gatherTopK<T, IndexType, Dim><<<grid, block, 0, stream>>>(
    input, inputSliceSize, outputSliceSize, largest, numInputSlices, inputWithinSliceStride,
    topK, topKWithinSliceStride, indices, indicesWithinSliceStride, items_per_thread,
    blocks_per_slice, kthValues, withinKCounts, kthCounts, num_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
  // Find topk values based on kth values
  {
    dim3 grid;
    TORCH_INTERNAL_ASSERT(getGridFromTiles(numInputSlices, grid), "Too many slices for topk");
    int warp_size = at::cuda::warp_size();
    dim3 block(std::min(at::ceil_div((int64_t)inputSliceSize, (int64_t)warp_size) * (int64_t)warp_size, (int64_t)1024));
    sbtopk::gatherTopK<T, IndexType, Dim, /* WithKthValues= */true><<<grid, block, 0, stream>>>(
        input,
        inputSliceSize,
        outputSliceSize,
        largest,
        numInputSlices,
        inputWithinSliceStride,
        topK,
        topKWithinSliceStride,
        indices,
        indicesWithinSliceStride,
        kthValues);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
```

那么就需要看一下CUB\_SUPPORTS\_SCAN\_BY\_KEY这个优化到哪里了，这里不深入cub库原理了，只是看一下用法。其实就是下面的效果，key相同的计算前缀和

```text
int data[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};

after inclusive_sum_by_key
// data is now {1, 2, 3, 1, 2, 1, 1, 2, 3, 4};
```

换句话说，也就是处理同一组数据的block之间进行前缀和计算。

我们举个例子，我现在有数据（2，12），每一组数也就是12个数，通过4个block计算，也就是总共8个block，那么这个key就是

\[0，0，0，0，1，1，1，1\]

现在知道withinKCounts的值，表明每个block中，有多少个值是确定大于kth的

\[1,1,0,2,1,3,1,0\]

那么前缀和就为

\[1,2,2,4,1,4,5,5\]

```text
class BlockIdxToKey {
  uint32_t blocks_per_slice;
public:
  BlockIdxToKey(uint32_t blocks_per_slice): blocks_per_slice(blocks_per_slice) {}
  __device__ __forceinline__ uint32_t operator()(uint32_t blk) const {
    return blk / blocks_per_slice;
  }
};
```

算这些值有什么用？还是为了坐标，因为最后将k个值写入显存，是不同的block同时工作的，每个block需要知道自己该从哪个block开始写。

其实topk最麻烦的一点，就在于最后的kth可能比较多，还得想明白要哪些kth。

看一下最终的gatherTopK，这个是mbtopk的，和sbtopk还不大一样。前面的初始化部分大同小异，我们直接看后面填数部分。

startWithinK是上一个block确定的大于kth的数量，这也是inclusive\_sum\_by\_key的必要性。

startKth 初始化为负责这个slice的所有确定的大于kth的数量，并且再向后偏移上一个block等于kth的数量。

然后开始线程内循环，这里有两个bool，一个是withinK，代表这个值是不是topk内部的值，一个是kth，代表着这个值就是kth

第一个坐标 withinKIndex + startWithinK比较好理解，withinKIndex就是线程之间的前缀和。

```text
  // Find the k-th highest element in our input
  T kthValue = kthValues[slice_idx];
  const auto kthValueConverted = at::native::TopKTypeConfig<T>::convert(kthValue);

  // Find the start index in output tensor of this block
  uint32_t startWithinK = 0;
  if (blk_idx_in_slice > 0) {
    startWithinK = withinKCounts[block_idx - 1];
  }
  uint32_t startKth = withinKCounts[slice_idx * blocks_per_slice + blocks_per_slice - 1];
  if (blk_idx_in_slice > 0) {
    startKth += kthCounts[block_idx - 1];
  }

  // Read input, select topk out and write
  typedef cub::BlockScan<uint32_t, BLOCK_THREADS> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;
  for (int i = 0; i < items_per_thread; ++i) {
    // Find the start offset for this slice
    IndexType idx = blk_idx_in_slice * items_per_block + i * BLOCK_THREADS + tidx;
    T val;
    int withinK = 0;
    int kth = 0;
    if (idx < inputSliceSize) {
      val = doLdg(inputSliceStart + idx * inputWithinSliceStride);
      const auto valConverted = at::native::TopKTypeConfig<T>::convert(val);
      withinK = (largest ? valConverted > kthValueConverted : valConverted < kthValueConverted);
      kth = (valConverted == kthValueConverted);
    }

    uint32_t withinKIndex;
    uint32_t numWithinK;
    BlockScan(temp_storage).ExclusiveSum(withinK, withinKIndex, numWithinK);
    __syncthreads();
    if (withinK) {
      uint32_t offset = withinKIndex + startWithinK;
      topKSliceStart[offset * topKWithinSliceStride] = val;
      indicesSliceStart[offset * indicesWithinSliceStride] = idx;
    }
    startWithinK += numWithinK;

    if (startKth < outputSliceSize) {
      uint32_t kthIndex;
      uint32_t numKth;
      BlockScan(temp_storage).ExclusiveSum(kth, kthIndex, numKth);
      __syncthreads();
      if (kth) {
        uint32_t offset = kthIndex + startKth;
        if (offset < outputSliceSize) {
          topKSliceStart[offset * topKWithinSliceStride] = val;
          indicesSliceStart[offset * indicesWithinSliceStride] = idx;
        }
      }
      startKth += numKth;
    }
  }
```

第二个坐标可能没那么好理解，首先要清楚,pytorch是不保证数值顺序的，比如：

```text
>>> import torch
>>> a = torch.tensor([3,2,2,2,1,1,1])
>>> a.topk(3)
torch.return_types.topk(
values=tensor([3, 2, 2]),
indices=tensor([0, 1, 3]))
```

如果kth等于2，总共3个2它是随便选出俩来，并不一定按顺序来。

其实它基本是下面的结构，算出坐标看看超了k没，没超就往里填数

| withinK0 | kth0 | withinK1 | kth1 | withinK2 | kth2 |

| block0 | block1 |block2 |

| withinK0 |startWithinK1

| withinK0 withinK1 withinK2 kth0 |startKth

我觉得靠上面图比较清晰了，首先先填withinK，碰见kth，就看看还有空余没，有就接着填，当然这样也导致，kth是随机选择滴。