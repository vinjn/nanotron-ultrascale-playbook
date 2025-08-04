# Dynamo 代码阅读（一）：如何用 OffsetAllocator 实现高效、低碎片的页式分配

**Author:** SuSun

**Date:** 2025-07-12

**Link:** https://zhuanlan.zhihu.com/p/1927422948811186672

## TL;DR:

-   Dynamo 把 System/GPU/Disk 等内存抽象成一块连续字节，按页切成帧。
-   分配器基于开源的OffsetAllocator的Rust binding
-   后者用 SmallFloat 压缩块大小，两级 bitmap 找桶，O(1) 分配/合并。

dynamo 是如何管理多级存储的存储空间，并进行KV Cache的分配的。分配器定义如下(不得不说还是Rust代码读起来舒服）

```rust
#[derive(Clone)]
pub struct ArenaAllocator<S: Storage> {
    storage: Arc<S>,
    allocator: Arc<Mutex<Allocator>>,
    page_size: u64,
}
```

`Storage` 泛型抽象了不同的memory resource，支持的类型如下

```rust
/// Represents the type of storage used for a block
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageType {
    /// System memory
    System,

    /// CUDA device memory
    Device(u32),

    /// CUDA page-locked host memory,
    Pinned,

    /// Disk memory
    Disk,

    /// Remote memory accessible through NIXL
    Nixl,

    /// Null storage
    Null,
}
```

抽象如下，类型，一块连续可随机访问的内存区域。

```rust
/// Core storage trait that provides access to memory regions
pub trait Storage: Debug + Send + Sync + 'static {
    /// Returns the type of storage
    fn storage_type(&self) -> StorageType;

    /// Returns the address of the storage
    fn addr(&self) -> u64;

    /// Returns the total size of the storage in bytes
    fn size(&self) -> usize;

    /// Get a raw pointer to the storage
    ///
    /// # Safety
    /// The caller must ensure:
    /// - The pointer is not used after the storage is dropped
    /// - Access patterns respect the storage's thread safety model
    unsafe fn as_ptr(&self) -> *const u8;

    /// Get a raw mutable pointer to the storage
    ///
    /// # Safety
    /// The caller must ensure:
    /// - The pointer is not used after the storage is dropped
    /// - No other references exist while the pointer is in use
    /// - Access patterns respect the storage's thread safety model
    unsafe fn as_mut_ptr(&mut self) -> *mut u8;
}
```

Allocator 是一个C++ 分配库的Rust binding ([https://github.com/sebbbi/OffsetAllocator](https://link.zhihu.com/?target=https%3A//github.com/sebbbi/OffsetAllocator))。

核心理念就是以 `page_size` 作为分配的基本单位，传入一个 `Storage` 会被切分，然后交给底层的 `Allocator`进行分配

```rust
let pages = storage.size() / page_size;

        let allocator = Allocator::new(
            pages
                .try_into()
                .map_err(|_| ArenaError::PagesNotConvertible)?,
        );
```

也就是说 `Allocator` 实际管理的单位也是页。

分配不直接返回 `(ptr,size)` ，而是定义了一个新的结构体，用来做生命周期的管理。

```rust
pub struct ArenaBuffer<S: Storage> {
    offset: u64,
    address: u64,
    requested_size: usize,
    storage: Arc<S>,
    allocation: Allocation,
    allocator: Arc<Mutex<Allocator>>,
}
```

具体的分配流程（精简），分出出来页后会再映射回去。

```rust
let pages = size.div_ceil(self.page_size); 
let allocation = self.allocator.lock().unwrap().allocate(pages);
let offset = allocation.offset as u64 * self.page_size;
let address = self.storage.addr() + offset;
```

不需要线程安全，以页面为单位的分配并不难写，Mutex 套个Bitmap即可。为什么需要这个Allocator呢？不同于一般存储系统的分chunk分配，chunk之间可以散落在各个地方。这里的内存分配虽然分了page，但实际上还是连续内存。二是考虑到内存碎片率，分配回收的效率问题。

所以核心其实都offload到了之前提到的[https://github.com/sebbbi/OffsetAllocator](https://link.zhihu.com/?target=https%3A//github.com/sebbbi/OffsetAllocator)。继续研究下，一个头文件，一个源代码文件，非线程安全，加起来大概六七百行代码不算太多。

为什么要研究这个呢？原因是秋招某大厂面试，终面，面试官让我设计一个内存分配（我寻思我简历也没写这个），挂了。后来毕设也是做的分配，不过是磁盘分配，非常挫的实现。对此耿耿于怀，今天刚好有机会。

比起最naive的实现，也就是用一个大的bitmap。它做了什么优化？

随着分配的进行，大块内存被切碎，可用的内存以不同的大小，散落在不同的offset处。

我们需要先以可用内存块的大小做一次查找，可用内存块的大小至少要大于我即将分配的大小。

如果是我写可能会用一个 `map<size,list<chunk>>` ，找到第一个 ≥ 分配大小的内存块，应该叫 `best-fit`。如果 `size`的分布跨度足够大，这个map很大的话，查找有开销，map内部可能是个树，元数据有开销。

作者实现伪流程如下（2.5 pro画的）

```cpp
1. 输入：需要 100 个页 (pages = 100)
   │
   ▼
2. 计算 Bucket 索引 (SmallFloat Trick)
   100 (二进制 01100100) -> 指数=6, 尾数=5 -> bucket_index = (6 << 3) | 5 = 53
   │
   ▼
3. 拆分索引，用于两级查找
   Area Index   = 53 / 8 = 6
   Bucket Index in Area = 53 % 8 = 5
   │
   ▼
4. 第一级查找：在 32 位 Area Bitmap 中找 ≥ Area 6 的空闲 Area
   m_free_areas (uint32_t):
   Bit:  ...  7   6   5   4   3   2   1   0
   Val:  ...  0   1   1   0   0   1   0   0
                 ↑
                 └─ 找到了！Area 6 有空闲桶。如果Area 6为0，则会继续找Area 7...

   │
   ▼
5. 第二级查找：在 Area 6 的 8 位 Bucket Bitmap 中找 ≥ Bucket 5 的空闲 Bucket
   m_area_free_buckets[6] (uint8_t):
   Bit:    7   6   5   4   3   2   1   0
   Val:    1   0   1   0   0   0   1   0
                   ↑
                   └─ 找到了！Bucket 5 对应的空闲链表有块。如果Bucket 5为0，则会找Bucket 6, 7。

   │
   ▼
6. 进入空闲链表分配
   Bucket[53] -> [空闲块A] -> [空闲块B] -> NULL
                     │
                     └─ 取出 [空闲块A] 进行分配，如果它比100页大，
                        会分裂成两块：一块(100页)用于分配，
                        另一块(剩余部分)重新计算Bucket索引并放回空闲链表。
   │
   ▼
7. 返回分配结果
   Allocation { offset: (块A的页起始偏移), size: 100 }
```

作者用了一个叫做 `SmallFloat` 的trick，指数加尾数，trade 出去精度。尾数乘以 2 的指数倍，尾数只有3个bit决定了细节的精度，指数则决定了”数量级“，比如是大小是kB级别，还是mB。

给定一个大小，先算指数

1.  先定位到最高位的1
2.  然后后面有非0的位，就加1

比如100，指数就是6。3个尾数，细分为8个小格子，把64-127的空间均匀分成8个格子（0-7），就能找到，100就应该去\[96,103\]里面找，格子号为5。最后我们拿到了（数量级=6，格子号=5）的索引信息，左移三位数量级，把两者拼成一个数字，也就是53(53 = (6 << 3) | 5)。最终这个数是一个uint8\_t。

接下来就会确切的查找。思想类似于虚拟内存，做了一个多级的bitmap。

1.  第一级：32个Area，每个area内又有8个桶，32 \* 8 = 256. Area 有8个bit，分别表示8个桶是否有元素，如果8个桶都没有，这个area = 0。
2.  第二级：在Area之上，用一个uint32\_t来管理32个Area，这里应该也是一个二进制的trick，从左向右找到第一个非0的位置？

所以， 比如前文100对应的53 / 8 = 6，先看大的uin32\_t，看看看一眼第六个area 是否有任何空余，

-   等于0，压根没有空闲块，就去第七个area找，以此类推
-   不等于就在第六个里面继续查找，拿到第六个area的uint8\_t，查找，假如此时有≥5 的桶有空余，跳转到空闲链表进行分配。如果没有，就继续前进。

释放时候立即合并。

给我看晕了，不读了。总的来说，玩了足够多的二进制花活，据说还有了一些特殊指令，性能应该确实不错。