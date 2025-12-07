# cuTile Python 学习笔记

**作者：** 孙钦华

---

> 本人纯小白，这个博客主要是我自己学习cuTile的一些记录，任何信息以源码为准。

## 目录

> 1. 核心概念：Tile、Array、执行模型
> 2. Kernel 编写：装饰器、启动、编译期常量
> 3. 数据搬运：load/store、gather/scatter
> 4. Tile 操作：工厂函数、形变、算术、规约
> 5. 矩阵乘与原子操作
> 6. 类型系统与框架互操作

cuTile Python 是 NVIDIA 推出的一种面向 GPU 的编程语言，官方定义是 "a programming language for NVIDIA GPUs"。

先来一个最简单的例子，感受下 cuTile 代码长什么样：

```
import cuda.tile as ct

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    # Get the 1D pid
    pid = ct.bid(0)

    # Load input tiles
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))

    # Perform elementwise addition
    result = a_tile + b_tile

    # Store result
    ct.store(c, index=(pid, ), tile=result)

```

这段代码做的事情很直接：从显存加载两个 tile，相加，写回显存。`@ct.kernel` 装饰器声明这是一个 GPU kernel，`ct.load` 和 `ct.store` 负责数据搬运，中间的 `+` 就是逐元素加法。

# 核心设计：以 Tile 为中心

cuTile 的核心抽象单位是 Tile。这个 Tile 是什么？官方文档的定义是：

> A tile array (or tile) is an immutable multidimensional collection of values that is local to a block.

几个关键点：

Tile 是 block 级别的局部数据，Tile 仅在 block 的执行空间内有效；同一个 block 在一次 kernel 执行中可多次加载/处理不同的 tile。Tile 是 immutable 的，所有看起来像“修改”的操作实际上都会返回一个新的 tile。Tile 的每个维度必须是 2 的幂（包括 1），这是硬性约束。

与 Tile 对应的是 Array，也就是显存中的全局数组。cuTile 的数据流动模型就是 Array 和 Tile 之间的往返：

| 概念 | 存储位置 | 生命周期 | 可变性 |
| --- | --- | --- | --- |
| Array | 显存（全局） | — | 可作为 kernel 参数（可配合 store/scatter/atomic_* 写入） |
| Tile | block 内部 | block 执行期间 | 不可变 |

Array 可以是 `torch.Tensor` 或 `cupy.ndarray`，任何实现了 DLPack 或 CUDA Array Interface 的对象都可以直接当 Array 用，不需要额外转换。Tile 不能作为 kernel 参数；Array 可以作为 kernel 参数，且在 tile 代码中仅可只读其属性（`dtype/shape/strides/ndim`）。

# 并行模型：block 级显式，线程级隐藏

cuTile 采用的是 Grid-Block 二级并行模型，但它只暴露 block 这一层。

传统 CUDA 编程，你需要关心 grid 有多少 block、每个 block 有多少 thread、thread 之间怎么同步。cuTile 的做法是把线程级的复杂性藏起来：你只需要描述每个 block 要做什么计算，线程级接口未暴露。

用 `ct.bid(axis)` 获取当前 block 的索引，axis 可以是 0、1、2，对应 grid 的三个维度：

每个 block 执行 kernel 函数体一次。

启动 kernel 必须通过 `ct.launch`，不能直接调用：

```
grid = (ct.cdiv(vector_size, tile_size), 1, 1)
ct.launch(cp.cuda.get_current_stream(),
          grid,  # 1D grid of processors
          vector_add,
          (a, b, c, tile_size))

```

这里有个设计决策：kernel 函数被 `@ct.kernel` 装饰后，如果你直接调用 `vector_add(a, b, c, TILE)`，会抛出 `TypeError: Tile kernels cannot be called directly. Use cuda.tile.launch() instead.`。必须显式通过 `ct.launch` 指定 stream 和 grid 来启动。

# 与框架的互操作

cuTile 和 PyTorch、CuPy 的互操作可以直接传入 torch.Tensor 或 cupy.ndarray。你可以直接把 `torch.Tensor` 当作 kernel 参数传进去：

```
ct.launch(torch.cuda.current_stream(), grid, kernel, (a, b, c, TILE))

```

stream 也直接用框架提供的，PyTorch 用 `torch.cuda.current_stream()`，CuPy 用 `cupy.cuda.get_current_stream()`。这样 cuTile kernel 就能和框架的其他操作在同一个 stream 上顺序执行。

# 执行空间：Host code 与 Tile code

cuTile 把代码分成两个执行空间：Host code 和 Tile code。

Host code 就是普通的 CPU 端 Python 代码，做数据准备、调用 `ct.launch` 启动 kernel、拿结果这些事情。Tile code 是 GPU 端执行的代码，包括 kernel 函数体和被 kernel 调用的辅助函数。

这两个空间的边界很清晰：你不能在 host 直接执行 tile code；tile code 为受限 Python 子集，未见常规 Python I/O 能力；如需在 tile 侧打印请使用 `ct.printf`。

另外，tile code 支持的 Python 子集有一些限制，例如 `range` 的步长必须为正数（不支持负步长）。建议调用方确保传给 kernel 的多个 Array 不互相别名，并在 kernel 执行完成前保持有效。

# @ct.kernel：Tile code 的入口

`@ct.kernel` 装饰器把一个 Python 函数声明为 tile kernel。kernel 是 tile code 的入口点，每个 block 执行 kernel 函数体一次。

```
@ct.kernel
def vector_add_kernel(a, b, result):
    block_id = ct.bid(0)
    a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
    b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))
    result_tile = a_tile + b_tile
    ct.store(result, index=(block_id,), tile=result_tile)

```

kernel 有几个可选参数用于控制编译和执行行为：

| 参数 | 取值范围 | 含义 |
| --- | --- | --- |
| num_ctas | 1, 2, 4, 8, 16（2 的幂） | 每个簇（CGA）中的 CTA 数 |
| occupancy | 1 到 32 | 期望每个 SM 的活跃 CTA 数 |
| opt_level | 0, 1, 2, 3（默认 3） | 编译优化级别 |

这些参数都可以用 `ct.ByTarget` 按 GPU 架构特化，例如：`@ct.kernel(num_ctas=ct.ByTarget(sm_100=8, default=2))`。`ByTarget` 的键名形如 `"sm_<major><minor>"`。

有个重要的设计决策：kernel 不能直接调用。如果你写 `my_kernel(a, b, c)`，会直接抛出 `TypeError: Tile kernels cannot be called directly. Use cuda.tile.launch() instead.`。必须通过 `ct.launch` 来启动。

# @ct.function：Tile code 内的辅助函数

除了 kernel，你还可以用 `@ct.function` 声明在 tile code 中可调用的辅助函数：

```
@ct.function
def helper_function_using_ct_api(x, output, B: ct.Constant[int], N: ct.Constant[int]):
    px = ct.bid(0)
    tile_x = ct.load(x, index=(px, 0), shape=(B, N))
    ct.store(output, index=(px, 0), tile=tile_x + 1)

@ct.kernel
def kernel_calling_function_using_ct_api(x, output, B: ct.Constant[int], N: ct.Constant[int]):
    helper_function_using_ct_api(x, output, B, N)

```

`@ct.function` 有两个参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| host | False | 是否可从 host code 调用 |
| tile | True | 是否可从 tile code 调用 |

默认是 `host=False, tile=True`，也就是 tile-only。如果设置 `host=True`，这个函数在 host 和 tile 两边都能用，但这种情况比较少见。

还有一个特性：如果一个没有装饰器的普通函数被 tile function 调用了，cuTile 会自动把它当作 tile function 处理，这个过程是递归的。所以简单的辅助函数不一定非要加 `@ct.function`。

# ct.launch：启动 kernel

启动 kernel 的唯一方式是 `ct.launch`：

```
ct.launch(cp.cuda.get_current_stream(),
          grid,  # 1D grid of processors
          vector_add,
          (a, b, c, tile_size))

```

四个参数：

`stream` 是 CUDA stream，从你用的框架拿。PyTorch 用 `torch.cuda.current_stream()`，CuPy 用 `cupy.cuda.get_current_stream()`。

`grid` 是一个 tuple，指定 block 的数量。可以是 1 维 `(gx,)`、2 维 `(gx, gy)` 或 3 维 `(gx, gy, gz)`。常用 `ct.cdiv(N, TILE)` 来计算需要多少个 block，`cdiv` 是向上取整的除法。

`kernel` 就是被 `@ct.kernel` 装饰的函数。

`kernel_args` 是一个 tuple，包含传给 kernel 的所有参数。

来一个完整的例子：

```
def test():
    # Create input data
    vector_size = 2**12
    tile_size = 2**4
    grid = (ct.cdiv(vector_size, tile_size), 1, 1)

    a = cp.random.uniform(-1, 1, vector_size)
    b = cp.random.uniform(-1, 1, vector_size)
    c = cp.zeros_like(a)

    # Launch kernel
    ct.launch(cp.cuda.get_current_stream(),
              grid,  # 1D grid of processors
              vector_add,
              (a, b, c, tile_size))

    # Copy to host only to compare
    a_np = cp.asnumpy(a)
    b_np = cp.asnumpy(b)
    c_np = cp.asnumpy(c)

    # Verify results
    expected = a_np + b_np
    np.testing.assert_array_almost_equal(c_np, expected)

    print("✓ vector_add_example passed!")

```

# 并行结构：Grid 和 Block

启动 kernel 时指定的 grid 决定了有多少个 block 并行执行。每个 block 通过 `ct.bid(axis)` 获取自己在 grid 中的位置，axis 取值 0、1、2，分别对应 grid 的三个维度。`ct.num_blocks(axis)` 可以获取某个维度的 block 总数。

几个要注意的点：

block 内部没有显式的线程概念。传统 CUDA 你要管 threadIdx、blockDim 这些，cuTile 把这层抽象掉了，线程级接口未暴露。

未暴露线程级同步原语；如需跨 block 协调可结合原子操作。未见 `__syncthreads()` 等同步原语。

不同 block 之间可以通过 atomic 操作协调，但不能假设执行顺序。

# Array：显存中的全局数组

Array 是 cuTile 中表示显存数据的类型。官方定义是：

> A global array (or array) is a container of objects stored in a logical multidimensional space. Global arrays are always stored in memory.

Array 有几个核心特性：

可以在 host code 和 tile code 中使用，可以作为 kernel 参数传入。任何实现了 DLPack 或 CUDA Array Interface 的对象都可以当 Array 用，`torch.Tensor` 和 `cupy.ndarray` 都可以直接传。

在 tile code 中，Array 的属性是只读的：

| 属性 | 返回类型 | 说明 |
| --- | --- | --- |
| dtype | DType（常量） | 元素的数据类型 |
| shape | tuple[int32, ...] | 各维度的元素数量 |
| strides | tuple[int32, ...] | 各维度的步长 |
| ndim | int（常量） | 维度数量 |

这里有个细节：shape 和 strides 的元素类型在 tile code 中默认是 int32（参考 `src/cuda/tile/_stub.py:157` 与 `src/cuda/tile/_stub.py:168`），不是 Python 的任意精度整数。这是出于性能考虑，但也意味着处理超大张量时要注意表示范围（参考 `docs/source/data.rst:149-151`）。

# Tile：block 内的不可变张量

如前所述，Tile 是 block 级别的不可变多维数据集合。补充一点：Tile 的内容不一定有内存表示（可能只存在于寄存器中）。

几个关键约束：

Tile 只能在 tile code 中使用，不能在 host code 中出现。外部数据必须通过 Array 传入；Tile 不能作为 kernel 参数，但可在 kernel 内通过工厂/形变算子创建，或用 ct.load 从 Array 加载成 Tile（参考 `src/cuda/tile/_stub.py:473`）。

最重要的约束：Tile 的每个维度必须是 2 的幂（包括 1）。当数据尺寸与 tile 形状不整除时，可结合 load 的 padding_mode 或使用 gather/scatter 的越界行为处理（参考 `src/cuda/tile/_stub.py:519, 622`）。

Tile 的属性和 Array 类似：

| 属性 | 返回类型 | 说明 |
| --- | --- | --- |
| dtype | DType（常量） | 元素的数据类型 |
| shape | tuple[const int, ...] | 各维度的元素数量（编译期常量） |
| ndim | int（常量） | 维度数量 |

注意 Tile 的 shape 是编译期常量，这和 Array 不同。

# 0D Tile 与标量

cuTile 支持 0D Tile，也就是只有一个元素的 tile。0D Tile 和标量可以互换使用：

> A scalar is a single immutable value of a specific data type. A scalar and 0D-tile can be used interchangeably in a tile kernel.

几种获取 0D Tile 的方式：

用 `shape=()` 加载单个元素（摘自 `_stub.py` 示例）：

```
tile = ct.load(array3d, (0, 0, 0), shape=())

```

从单元素 Tile 提取（摘自 `_stub.py` 示例）：

```
tx = ct.full((1,), 0, dtype=ct.int32)
x = tx.item()
ty = ct.load(array, (0, x), shape=(4, 4))

```

0D Tile 可以用作索引，包括 `range` 的参数（摘自测试用例思路）：

```
idx = ct.full((), i, dtype=ct.int32)

```

# Tile Space

cuTile 的 ct.load 和 ct.store（参考 `src/cuda/tile/_stub.py:473`、`src/cuda/tile/_stub.py:547`）操作基于 Tile Space 的概念。Tile Space 是把 Array 按 tile shape 划分成的逻辑网格。

举个例子，一个 shape 为 `(32, 16)` 的 Array，如果用 tile shape `(4, 8)` 来划分：

```
Tile Space 大小 = (cdiv(32, 4), cdiv(16, 8)) = (8, 2)

```

也就是说这个 Array 被划分成了 8×2 = 16 个 tile。ct.load 的 index 参数指的是 Tile Space 中的坐标，不是元素坐标（摘自 `_stub.py` 示例）：

```
# 加载 Tile Space 中位置 (i, j) 的 tile
t = ct.load(array, (i, j), (tm, tn))  # `t` has shape (tm, tn)
# t[x, y] = array[i * tm + x, j * tn + y]

```

ct.num_tiles（参考 `src/cuda/tile/_stub.py:443`）可以获取某个轴上有多少个 tile：

```
# array shape: (32, 16), tile shape: (4, 8)
ct.num_tiles(array, axis=0, shape=(4, 8))  # 返回 8
ct.num_tiles(array, axis=1, shape=(4, 8))  # 返回 2

```

这个函数在写循环遍历所有 tile 时很有用（num_tiles 的 `shape` 也是编译期常量）。示例见 `samples/MatMul.py`：

```
num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
for k in range(num_tiles_k):
    a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=zero_pad).astype(dtype)
    b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=zero_pad).astype(dtype)

```

# 对象模型：不可变性与生命周期

tile code 中创建的对象都是不可变的。所有看起来像"修改"的操作实际上都返回新对象（参考 `docs/source/execution.rst:1`）。

唯一的可变对象是 Array，但 Array 必须作为 kernel 参数从外部传入。调用方需要保证两件事：

传给 kernel 的多个 Array 参数不能指向同一块内存（不能有 alias）。Array 在 kernel 执行完成之前必须保持有效，不能在 kernel 还在跑的时候把 Array 释放了。

# load 和 store

ct.load 和 ct.store（参考 `src/cuda/tile/_stub.py:473`、`src/cuda/tile/_stub.py:547`）是 cuTile 中最基本的数据访问操作，用于在 Array 和 Tile 之间搬运规则的数据块。

先看 ct.load 的签名：

```
def load(array: Array, /,
         index: Shape,
         shape: Constant[Shape], *,
         order: Constant[Order] = "C",
         padding_mode: PaddingMode = PaddingMode.UNDETERMINED,
         latency: Optional[int] = None,
         allow_tma: Optional[bool] = None) -> Tile:

```

`index` 是 Tile Space 中的坐标，不是元素坐标。`shape` 是要加载的 tile 的形状，必须是编译期常量，每个维度必须是 2 的幂；`order` 同样是编译期常量（可为 `'C'/'F'` 或常量置换元组）。

`latency` 与 `allow_tma` 是与访存相关的性能提示：

- `latency` 取值 1..10（数值越大表示 DRAM 压力越高），用于帮助编译器做调度与指令选择；
- `allow_tma` 用于允许/禁止在支持的架构上使用 TMA；
- 二者作为编译期提示存在，未指定时由编译器自行推断（参考 `docs/source/performance.rst:1`）。

加载的语义很直接：

```
t = ct.load(array, (i, j), (tm, tn))
# t[x, y] = array[i * tm + x, j * tn + y]  (对所有 0<=x<tm, 0<=y<tn)

```

如果访问越界，返回值由 `padding_mode` 决定：

| PaddingMode | 越界时的填充值 |
| --- | --- |
| UNDETERMINED | 未定义（默认） |
| ZERO | 0 |
| NEG_ZERO | -0 |
| NAN | NaN |
| POS_INF | +∞ |
| NEG_INF | -∞ |

ct.store 的签名：

```
def store(array: Array, /,
          index: Shape,
          tile: TileOrScalar, *,
          order: Constant[Order] = "C",
          latency: Optional[int] = None,
          allow_tma: Optional[bool] = None) -> None:

```

存储的语义：

```
ct.store(array, (i, j), t)
# array[i * tm + x, i * tn + y] = t[x, y]  (对所有 0<=x<tm, 0<=y<tn)

```

store 的越界行为和 load 不同：越界的写入会被忽略，不会报错也不会写到别的地方。当写入值为标量或 0D Tile 时，允许其秩与 Array 不匹配。

# order 参数：轴映射

`order` 参数用来控制 tile 的轴和 array 的轴之间的映射关系，有三种写法（注意：`order` 也是编译期常量）：

`'C'` 是默认值，等价于 `(0, 1, 2, ...)`，不做任何置换。

`'F'` 是轴逆序，等价于 `(..., 2, 1, 0)`。

也可以直接传一个置换元组，比如 `(0, 2, 1)` 表示交换最后两个轴。

来看一个转置的例子：

```
# 普通加载
tile = ct.load(array2d, (0, 0), shape=(2, 4))

# 加载时转置
tile = ct.load(array2d, (0, 0), shape=(4, 2), order='F')

# 3D 数组，交换最后两个轴
tile = ct.load(array3d, (0, 0, 0), shape=(8, 4, 2), order=(0, 2, 1))

```

store 同样支持 `order` 参数，语义与 load 保持一致。

# gather 和 scatter

ct.gather 和 ct.scatter（参考 `src/cuda/tile/_stub.py:596`、`src/cuda/tile/_stub.py:636`）提供了更灵活的访问方式，可以按任意索引读写数据。

ct.gather 的签名（参考 `src/cuda/tile/_stub.py:596`）：

```
def gather(array, indices, /, *, padding_value=0, check_bounds=True, latency=None) -> Tile:

```

`indices` 是一个元组，长度等于 array 的维度数。元组中的每个元素是整数 tile 或标量，它们会广播到一个公共形状，这个公共形状就是返回 tile 的形状。

看一个 2D 的例子：

```
# ind0 shape: (M, N, 1)
# ind1 shape: (M, 1, K)
t = ct.gather(array, (ind0, ind1))
# t shape: (M, N, K)  -- 广播后的公共形状
# t[i, j, k] = array[ind0[i, j, 0], ind1[i, 0, k]]

```

对于 1D array，有个简便写法，可以直接传单个索引 tile 而不是元组：

```
ct.gather(array1d, indices)  # 等价于 ct.gather(array1d, (indices,))

```

越界处理：默认返回 `padding_value`（默认是 0），可以是标量或者可广播到结果形状的 tile。一个重要的点：负索引被视为越界，不会像 Python 那样从末尾开始数。

ct.scatter 的签名和语义类似：

```
def scatter(array, indices, value, /, *, check_bounds=True, latency=None) -> None:

```

`indices` 的广播规则和 gather 一样，`value` 必须能广播到 indices 的公共形状。越界的写入会被忽略（当 `check_bounds=True` 时）。当 `check_bounds=False` 时，越界行为未定义，由调用方自行保证索引合法。

# load/store vs gather/scatter 怎么用

来看两种写法实现同样的向量加法：

用 load/store：即开篇的 `vector_add` 示例，基于 Tile Space 索引。

用 gather/scatter（摘自 `samples/VectorAddition.py:1`）：

```
@ct.kernel
def vec_add_kernel_1d_gather(a, b, c, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    indices = bid * TILE + ct.arange(TILE, dtype=torch.int32)
    a_tile = ct.gather(a, indices)
    b_tile = ct.gather(b, indices)
    sum_tile = a_tile + b_tile
    ct.scatter(c, indices, sum_tile)

```

两种方式的差异：

load/store 基于 Tile Space 的规则分块，并提供 `latency` 与 `allow_tma` 等访存提示；是否采用 TMA 由实现决定（参考 `src/cuda/tile/_stub.py:521-524`、`docs/source/performance.rst:1`）。

gather/scatter 按元素索引，更灵活。边界处理天然内置（越界返回 padding_value 或忽略写入）。适合处理非规则访问、掩码访问、或者数据大小不是 2 的幂的情况。

2D gather/scatter 的索引构造稍微复杂一点，需要手动处理广播（摘自 `samples/VectorAddition.py:1`）：

```
@ct.kernel
def vec_add_kernel_2d_gather(
    a, b, c,
    TILE_X: ct.Constant[int], TILE_Y: ct.Constant[int]
):
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)

    x = bid_x * TILE_X + ct.arange(TILE_X, dtype=torch.int32)
    y = bid_y * TILE_Y + ct.arange(TILE_Y, dtype=torch.int32)

    x = x[:, None]
    y = y[None, :]

    a_tile = ct.gather(a, (x, y))
    b_tile = ct.gather(b, (x, y))
    sum_tile = a_tile + b_tile
    ct.scatter(c, (x, y), sum_tile)

```

# 编译期常量：ct.Constant

cuTile 有一套编译期常量机制，用 `ct.Constant` 类型提示来标注。被标注为常量的参数会在编译时嵌入到生成的代码中（如开篇 `vector_add` 中的 `tile_size: ct.Constant[int]`）。

`ct.Constant` 可以带类型参数 `ct.Constant[int]`，也可以不带类型参数，写作 `ct.Constant`，表示任意类型的常量。

# 常量嵌入的语义

当一个参数被标注为 `ct.Constant` 时，会发生几件事：

所有使用这个参数的地方，都等同于直接把字面值替换进去。这个参数在编译后的机器表示中大小为 0 字节，因为值已经被"烧"进代码里了。

最重要的一点：不同的常量值会生成不同的编译变体。如果你用 `TILE=128` 调用一次，再用 `TILE=256` 调用一次，cuTile 会编译两份不同的 kernel 代码。每个取值各编译一次，JIT 缓存不会合并。

基于上述语义，通常建议控制常量取值的集合规模（不同值会分别编译）。

# 什么必须是常量

有几类参数必须是编译期常量：

`ct.load` 和 `ct.store` 的 `shape` 与 `order` 参数必须是常量。因为 tile 的形状决定了编译器如何生成数据搬运代码，这些必须在编译时确定。

```
tile = ct.load(array2d, (0, 0), shape=(2, 4))
tile = ct.load(array2d, (0, 0), shape=(4, 2), order='F')

```

`ct.num_tiles` 的 `shape` 与 `order` 参数也必须是常量，且 `axis` 为常量整数。

`ct.arange` 的 `size` 参数必须是常量，并且取值需满足 2 的幂（含 1）。

`ct.full`、`ct.zeros`、`ct.ones` 的 `shape` 参数必须是常量，并且每个维度必须为 2 的幂（含 1）。

# 常量表达式

什么样的表达式可以作为常量？官方文档列出了这些情况：

字面量对象，比如 `128`、`'C'`。

仅含字面量的整数算术表达式，比如 `16 * 8`。

从字面量或常量表达式赋值而得的局部变量或参数。

在编译或 launch 时已经定义的全局对象。

所以这样写是可以的：

```
# example-begin imports
import cuda.tile as ct
# example-end imports

# example-begin constant
def needs_constant(x: ct.Constant):
    pass

def needs_constant_int(x: ct.Constant[int]):
    pass
# example-end constant

```

# 宽松常量与严格常量

cuTile 区分两种常量类型：

宽松常量（loosely typed）是默认的常量类型。整数宽松常量在具体化之前具有无限精度，浮点宽松常量以 IEEE754 double 表示。字面量 `128` 就是宽松常量。

严格常量（strictly typed）是通过 dtype 构造器创建的常量，比如 `ct.int16(5)`。严格常量有明确的类型。

当宽松常量和严格常量一起运算时，结果变为严格常量。严格常量之间运算按类型提升规则确定结果类型。

当宽松常量被用在需要具体类型的上下文时，会按算术提升规则具体化。比如作为 tile 运算的操作数时，会根据另一个操作数的类型来确定具体化为什么类型。

# 典型用法

常量最常见的用法是定义 tile 的形状：

```
ConstInt = ct.Constant[int]

@ct.kernel(num_ctas=ct.ByTarget(sm_100=2))
def matmul_kernel(A, B, C,
                  tm: ConstInt,         # Tile size along M dimension (rows of C)
                  tn: ConstInt,         # Tile size along N dimension (columns of C)
                  tk: ConstInt):        # Tile size along K dimension (inner product dimension)
    GROUP_SIZE_M = 8
    M = A.shape[0]
    N = B.shape[1]
    bidx, bidy = swizzle_2d(M, N, tm, tn, GROUP_SIZE_M)

    # Calculate the total number of K-tiles that need to be processed.
    # `ct.num_tiles(A, axis=1, shape=(tm, tk))` extracts the K-dimension (axis 1)
    # from matrix A's shape, assuming A's shape is conceptually (M_tiles, K_tiles),
    # and then implicitly performs ceiling division by `tk` to get the number of K-tiles.
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))

    # Initialize an accumulator for the current output tile (tm x tn).
    # It's common practice to use `float32` for accumulation even with `float16` inputs
    # to maintain higher precision during the sum-reduction of the matrix multiplication.
    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    # Convert fp32 to tf32 to use tensorcore
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    # K-dimension loop: Iterate over the K-dimension in chunks of 'tk'.
    # In each iteration, a `tm` x `tk` tile from A and a `tk` x `tn` tile from B
    # are loaded, multiplied, and accumulated.
    for k in range(num_tiles_k):
        # Load tile from matrix A.
        # The `index=(bidx, k_tile_idx)` specifies which (M-tile, K-tile) to load
        # from global memory A. `shape=(tm, tk)` defines the size of this tile.
        a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=zero_pad).astype(dtype)

        # Load tile from matrix B.
        # The `index=(k_tile_idx, bidy)` specifies which (K-tile, N-tile) to load
        # from global memory B. `shape=(tk, tn)` defines the size of this tile.
        b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=zero_pad).astype(dtype)

        # Perform Matrix Multiplication for the current tiles.
        # `ct.mma` computes the product of the two loaded tiles and accumulates the result.
        accumulator = ct.mma(a, b, accumulator)

    # Convert the final accumulated result to the desired output data type (C.dtype).
    # This might downcast from float32 to float16 if the output is float16.
    accumulator = ct.astype(accumulator, C.dtype)

    # Store the computed tile to the global memory of the output matrix C.
    # The `(bidx, bidy)` directly corresponds to the tile's position in the 2D output matrix.
    ct.store(C, index=(bidx, bidy), tile=accumulator)

```

这里 `tm`、`tn`、`tk` 都是编译期常量，用于定义 tile 的形状。在调用时传入具体值：

```
kernel = persistent_matmul_kernel if persistent else matmul_kernel
ct.launch(torch.cuda.current_stream(), grid, kernel, (A, B, C, tm, tn, tk))

```

如果后续用不同的 tile 尺寸调用，比如 `(64, 128, 32)`，cuTile 会编译一个新的变体。

当维度不整除 `tm/tn/tk` 时，可使用 `padding_mode=ct.PaddingMode.ZERO` 处理尾块，也可使用 `gather/scatter`，或对输入进行 padding；`range` 的步长需为正。

# 工厂函数：创建 Tile

cuTile 提供四个工厂函数来创建 Tile，不需要从 Array 加载。

`ct.arange` 创建一个 1D 递增序列：

```
tile = ct.arange(16, dtype=ct.int32)

```

`size` 参数必须是编译期常量，且为 2 的幂（含 1）。这个函数在构造索引时特别有用，配合 gather/scatter 使用。

`ct.full` 创建一个填充指定值的 Tile：

```
tile = ct.full((4, 4), 3.14, dtype=ct.float32)

```

`ct.zeros` 和 `ct.ones` 是 `ct.full` 的特例：

```
tile = ct.ones((4, 4), dtype=ct.float32)
tile = ct.zeros((4, 4), dtype=ct.float32)

```

所有工厂函数的 `shape` 参数都必须是编译期常量，每个维度必须是 2 的幂（含 1）；`ct.arange` 的 `size` 也应满足 2 的幂约束。

# reshape

`ct.reshape` 改变 tile 的形状，元素总数必须保持不变：

```
tx = ct.arange(8, dtype=ct.float32)
tx.shape
(8,)
ty = ct.reshape(tx, (2, 4))
ty.shape
(2, 4)
tz = ct.reshape(tx, (2, -1))
tz.shape
(2, 4)

```

目标 `shape` 必须是编译期常量，且各维需为 2 的幂（含 1）。只能有一个维度是 `-1`。

# expand_dims

`ct.expand_dims` 在指定位置插入一个大小为 1 的新维度：

```
x = bid_x * TILE_X + ct.arange(TILE_X, dtype=torch.int32)
y = bid_y * TILE_Y + ct.arange(TILE_Y, dtype=torch.int32)
x = x[:, None]
y = y[None, :]

```

`axis` 必须是单个编译期常量整数。如需插入多个维度，可多次调用或使用切片语法链式插入。

文档示例展示使用 `None` 插入新维度（如 `x[:, None]`、`y[None, :]`）。

# transpose 和 permute

`ct.transpose` 交换两个轴：二维可省略参数进行转置；≥3D 时必须显式给出 `axis0/axis1`。

```
ConstInt = ct.Constant[int]

@ct.kernel
def transpose_kernel(x, y,
                     tm: ConstInt,
                     tn: ConstInt):
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    input_tile = ct.load(x, index=(bidx, bidy), shape=(tm, tn))
    transposed_tile = ct.transpose(input_tile)
    ct.store(y, index=(bidy, bidx), tile=transposed_tile)

```

`ct.permute` 可以做任意的轴置换（`axes` 为编译期常量整型元组）：

```
tx = ct.full((2, 4, 8), 0., dtype=ct.float32)
ty = ct.permute(tx, (0, 2, 1))
ty.shape
(2, 8, 4)

```

`axes` 参数是一个元组，指定新的轴顺序，且必须为编译期常量。

# broadcast_to

`ct.broadcast_to` 把 tile 广播到指定的形状（目标 `shape` 为编译期常量，且各维为 2 的幂）：

```
tx = ct.arange(4, dtype=ct.float32)
tx.shape
(4,)
ty = ct.broadcast_to(tx, (2, 4))
ty.shape
(2, 4)

```

广播规则和 NumPy 一致：

从尾部对齐维度。

如果两个维度相同，或者其中一个是 1，则兼容。

维度数较少的一方在左侧补 1。

（按照上述规则，较短形状会在左侧补 1 后参与广播。）

# cat

`ct.cat` 沿指定轴拼接两个 tile（`axis` 为编译期常量）：

```
tx = ct.full((2, 4), 3., dtype=ct.float32)
ty = ct.full((2, 4), 4., dtype=ct.float32)
tz = ct.cat((tx, ty), 0)
tz.shape
(4,4)
tz = ct.cat((tx, ty), 1)
tz.shape
(2,8)

```

由于 tile 形状都要求为 2 的幂，两个输入 tile 必须形状相同。

# 类型转换：astype 和 bitcast

`ct.astype` 做数值类型转换：

```
tx = ct.arange(8, dtype=ct.float32)
ty = ct.astype(tx, ct.float16)
ty.dtype
float16

```

`ct.bitcast` 做位模式重新解释，不改变底层的二进制表示：

```
tx = ct.arange(8, dtype=ct.float32)
ty = ct.bitcast(tx, ct.int32)
ty.dtype
int32

```

bitcast 要求源类型和目标类型的位宽兼容，不兼容将触发类型错误。

# 选择与提取：where 和 extract

`ct.where` 根据条件选择元素：

```
cond = ct.arange(4, dtype=ct.int32)
cond = cond > 2
x_true = ct.full((4,), 1.0, dtype=ct.float32)
x_false = ct.full((4,), -1.0, dtype=ct.float32)
y = ct.where(cond, x_true, x_false)
y
[1., 1., -1., -1.]
z = ct.where(cond, 1.0, -1.0)
z
[1., 1., -1., -1.]

```

`ct.extract` 从 tile 中提取子块，类似在 tile 上做 load：

```
tile = ct.full((8, 8), 3.14, dtype=ct.float32)
sub_tile = ct.extract(x, (0, 0), shape=(4, 4))
sub_tile.shape
(4, 4)

```

# 二元运算与运算符重载

cuTile 的 Tile 类型重载了 Python 的算术运算符，可以直接用 `+`、`-`、`*` 等符号进行运算（如开篇 `vector_add` 中的 `result = a_tile + b_tile`）。

也可以用对应 API 形式（如 `ct.add/ct.sub/ct.mul/ct.truediv/ct.floordiv/ct.mod/ct.pow`）。具体签名与语义见源码 `src/cuda/tile/_stub.py`。

位运算也支持 `&`/`|`/`^`、`<<`/`>>` 与 `ct.bitwise_*` 系列（详见 `src/cuda/tile/_stub.py` 与 `test/test_binary_elementwise.py`）。

注意：

- 位逻辑（`&`/`|`/`^`）与位移（`<<`/`>>`）仅适用于整数/布尔类型；任一操作数为浮点将报错（见 `test/test_binary_elementwise.py`）。
- 位逻辑的两个操作数必须具有相同 dtype；数组与标量混用时，标量按 int32 参与类型检查（见 `test/test_binary_elementwise.py`）。
- 位移操作的移位量在测试中使用非负范围（low=0, high=8），作为实践建议（见 `test/test_binary_elementwise.py`）。

# 形状广播

二元运算支持形状广播，规则和 NumPy 一致（同 `broadcast_to` 章节所述）。标量也可以参与运算。

# dtype 提升

当两个操作数的 dtype 不同时，会按照算术提升规则确定结果的 dtype。

类别优先级：boolean < integral < floating-point。

较高类别的类型优先。如果两个操作数都是整数类型，或者都是浮点类型，会按照位宽和符号性选择更宽的类型。

宽松常量（字面量）的具体化规则：

整型字面量根据值的范围具体化为 int32、int64 或 uint64。

浮点字面量具体化为 float32。

（示例与提升表参见 `docs/source/data.rst:Arithmetic Promotion`）

注意：

- 某些 dtype 组合（例如 float16 与 bfloat16 的组合）在多种二元运算（如加/除/幂/取模）中不支持隐式提升，需先显式转换到 float32 再运算（见 `test/test_binary_elementwise.py`）。
- 最终结果以 cuTile 数据模型中的“算术提升表”为准（类别比较 + 宽松常量具体化 + 提升表）。

# 一元数学函数

cuTile 提供了一系列一元数学函数：

| 函数 | 说明 |
| --- | --- |
| ct.exp(x) | 指数函数 e^x |
| ct.exp2(x) | 2^x |
| ct.log(x) | 自然对数 |
| ct.log2(x) | 以 2 为底的对数 |
| ct.sqrt(x) | 平方根 |
| ct.rsqrt(x) | 平方根的倒数 1/√x |
| ct.sin(x) | 正弦 |
| ct.cos(x) | 余弦 |
| ct.tan(x) | 正切 |
| ct.sinh(x) | 双曲正弦 |
| ct.cosh(x) | 双曲余弦 |
| ct.tanh(x) | 双曲正切 |
| ct.floor(x) | 向下取整 |
| ct.ceil(x) | 向上取整 |
| ct.negative(x) | 取负 |

还有两个二元的 min/max 函数：

（逐元素 min/max 见 `src/cuda/tile/_stub.py` 的 `minimum/maximum`）

提示：`ct.minimum/ct.maximum` 为逐元素运算，区别于按轴规约的 `ct.min/ct.max`。

# 比较运算

比较运算返回布尔语义的 Tile：

```
mask = (j * TILE_N + ct.arange(TILE_N, dtype=ct.int32)) < N

```

函数形式对应 `ct.greater/greater_equal/less/less_equal/equal/not_equal`（签名见 `src/cuda/tile/_stub.py`）。

比较运算也支持广播和 dtype 提升。返回的 bool tile 常用于 ct.where 做条件选择：

```
centered_tx = ct.where(mask, tx - mean, 0)

```

# 规约操作

cuTile 提供六个规约函数，用于在指定轴上聚合数据：

| 函数 | 说明 |
| --- | --- |
| ct.sum | 求和 |
| ct.max | 最大值 |
| ct.min | 最小值 |
| ct.prod | 求积 |
| ct.argmax | 最大值的索引 |
| ct.argmin | 最小值的索引 |

基本用法：

```
mean = ct.sum(mean, axis=1) / N
var = ct.sum(var, axis=1) / N

```

# axis 参数

`axis` 指定在哪个轴上做规约。规约后该轴会被消除（除非用 `keepdims=True`）。

`axis` 可以是单个整数：

（单轴规约示例参见 `test/test_reduction.py`）

也可以是元组，同时在多个轴上规约：

（多轴规约示例参见 `test/test_reduction.py`）

`axis=None` 表示在所有轴上规约，结果是 0D tile：

（`axis=None` 的 0D 结果用法参见 `test/test_reduction.py`）

说明：如需将 0D tile 的结果写回数组，可先用 `item()` 提取标量，或将其 reshape 成带维度的 tile 后再 `store`。

负数索引也支持，`axis=-1` 表示最后一个轴：

（负数索引的规约示例参见 `test/test_reduction.py`）

# keepdims 参数

`keepdims=True` 保留被规约的轴，但大小变为 1：

（keepdims 的行为参见 `test/test_reduction.py`）

`keepdims=True` 在需要和原 tile 做广播运算时很有用，比如计算 softmax：

（softmax 的 keepdims 用法参阅 `samples/AttentionFMHA.py`）

# argmax 和 argmin

ct.argmax 和 ct.argmin 返回最大值/最小值的索引，返回类型为 int32（IR 使用 default_int_type；见 `src/cuda/tile/_ir/ops.py` 与 `src/cuda/tile/_datatype.py`；测试验证见 `test/test_reduction.py`）。

# rounding_mode 和 flush_to_zero

ct.sum 和 ct.prod 支持额外的数值控制参数：

`rounding_mode` 控制舍入模式，可选值包括：

| RoundingMode | 含义 |
| --- | --- |
| RN | nearest even（默认） |
| RZ | toward zero |
| RM | toward negative infinity |
| RP | toward positive infinity |

（rounding_mode 的语义与可用枚举参见 `test/test_reduction.py:test_reduce_sumprodf_rounding_mode`）

`flush_to_zero` 控制是否将非常小的浮点数刷为零，只对 float32 类型有效：

（flush_to_zero 的行为参见 `test/test_reduction.py:test_reduce_flush_to_zero`）

如果对非 float32 类型使用 `flush_to_zero=True`，会抛出 `TileTypeError`。

补充说明：

- 规约算子中，rounding_mode 适用于 ct.sum/ct.prod 且仅支持 RN/RZ/RM/RP；RZI/FULL/APPROX 会报错。
- ct.max/ct.min 支持 `flush_to_zero`（同样仅对 float32 生效）。
- `axis`、`keepdims`、`flush_to_zero` 需为编译期常量。

# 扫描操作：cumsum 和 cumprod

ct.cumsum 和 ct.cumprod 是前缀和/前缀积操作：

（cumsum/cumprod 的用法参见 `src/cuda/tile/_stub.py` 与 `test/test_scan.py`）

`axis` 参数指定在哪个轴上做扫描，默认是 0。

`reverse` 参数控制扫描方向：

（reverse 参数用法参见 `test/test_scan.py`）

扫描操作也支持 `rounding_mode` 和 `flush_to_zero` 参数，规则和规约一样（见 `test/test_scan.py`）。

# matmul：通用矩阵乘法

`ct.matmul` 是通用的矩阵乘法函数：

注意

- 以下示例均为 tile 代码，仅可在 `@ct.kernel` 的 kernel 函数体内使用。
- `a`/`b` 应为 Tile（例如经 `ct.load`/`ct.full`/`ct.ones` 等获得），不能直接传 Array。

支持的输入 dtype 包括：float16、bfloat16、float32、float64、tfloat32、float8_e4m3fn、float8_e5m2、int8、uint8。`matmul` 的一个重要特性是会做 dtype 提升：如果 `x` 和 `y` 的 dtype 不同，会先提升到公共 dtype，结果的 dtype 和提升后的类型相同。

和 `matmul` 的关键区别：`mma` 不做输入 dtype 提升。`x` 和 `y` 的 dtype 必须相同，否则会抛出 `TileTypeError`。`mma` 保持 `acc` 的 dtype。结果的 dtype 由 `acc` 决定，不是由输入决定。

支持的 dtype 组合：

| 输入 dtype | acc/输出 dtype |
| --- | --- |
| float16 | float16 或 float32 |
| bfloat16 | float32 |
| float32 | float32 |
| float64 | float64 |
| tfloat32 | float32 |
| float8_e4m3fn | float16 或 float32 |
| float8_e5m2 | float16 或 float32 |
| int8 | int32 |
| uint8 | int32 |

`mma` 最常见的用法是在分块矩阵乘中做累加，完整示例见前文“典型用法”章节的 `matmul_kernel`。

# 注意

- `tm`/`tn`/`tk` 必须是编译期常量（`ct.Constant[int]`）且各维为 2 的幂。
- `ct.num_tiles` 的 `axis`/`shape`/`order` 参数必须为编译期常量。

# TF32 路径

对于 float32 输入，可以先将 tile 转为 tfloat32，再使用 `mma` 以利用 Tensor Core：

```
@ct.kernel
def mma_tf32_kernel(A, B, C,
                    tm: ct.Constant[int],
                    tn: ct.Constant[int],
                    tk: ct.Constant[int]):
    tx = ct.load(A, index=(0, 0), shape=(tm, tk)).astype(ct.tfloat32)
    ty = ct.load(B, index=(0, 0), shape=(tk, tn)).astype(ct.tfloat32)
    acc = ct.load(C, index=(0, 0), shape=(tm, tn))
    acc = ct.mma(tx, ty, acc)
    ct.store(C, index=(0, 0), tile=acc)

```

说明

- tfloat32 属于“受限浮点”（RestrictedFloat）；示例中通过显式 `astype` 使用 TF32（`mma`）；`matmul` 是否自动提升到 TF32，源码文档未明确说明。 参见 samples/LayerNorm.py 中 `layer_norm_bwd_dx_partial_dwdb` 函数的自旋锁实现。

# atomic_cas：compare-and-swap

`atomic_cas` 的用法示例如下：

```
@ct.kernel
def atomic_cas(x, y, z, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    offset = ct.arange(TILE, dtype=ct.int64)
    offset += bid*TILE
    cmp = ct.gather(x, offset)
    val = ct.gather(y, offset)
    old_val = ct.atomic_cas(x, offset, cmp, val,
                            memory_order=ct.MemoryOrder.ACQ_REL,
                            memory_scope=ct.MemoryScope.DEVICE)
    ct.scatter(z, offset, old_val)

```

# 其他 read-modify-write 操作

`atomic_add`、`atomic_max`、`atomic_min`、`atomic_and`、`atomic_or`、`atomic_xor` 的签名都类似： 每个函数都是读取当前值，执行对应操作，写回新值，返回旧值。

补充

- 当 `check_bounds=True` 且索引越界时，RMW 类原子（如 `atomic_add`/`atomic_max` 等）不执行写入，返回值为实现自定义；`atomic_cas` 则返回 `expected`。
- 位运算原子（`atomic_and`/`atomic_or`/`atomic_xor`）：`update` 的 dtype 必须与 `array` 的 dtype 完全一致，且 `array` 不能为浮点类型。
- 算术原子（`atomic_add`/`atomic_max`/`atomic_min` 等）：`update` 的 dtype 需能隐式转换到 `array` 的 dtype，否则会抛出类型错误。

# memory_order 和 memory_scope

所有原子操作都有 `memory_order` 和 `memory_scope` 参数：`memory_order` 控制内存操作的顺序保证：

| MemoryOrder | 含义 |
| --- | --- |
| RELAXED | 无顺序保证 |
| ACQUIRE | 后续读写不能重排到此操作之前 |
| RELEASE | 之前的读写不能重排到此操作之后 |
| ACQ_REL | ACQUIRE + RELEASE（默认） |
| memory_scope 控制同步的可见范围： |  |
| MemoryScope | 含义 |
| ------------- | ------ |
| BLOCK | 同一 block 内 |
| DEVICE | 同一 GPU 设备（默认） |
| SYS | 整个系统 |

提示

- Acquire/Release 的可见性与重排保证需与“成功获取/释放”路径配对使用；常见用法为“加锁用 ACQUIRE”、“解锁用 RELEASE/ACQ_REL”。
- 默认 `memory_scope=DEVICE`；跨设备/宿主同步需 `SYS`，但会带来更高的开销与语义差异。

# 实际例子：用原子锁做同步

LayerNorm 的反向传播中，多个 block 需要累加部分梯度，用 `atomic_cas` 和 `atomic_xchg` 实现自旋锁： 参见 samples/LayerNorm.py 中 `layer_norm_bwd_dx_partial_dwdb` 函数的自旋锁实现。 这里用 `ACQUIRE` 确保临界区的操作不会被重排到获取锁之前，用 `RELEASE` 确保临界区的操作不会被重排到释放锁之后。

注意

- 样例中 `Locks` 使用 `int32`；具体 dtype 需符合所用原子操作的支持范围与隐式转换规则。
- `TILE_N` 必须为编译期常量，且为 2 的幂。

# 支持的 Python 子集

cuTile 的 tile code 支持 Python 的一个子集。控制流方面，`if`、`for`、`while` 都可以用，并且可以嵌套。

# range 的限制

下例展示真实源码中的 `for range(...)` 用法（节选自 samples/MatMul.py）：

```
    for k in range(num_tiles_k):
        # Load tile from matrix A.
        # The `index=(bidx, k_tile_idx)` specifies which (M-tile, K-tile) to load
        # from global memory A. `shape=(tm, tk)` defines the size of this tile.
        a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=zero_pad).astype(dtype)

        # Load tile from matrix B.
        # The `index=(k_tile_idx, bidy)` specifies which (K-tile, N-tile) to load
        # from global memory B. `shape=(tk, tn)` defines the size of this tile.
        b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=zero_pad).astype(dtype)

        # Perform Matrix Multiplication for the current tiles.
        # `ct.mma` computes the product of the two loaded tiles and accumulates the result.

```

`range` 的 step 参数必须为正数。负步长不支持。另，通过变量间接传入负步长同样不支持，可能导致未定义行为。

# 不支持的特性

下例展示真实源码中的 `while` 用法（节选自 samples/LayerNorm.py）：

```
        while ct.atomic_cas(Locks, group_bid_m, 0, 1, memory_order=ct.MemoryOrder.ACQUIRE) == 1:
            pass


```

tile code 不支持以下 Python 特性：

异常处理（try/except/finally）

协程（async/await）

动态属性添加

lambda 表达式（lambda）

未在当前文档中列为受支持的其他特性（tile 代码无 Python 运行时）

# 对象不可变性

tile code 中创建的对象都是不可变的。任何看起来像"修改"的操作实际上都会返回一个新对象：

这意味着你不能"原地修改"一个 tile 的某个元素。每次操作都是创建新 tile。

# Array 是唯一的可变对象

Array 是 tile code 中唯一的可变对象。但 Array 必须作为 kernel 参数从外部传入，不能在 tile code 内部创建。

通过 ct.store、ct.scatter、原子操作等修改 Array 的内容。

补充：gather/scatter 的负索引被视为越界（不采用 Python 负索引语义），可通过 `padding_value` 或 `check_bounds` 控制行为。

# 别名约束

传给 kernel 的多个 Array 参数不能互相别名，也就是说它们不能指向同一块内存：

该约束需要调用方自行保证；否则行为不保证。

# 生命周期约束

所有传入 kernel 的 Array 在 kernel 执行完成之前必须保持有效。是否需要在 host 侧进行同步取决于具体场景；在需要时可调用框架提供的同步接口（例如 `torch.cuda.synchronize()` 或相应 `stream.synchronize()`）确保执行时序：

```
torch.cuda.synchronize()

```

示例：

```
ct.launch(torch.cuda.current_stream(), (M,), layer_norm_fwd,
          (x, weight, bias, y, mean, rstd, eps, TILE_N))

```

```
ct.launch(cp.cuda.get_current_stream(),
          grid,
          vector_add,
          (a, b, c, tile_size))

```

# 类型一致性（全局）

类型检查在编译期进行，若算子或数据类型不符合预期，可能抛出 `TileTypeError`。若需要改变类型，可显式使用 `ct.astype`。

# DType 体系

cuTile 定义了一套完整的数据类型系统，通过 `ct.DType` 访问。

# 整数类型

有符号整数：

| 类型 | 位宽 |
| --- | --- |
| ct.int8 | 8 位 |
| ct.int16 | 16 位 |
| ct.int32 | 32 位 |
| ct.int64 | 64 位 |

无符号整数：

| 类型 | 位宽 |
| --- | --- |
| ct.uint8 | 8 位 |
| ct.uint16 | 16 位 |
| ct.uint32 | 32 位 |
| ct.uint64 | 64 位 |

# 浮点类型

标准浮点：

| 类型 | 位宽 | 说明 |
| --- | --- | --- |
| ct.float16 | 16 位 | 半精度 |
| ct.float32 | 32 位 | 单精度 |
| ct.float64 | 64 位 | 双精度 |

特殊浮点：

| 类型 | 说明 |
| --- | --- |
| ct.bfloat16 | 1 个符号位、8 个指数位、7 个尾数位 |
| ct.tfloat32 | 19 位表示存于 32 位容器 |
| ct.float8_e4m3fn | 8 位浮点，4 位指数 3 位尾数 |
| ct.float8_e5m2 | 8 位浮点，5 位指数 2 位尾数 |

# 布尔类型

注意是 `bool_` 带下划线，避免和 Python 内置的 `bool` 冲突。

# 机器表示

cuTile 的类型和 CUDA C++ 的类型有相同的机器表示。比如 `ct.float16` 和 CUDA C++ 的 `__half` 是同一种二进制格式。

# 默认类型

当需要具体化宽松常量时：整型字面量根据值的范围具体化为 int32、int64 或 uint64；浮点字面量具体化为 float32。

补充：在 tile 代码中，`Array.shape` 与 `Array.strides` 的元素类型默认是 int32，且默认不视为常量。

# 类型分类

DType 有几个分类，按优先级从低到高：

Boolean < Integral < Floating-point

其中 Floating-point 又分为普通浮点（float16/32/64、bfloat16）和受限浮点（tfloat32、float8 系列）。

# 算术提升规则

当两个不同 dtype 的操作数进行运算时，会按照算术提升规则确定结果类型：

较高类别优先。如果一个是 boolean 一个是 integral，结果是 integral。如果一个是 integral 一个是 floating-point，结果是 floating-point。

同类别时：如果一方是宽松常量（字面量），采用另一方的 dtype。否则按照内置的类型提升表选择更宽的类型。

宽松常量具体化规则：

整型字面量根据值的范围具体化为 int32、int64 或 uint64。

浮点字面量具体化为 float32。

# RoundingMode

某些操作（如 ct.sum、ct.truediv）支持指定舍入模式：

| RoundingMode | 含义 |
| --- | --- |
| ct.RoundingMode.RN | round to nearest even |
| ct.RoundingMode.RZ | round toward zero |
| ct.RoundingMode.RM | round toward negative infinity |
| ct.RoundingMode.RP | round toward positive infinity |
| ct.RoundingMode.FULL | 完整精度 |
| ct.RoundingMode.APPROX | 近似计算 |
| ct.RoundingMode.RZI | round to nearest integer toward zero |

注意：`rounding_mode` 用于浮点运算；部分运算还支持 `flush_to_zero`（默认 False）。

# PaddingMode

ct.load 的 `padding_mode` 参数指定越界时的填充值：

| PaddingMode | 填充值 |
| --- | --- |
| ct.PaddingMode.UNDETERMINED | 未定义（默认） |
| ct.PaddingMode.ZERO | 0 |
| ct.PaddingMode.NEG_ZERO | -0 |
| ct.PaddingMode.NAN | NaN |
| ct.PaddingMode.POS_INF | +∞ |
| ct.PaddingMode.NEG_INF | -∞ |

# 使用示例

创建指定 dtype 的 tile：

```
tile = ct.full((4, 4), 3.14, dtype=ct.float32)

```

注意：Tile 的每个维度必须为 2 的幂（含 1）。

类型转换：

```
tx = ct.arange(8, dtype=ct.float32)
ty = ct.astype(tx, ct.float16)
ty.dtype
float16

```

# 框架互操作

cuTile 支持与 PyTorch、CuPy 互操作（通过 DLPack/CUDA Array Interface）。

# 直接传递 Tensor 和 ndarray

任何实现了 DLPack 或 CUDA Array Interface 的对象都可以直接作为 kernel 的 Array 参数（完整示例见开篇 `vector_add` 及前文“ct.launch”章节的 `test()` 函数）。

注意事项：

- 张量/ndarray 必须在 CUDA 设备上。
- 建议确保与传入的 stream 属于同一设备。
- 传入 kernel 的多个 Array 不应互相别名，且在 kernel 完成执行前必须保持有效。
- grid 必须是 1/2/3 维整数元组；1D 情况可以写成 `(g,)` 或 `(g, 1, 1)`。

# 流管理

`ct.launch` 的第一个参数是 CUDA stream，需要从你使用的框架获取：

| 框架 | 获取/创建流的方式（示例） |
| --- | --- |
| PyTorch | torch.cuda.current_stream() |
| CuPy | cp.cuda.get_current_stream() |
| Numba | numba.cuda.stream() |

使用框架的 stream 意味着 cuTile kernel 会和框架的其他操作在同一个流上按顺序入队执行。以下示例展示了在不同框架中传递流对象或其指针，并在 kernel 之后通过同步验证结果：

```
import torch
from torch.testing import make_tensor
import cupy
import cuda.tile as ct


@ct.kernel
def array_copy_1d(x, y, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ct.store(y, index=(bid,), tile=tx)


def _test_stream(stream, sync):
    x = make_tensor(4096, dtype=torch.float32, device="cuda")
    y = torch.zeros_like(x)
    torch.cuda.synchronize()
    ct.launch(stream, (1,), array_copy_1d, (x, y, 4096))
    sync()
    torch.testing.assert_close(x, y)


# -- Test PyTorch Stream --
def test_torch_pass_stream():
    stream = torch.cuda.Stream()
    _test_stream(stream, stream.synchronize)


def test_torch_pass_stream_ptr():
    stream = torch.cuda.Stream()
    _test_stream(stream.cuda_stream, stream.synchronize)


# -- Test CuPy Stream --
def test_cupy_pass_stream():
    stream = cupy.cuda.Stream()
    _test_stream(stream, stream.synchronize)


def test_cupy_pass_stream_ptr():
    stream = cupy.cuda.Stream()
    _test_stream(stream.ptr, stream.synchronize)


# -- Test Numba Stream --
def test_numba_pass_stream(numba_cuda):
    stream = numba_cuda.stream()
    _test_stream(stream, stream.synchronize)


def test_numba_pass_stream_ptr(numba_cuda):
    stream = numba_cuda.stream()
    _test_stream(stream.handle.value, stream.synchronize)

```

补充说明：

- 可直接传递流对象或其原始指针（PyTorch/CuPy/Numba 均可）。
- 若使用不同的流，建议使用同步或事件确保依赖顺序。

# 机器表示一致性

cuTile 的函数、类型和对象具有与 CUDA C++ 实体对应的机器表示。例如：`cuda.tile.float16` 与 CUDA C++ `__half` 具有相同的机器表示。

# 与 PyTorch autograd 集成

cuTile kernel 可以和 PyTorch 的 autograd 系统集成，通过 `torch.autograd.Function` 包装：

```
class CuTileLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, eps):
        x = input.reshape(-1, input.shape[-1])
        y = torch.empty_like(x)
        M, _ = x.shape

        # Allocate temporary buffers for mean and reciprocal standard deviation
        mean = torch.empty(M, dtype=torch.float32, device=x.device)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)

        TILE_N = 1024
        # Launch the forward kernel with a 1D grid (M blocks)
        ct.launch(torch.cuda.current_stream(), (M,), layer_norm_fwd,
                  (x, weight, bias, y, mean, rstd, eps, TILE_N))

        # Save tensors needed for the backward pass
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.TILE_N = TILE_N

        return y.reshape(*input.shape)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, mean, rstd = ctx.saved_tensors
        TILE_N = ctx.TILE_N
        M, N = x.shape
        GROUP_SIZE_M = 64

        # Flatten gradient output to (M, N)
        dy = grad_output.reshape(-1, grad_output.shape[-1])
        dx = torch.empty_like(dy)

        # Allocate buffers for partial gradients and synchronization locks
        dw = torch.zeros((GROUP_SIZE_M, N), dtype=torch.float32, device=weight.device)
        db = torch.zeros((GROUP_SIZE_M, N), dtype=torch.float32, device=bias.device)
        locks = torch.zeros(GROUP_SIZE_M, dtype=torch.int32, device=weight.device)

        # Launch the first backward kernel to compute dX and partial dW/dB
        ct.launch(torch.cuda.current_stream(), (M,), layer_norm_bwd_dx_partial_dwdb,
                  (dx, dy, dw, db, x, weight, mean, rstd, locks, TILE_N))

        final_dw = torch.empty((N,), dtype=weight.dtype, device=weight.device)
        final_db = torch.empty((N,), dtype=bias.dtype, device=bias.device)
        TILE_M = 32

        # Launch the second backward kernel to reduce partial dW/dB
        ct.launch(torch.cuda.current_stream(), (math.ceil(N / TILE_N),), layer_norm_bwd_dwdb,
                  (dw, db, final_dw, final_db, TILE_M, TILE_N))

        return dx.reshape(*grad_output.shape), final_dw, final_db, None

```

