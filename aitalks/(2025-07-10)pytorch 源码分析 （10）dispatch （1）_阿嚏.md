# pytorch 源码分析 （10）dispatch （1）

**Author:** 阿嚏

**Date:** 2025-07-10

**Link:** https://zhuanlan.zhihu.com/p/1926690924211189607

[pytorch](https://zhida.zhihu.com/search?content_id=260177009&content_type=Article&match_order=1&q=pytorch&zhida_source=entity)作为一个多平台框架，针对不同的场景往往有不同的实现代码，一般pytorch的[dispatch](https://zhida.zhihu.com/search?content_id=260177009&content_type=Article&match_order=1&q=dispatch&zhida_source=entity)是指的设备级别的dispatch,也就是根据是不是cuda还是cpu选择不同的实现函数，但是今天主要分析的是类型的dispatch。

这一部分的核心内容在[aten/src/ATen/Dispatch.h](https://zhida.zhihu.com/search?content_id=260177009&content_type=Article&match_order=1&q=aten%2Fsrc%2FATen%2FDispatch.h&zhida_source=entity)之中，几乎全是宏，看代码是有些费劲的。

[AT\_DISPATCH\_SWITCH](https://zhida.zhihu.com/search?content_id=260177009&content_type=Article&match_order=1&q=AT_DISPATCH_SWITCH&zhida_source=entity) 和 [AT\_DISPATCH\_CASE](https://zhida.zhihu.com/search?content_id=260177009&content_type=Article&match_order=1&q=AT_DISPATCH_CASE&zhida_source=entity)是最重要的两个基宏

AT\_DISPATCH\_FLOATING\_TYPES是其中的一种用法，用于分发浮点数类型，下面是一个示例

可以看出，需要的是tensor的类型，函数名，和一个匿名函数, 最后运行的是histogramdd\_kernel\_impl，其中多了一个scalar\_t，这个是分发的类型。

```text
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "histogram_mps", [&]() {
    mps::histogramdd_kernel_impl<scalar_t, bin_algorithm>(hist, bin_edges_contig, reshaped_input, reshaped_weight);
  });
```

为什么有类型scalar\_type了，还需要dispatch类型scalar\_t，看一下这个scalar\_type  

```text
  inline ScalarType toScalarType() {
    if (C10_LIKELY(isScalarType())) {
      return static_cast<ScalarType>(index_);
    }
    error_unsupported_typemeta(*this);
  } 

 inline ScalarType typeMetaToScalarType(caffe2::TypeMeta dtype) {
  return dtype.toScalarType();
  }

  const caffe2::TypeMeta dtype() const {
    return data_type_;
  }

  ScalarType scalar_type() const {
    return typeMetaToScalarType(impl_->dtype());
  }
```

可以看出，tensor的本身类型，是caffe2::TypeMeta，后续转化为ScalarType，这两者有什么区别呢？

TypeMeta是一个类，其有一个重要的成员变量uint16\_t index\_; 这个对应着下面要讲的ScalarType枚举，就是一个值，每个值对应着一种类型。除此之外，其还实现了很多方法，例如判断两个类型是不是一样：

```text
inline bool operator==(const TypeMeta& lhs, const TypeMeta& rhs) noexcept {
  return (lhs.index_ == rhs.index_);
}
```


ScalarType是一个枚举，每一种类型对应着一个数字，这个数字用int8\_t表示，这里只展示部分类型表示

```text
enum class ScalarType : int8_t {
#define DEFINE_ST_ENUM_VAL_(_1, n) n,
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_ST_ENUM_VAL_)
#undef DEFINE_ENUM_ST_ENUM_VAL_
      Undefined,
  NumOptions
};
```

例如（下面代码没有截全），ScalarType::Float 是一个int8\_t值，其值为6，但是其代表着float类型。

```text
// NB: Order matters for this macro; it is relied upon in
// _promoteTypesLookup and the serialization format.
#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(_) \
  _(uint8_t, Byte) /* 0 */                               \
  _(int8_t, Char) /* 1 */                                \
  _(int16_t, Short) /* 2 */                              \
  _(int, Int) /* 3 */                                    \
  _(int64_t, Long) /* 4 */                               \
  _(at::Half, Half) /* 5 */                              \
  _(float, Float) /* 6 */                                \
  _(double, Double) /* 7 */                              \
  _(c10::complex<c10::Half>, ComplexHalf) /* 8 */        \
  _(c10::complex<float>, ComplexFloat) /* 9 */           \
  _(c10::complex<double>, ComplexDouble) /* 10 */        \
  _(bool, Bool) /* 11 */                                 \
  _(c10::qint8, QInt8) /* 12 */                          \
  _(c10::quint8, QUInt8) /* 13 */                        \
  _(c10::qint32, QInt32) /* 14 */                        \
```

我们看一下一个使用示例：枚举的作用就是这样，如果拿字符串记录会有很多不必要开销，用纯数字记录又容易乱套，所以用枚举。

```text
  switch (tensor.scalar_type()) {
    case at::ScalarType::Half:
      TORCH_CHECK(n <= (int64_t(1) << 11) + 1, "n cannot be greater than 2049 for Half type.");
      break;
    case at::ScalarType::Float:
      TORCH_CHECK(n <= (int64_t(1) << 24) + 1, "n cannot be greater than 2^24+1 for Float type.");
      break;
    case at::ScalarType::Double:  // Unlikely to happen, but doesn't hurt to check
      TORCH_CHECK(n <= (int64_t(1) << 53) + 1, "n cannot be greater than 2^53+1 for Double type.");
      break;
    default:
      break;
  }
```

其实到这时就可以看出来，一个tensor的类型底层实现只是一个数字而已，并不是大家理解的float double这样，原因应该是pytorch更多的作用是给大家python接口，python的float和cpp的float又不是一回事，而且还有类似half等多种类型，所以在python接口上，设置类型只是设置了一个数字，真正需要对这个tensor进行计算的时候，再转化成真正的类型表示。 那么如何转化成大家熟悉使用的类型？就需要AT\_DISPATCH等各种宏了。

它们的源代码如下：

```text
#define AT_PRIVATE_CASE_TYPE_USING_HINT(enum_type, HINT, ...)           \
  case enum_type: {                                                     \
    AT_PRIVATE_CHECK_SELECTIVE_BUILD(enum_type);                        \
    using HINT C10_UNUSED = c10::impl::ScalarTypeToCPPTypeT<enum_type>; \
    return __VA_ARGS__();                                               \
  }

#define AT_DISPATCH_CASE(enum_type, ...) \
  AT_PRIVATE_CASE_TYPE_USING_HINT(enum_type, scalar_t, __VA_ARGS__)



#define AT_DISPATCH_SWITCH(TYPE, NAME, ...)                                 \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    constexpr const char* at_dispatch_name = NAME;                          \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                   \
    RECORD_KERNEL_FUNCTION_DTYPE(at_dispatch_name, _st);                    \
    switch (_st) {                                                          \
      __VA_ARGS__                                                           \
      default:                                                              \
        AT_ERROR(                                                           \
            '"',                                                            \
            at_dispatch_name,                                               \
            "\" not implemented for '",                                     \
            toString(_st),                                                  \
            "'");                                                           \
    }                                                                       \
  }()

#define AT_DISPATCH_CASE_FLOATING_TYPES(...)            \
  AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))
```

即便没有宏展开，也大体可以看出来，是将at::ScalarType::Float等，通过ScalarTypeToCPPTypeT转化为cpp类型，using HINT C10\_UNUSED = c10::impl::ScalarTypeToCPPTypeT<enum\_type>这个展开是

using scalar\_t C10\_UNUSED = c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::Float>

后续scalar\_t 就是float类型，可以直接用了，例如:

scalar\_t a = 1.0;

  

再看下ScalarTypeToCPPTypeT是如何工作的，发现还是AT\_FORALL\_SCALAR\_TYPES\_WITH\_COMPLEX\_AND\_QINTS这个宏，只是这次用来偏特化ScalarTypeToCPPType

```text
// NB: Order matters for this macro; it is relied upon in
// _promoteTypesLookup and the serialization format.
#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(_) \
  _(uint8_t, Byte) /* 0 */                               \
  _(int8_t, Char) /* 1 */                                \
  _(int16_t, Short) /* 2 */                              \
  _(int, Int) /* 3 */                                    \
  _(int64_t, Long) /* 4 */                               \
  _(at::Half, Half) /* 5 */                              \
  _(float, Float) /* 6 */                                \
  _(double, Double) /* 7 */                              \
  _(c10::complex<c10::Half>, ComplexHalf) /* 8 */        \
  _(c10::complex<float>, ComplexFloat) /* 9 */           \
  _(c10::complex<double>, ComplexDouble) /* 10 */        \
  _(bool, Bool) /* 11 */                                 \
  _(c10::qint8, QInt8) /* 12 */                          \
  _(c10::quint8, QUInt8) /* 13 */                        \
  _(c10::qint32, QInt32) /* 14 */                        \
  _(at::BFloat16, BFloat16) /* 15 */                     \

#define SPECIALIZE_ScalarTypeToCPPType(cpp_type, scalar_type)                \
  template <>                                                                \
  struct ScalarTypeToCPPType<c10::ScalarType::scalar_type> {                 \
    using type = cpp_type;                                                   \
                                                                             \
    /* This is a workaround for the CUDA bug which prevents */               \
    /* ::detail::ScalarTypeToCType<T>::type being used directly due to */    \
    /* ambiguous reference which can't to be resolved. For some reason it */ \
    /* can't pick between at::detail and at::cuda::detail. */                \
    /* For repro example, please see: */                                     \
    /* https://gist.github.com/izdeby/952ae7cf256ddb740a73776d39a7e7ba */    \
    /* TODO: remove once the bug is fixed. */                                \
    static type t;                                                           \
  };

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(SPECIALIZE_ScalarTypeToCPPType)


template <c10::ScalarType N>
using ScalarTypeToCPPTypeT = typename ScalarTypeToCPPType<N>::type;
```

举一个例子：从而也就将数字6（c10::ScalarType::Float）转为了类型float,用scalar\_t 来统一表示。

```text
struct ScalarTypeToCPPType<c10::ScalarType::Float> {
  using type = float;
  static type t;
}
```

AT\_DISPATCH\_FLOATING\_TYPES看着也就只能分发double和float两种类型，很多时候使用的类型不止这些，尤其是现在半精度比较流行，如何做：

下面看一下高端用法，例如AT\_DISPATCH\_FLOATING\_TYPES\_AND2

```text
      AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "host_softmax", [&] {
        using accscalar_t = acc_type<scalar_t, true>;
```

其实就相当于多了两个case，如果scalar\_type()匹配上这两个case，也可以分发类型：

```text
#define AT_DISPATCH_CASE_FLOATING_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, ...) \
  AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__)                              \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                                \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES_AND2(       \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                          \
      TYPE,                                    \
      NAME,                                    \
      AT_DISPATCH_CASE_FLOATING_TYPES_AND2(    \
          SCALARTYPE1, SCALARTYPE2, __VA_ARGS__))
```

注意到在scalar\_t基础上，还有一个accscalar\_t ，这个又是用来做什么的？

这个是计算过程中产生的一些中间计算量（累积）类型，是为了精度保证，比如softmax，数值全是bf16, 在计算sum exp的时候，如果还用bf16表示，误差会比较高，所以在一些关键计算量时，可能会用更高精度的数据来表示。

我把核心代码列到了下面，可以看出，对于半精度bf16 half，或者一些float8类型，都用float做累积类型，而float类型就用float本身，这里只列出了部分类型，详细可以参考aten/src/ATen/[AccumulateType](https://zhida.zhihu.com/search?content_id=260177009&content_type=Article&match_order=1&q=AccumulateType&zhida_source=entity).h

```text
CUDA_ACC_TYPE(BFloat16, float);
CUDA_ACC_TYPE(Half, float);
CUDA_ACC_TYPE(Float8_e5m2, float);
CUDA_ACC_TYPE(Float8_e4m3fn, float);
CUDA_ACC_TYPE(Float8_e5m2fnuz, float);
CUDA_ACC_TYPE(Float8_e4m3fnuz, float);
CUDA_ACC_TYPE(float, float);
CUDA_ACC_TYPE(double, double);

#define ACC_TYPE(t, acc_t, device_type)         \
  template <>                                   \
  struct AccumulateTypeDevice<t, device_type> { \
    using type = acc_t;                         \
  };
#define MPS_ACC_TYPE(t, acc_t) ACC_TYPE(t, acc_t, c10::DeviceType::MPS)
#define CUDA_ACC_TYPE(t, acc_t) ACC_TYPE(t, acc_t, c10::DeviceType::CUDA)
#define CPU_ACC_TYPE(t, acc_t) ACC_TYPE(t, acc_t, c10::DeviceType::CPU)

enum class DeviceType : int8_t {
  CPU = 0,
  CUDA = 1, // CUDA.
  MKLDNN = 2, // Reserved for explicit MKLDNN
  OPENGL = 3, // OpenGL
  OPENCL = 4, // OpenCL
  IDEEP = 5, // IDEEP.
  HIP = 6, // AMD HIP
  ... 省略一部分
}

template <typename T>
struct AccumulateType<T, true> {
  using type = typename AccumulateTypeDevice<T, c10::DeviceType::CUDA>::type;
};


template <typename T, bool is_cuda>
using acc_type = typename AccumulateType<T, is_cuda>::type;
```

本文讲解了pytorch中，类型dispatch的内容，有机会会分析一下pytorch设备dispatch的内容！