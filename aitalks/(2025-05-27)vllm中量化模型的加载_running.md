# vllm中量化模型的加载

**Author:** running

**Date:** 2025-05-27

**Link:** https://zhuanlan.zhihu.com/p/1908199116741255546

1.[量化模型](https://zhida.zhihu.com/search?content_id=257973171&content_type=Article&match_order=1&q=%E9%87%8F%E5%8C%96%E6%A8%A1%E5%9E%8B&zhida_source=entity)加载

加载入口在[model\_runner.py](https://zhida.zhihu.com/search?content_id=257973171&content_type=Article&match_order=1&q=model_runner.py&zhida_source=entity)中的load\_model()方法中。通过创建loader实例来进行模型的加载。

![](https://pic4.zhimg.com/v2-253b6407089bbe5b834455e67d118a41_1440w.jpg)

get\_mode()方法会再次创建一个loader类，此类是模型加载的主体，除了特殊的模型格式外，一般常用的都是默认的模型加载类class [DefaultModelLoader](https://zhida.zhihu.com/search?content_id=257973171&content_type=Article&match_order=1&q=DefaultModelLoader&zhida_source=entity)(BaseModelLoader)，使用load\_model()方法加载模型

![](https://pic2.zhimg.com/v2-ad0d2c15fd760773a73e993a94818e53_1440w.jpg)

图2.loader中的模型加载

在此方法中，会使用model = \_initialize\_model(vllm\_config=vllm\_config)方法先获取对应的模型的结构（torch.nn.module）

![](https://pic4.zhimg.com/v2-048ac8e526f89821e76dbc5c24f1248f_1440w.jpg)

在标注位置会获取模型的结构文件，模型的结构文件会在vllm\\model\_executor\\models\\[registry.py](https://link.zhihu.com/?target=http%3A//registry.py/)文件中已经加载。ModelRegistry.resolve\_model\_cls(architectures)会找到对应架构的.py文件，把相应的类导入为model\_class（nn.module模块），用于加载权重前的实例化推理模型结构。

![](https://pic4.zhimg.com/v2-2cbb7c0219cc25a559db10aa7e1a130d_1440w.jpg)

在这里会判断使用量化方法后，使用的map方法。

之后，就是使用

![](https://pic3.zhimg.com/v2-5ccda7a6dc0a9b2760d26e415eba6e32_1440w.jpg)

图2中的绿框

get\_all\_weights(）进行模型参数的加载。

在参数加载过程中，get\_all\_weights(）会使用\_prepare\_weights(）和\_get\_weights\_iterator(）方法来加载模型权重。

需要注意的是，这里的model.load\_weights()不是torch中的load\_weight，而且每个模型架构中自己封装的加载方式，用于生成器形式的模型加载。vllm\\model\_executor\\models\\[utils.py](https://link.zhihu.com/?target=http%3A//utils.py)文件中的[AutoWeightsLoader](https://zhida.zhihu.com/search?content_id=257973171&content_type=Article&match_order=1&q=AutoWeightsLoader&zhida_source=entity)类。