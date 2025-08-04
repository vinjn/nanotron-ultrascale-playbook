# VLLM中模型加载的逻辑代码分析

**Author:** running

**Date:** 2025-05-20

**Link:** https://zhuanlan.zhihu.com/p/1908118779134710858

1.模型加载的入口

前面的加载过程不在赘述，模型的加载的入口在[multiproc\_executor](https://zhida.zhihu.com/search?content_id=257957494&content_type=Article&match_order=1&q=multiproc_executor&zhida_source=entity).py文件下的[workerProc](https://zhida.zhihu.com/search?content_id=257957494&content_type=Article&match_order=1&q=workerProc&zhida_source=entity)类中，此类定义了当前的推理运行方式是多进程执行器（executor）。

![](https://pic1.zhimg.com/v2-1a112c0519f3e9cb9c8274ba5ffb6eb0_1440w.jpg)

这段代码中创建了多个worker进程，会有多层嵌套，最后来到WorkerProc类中

![](https://pic1.zhimg.com/v2-4e25ed2ccc42e7e30e7115d6b7013296_1440w.jpg)

在此处进行模型加载

![](https://picx.zhimg.com/v2-dc4255c838fee0a2ae79c2a019665879_1440w.jpg)

在进行模型加载过程中，会从worker进入。worker的类型分很多种，分为gpu\_worcker, cpu\_worker, tpu\_work等多种，在使用[vllm server](https://zhida.zhihu.com/search?content_id=257957494&content_type=Article&match_order=1&q=vllm+server&zhida_source=entity)之后，[config.py](https://zhida.zhihu.com/search?content_id=257957494&content_type=Article&match_order=1&q=config.py&zhida_source=entity)文件中会探测当前平台是哪一个，直接确定worker的类型，传入到vllm\_config参数字典中。这里有一个类方法的转发，用来处理self.worker.load\_model()方法：

![](https://pica.zhimg.com/v2-343b660a39d7713465754ef8d2467654_1440w.jpg)

进入模型加载代码：通过[gpu\_worker](https://zhida.zhihu.com/search?content_id=257957494&content_type=Article&match_order=1&q=gpu_worker&zhida_source=entity).py代码进入

![](https://pic2.zhimg.com/v2-1086da2f70d138504b1f635cb382e975_1440w.jpg)

模型加载由[model\_runner](https://zhida.zhihu.com/search?content_id=257957494&content_type=Article&match_order=1&q=model_runner&zhida_source=entity).py执行：

![](https://pica.zhimg.com/v2-55b9178b1bf5ba61189b5a25340334a2_1440w.jpg)

加载过程中会实例化一个loader，loader会选择不同的模型文件格式的加载方式。

模型加载：

![](https://picx.zhimg.com/v2-8fe3dcd2c67344cefa5b0f02f9d8a6bf_1440w.jpg)