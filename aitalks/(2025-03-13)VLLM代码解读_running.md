# VLLM代码解读

**Author:** running

**Date:** 2025-03-13

**Link:** https://zhuanlan.zhihu.com/p/29385205755

对vllm源码进行一个简单的解读，结合copilot

1.vllm server启动命令的进入点：vllm\\vllm\\entrypoints\\cli\\[main.py](https://link.zhihu.com/?target=http%3A//main.py/)

实现方式：在[pyproject.toml](file:///d%3A/project/vllm/pyproject.toml)文件中，有一个关键的配置项

\[project.scripts\]

vllm = "vllm.entrypoints.cli.main:main"

  

vllm serve 命令执行流程

1\. 命令行参数解析阶段

当执行 vllm serve model-path 命令时：

入口点在 scripts.py 中的 main() 函数，它调用 vllm.entrypoints.cli.main.main()。在 main.py 中的 main() 函数解析命令行参数。main() 函数加载 vllm.entrypoints.cli.serve 模块，识别 serve 子命令

创建 ServeSubcommand 类的实例并调用其 cmd() 方法

2\. Server 初始化阶段

在 ServeSubcommand.cmd() 方法中：

处理命令行参数，将 model\_tag 作为 model 参数

调用 [uvloop.run](https://link.zhihu.com/?target=http%3A//uvloop.run)(run\_server(args)) 运行服务器

3\. 引擎启动阶段

在 run\_server() 函数中（位于 api\_server.py）：

创建服务器套接字以绑定端口

使用 build\_async\_engine\_client() 异步上下文管理器创建 engine\_client：

```text
async with build_async_engine_client(args) as engine_client:
    # 创建并初始化 FastAPI 应用
    # ...
```

在 build\_async\_engine\_client() 根据配置，选择：

如果需要多进程前端或设置了特定配置，创建 MQLLMEngineClient

否则，创建 AsyncLLMEngine.from\_engine\_args()

4\. 模型加载阶段

在 AsyncLLMEngine.from\_engine\_args() 方法中：

  

创建 engine\_config 配置，包含模型配置、并行配置等

确定执行器类型（通过 \_get\_executor\_cls() 方法）

创建 AsyncLLMEngine 实例，它内部包含 \_AsyncLLMEngine（继承自 LLMEngine）

在 AsyncLLMEngine 初始化时：

创建 \_AsyncLLMEngine 实例

设置请求处理回调

在 start\_engine\_loop 设为 True 时（默认），启动后台循环处理请求

5\. Server 启动和路由设置阶段

回到 run\_server() 函数：

调用 build\_app() 创建 FastAPI 应用

调用 init\_app\_state() 初始化应用状态：

在 init\_app\_state() 中：

```text
await init_app_state(engine_client, model_config, app.state, args)
```

创建 [OpenAIServingModels](https://zhida.zhihu.com/search?content_id=254899458&content_type=Article&match_order=1&q=OpenAIServingModels&zhida_source=entity) 实例

根据模型类型创建各种服务组件：

[OpenAIServingChat](https://zhida.zhihu.com/search?content_id=254899458&content_type=Article&match_order=1&q=OpenAIServingChat&zhida_source=entity)（用于聊天生成）

[OpenAIServingCompletion](https://zhida.zhihu.com/search?content_id=254899458&content_type=Article&match_order=1&q=OpenAIServingCompletion&zhida_source=entity)（用于文本补全）

[OpenAIServingPooling](https://zhida.zhihu.com/search?content_id=254899458&content_type=Article&match_order=1&q=OpenAIServingPooling&zhida_source=entity)（用于嵌入池化）

[OpenAIServingEmbedding](https://zhida.zhihu.com/search?content_id=254899458&content_type=Article&match_order=1&q=OpenAIServingEmbedding&zhida_source=entity)（用于嵌入生成）

[OpenAIServingTokenization](https://zhida.zhihu.com/search?content_id=254899458&content_type=Article&match_order=1&q=OpenAIServingTokenization&zhida_source=entity)（用于分词）

其他特定服务组件

6\. API 服务器运行阶段

完成初始化后：

调用 serve\_http() 启动 HTTP 服务器

监听指定端口，处理用户请求

当接收到用户请求时（如 /v1/chat/completions），调用相应的路由处理函数

7\. 请求处理阶段

当用户发送请求时：

API 请求经过 FastAPI 路由到相应的处理函数（如 create\_chat\_completion）

处理函数调用相应服务组件（如 OpenAIServingChat.create\_chat\_completion）

服务组件通过 AsyncLLMEngine 将请求添加到引擎中：

```text
generator = await handler.create_chat_completion(request, raw_request)
```

AsyncLLMEngine 将请求添加到内部队列，并启动异步处理

引擎的后台循环会持续处理请求队列中的任务

8\. 模型推理阶段

在引擎的 run\_engine\_loop 方法中：

engine\_step 方法调用底层 \_AsyncLLMEngine 的 step\_async 方法

模型根据输入生成输出

输出通过异步流方式返回给用户

总结

vllm serve 命令执行流程是一个从命令行解析，到服务器初始化，再到模型加载和API服务提供的完整过程。核心组件是 AsyncLLMEngine，它负责协调用户请求与模型推理之间的交互，并通过异步方式高效处理并发请求。

关键函数调用链：

  

vllm/scripts.py: main() → vllm/entrypoints/cli/main.py: main()

ServeSubcommand.cmd() → [uvloop.run](https://link.zhihu.com/?target=http%3A//uvloop.run)(run\_server(args))

run\_server() → build\_async\_engine\_client() → AsyncLLMEngine.from\_engine\_args()

build\_app() + init\_app\_state() → 创建和配置FastAPI应用

serve\_http() → 启动HTTP服务器监听请求

请求处理 → 路由到相应处理函数 → 通过 AsyncLLMEngine 处理