# `MCP`教程

## 基本概念

`MCP` (Model Context Protocol) 翻译为中文是`模型上下文协议`，但是这个翻译谁听了都迷糊。实际上`MCP`就是一个能让大模型更好的使用工具的协议，看到这里，很多人就认为这个是大模型和工具交互的协议，**事实上这个协议是`agent`和`工具`之间交互的协议**。

如果你不明白`agent`是什么，那么可以简单看一下[Agent从入门到放弃](./Agent从入门到放弃.md)这篇文章，这篇文章中写了一个例子，这篇文章中使用的`工具`及其相关使用文档是直接写死在`agent`中，如果说后续新增一个`工具`就需要修改`agent`中的代码，那就无法实现解耦，所以最好的办法就是搞一个协议，让其`agent`和`工具`自动交换信息那就完美了，于是`MCP`就应运而生，当然这里说的`MCP`交互实际上是`MCP Client`和`MCP Server`交互的协议，而一个`MCP Host`可以包含多个`MCP Client`，也就是说`agent`就是一个`MCP Host`。

## `MCP`架构

`MCP`架构：

- `MCP Host`：协调和管理一个或多个 MCP 客户端的 AI 应用程序
- `MCP 客户端`：维护与 MCP 服务器的连接并从 MCP 服务器获取上下文以供 MCP 主机使用的组件
- `MCP 服务器`：为 MCP 客户端提供上下文的程序，可以在本地或远程执行。

`MCP`包含两层：

- 数据层：定义基于 JSON-RPC 的客户端-服务器通信协议，包括生命周期管理，以及核心原语，如工具、资源、提示和通知。
  - 工具：AI 应用程序可以调用以执行操作的可执行函数（例如文件操作、API 调用、数据库查询）
  - 资源：向 AI 应用程序提供上下文信息的数据源（例如文件内容、数据库记录、API 响应）
  - 提示：可重复使用的模板，有助于构建与语言模型的交互（例如，系统提示、少量示例）
- 传输层：定义实现客户端和服务器之间数据交换的通信机制和渠道，包括传输特定的连接建立、消息框架和授权。MCP 支持两种传输机制：
  - Stdio 传输：使用标准输入/输出流在同一台机器上的本地进程之间进行直接进程通信，提供最佳性能且无网络开销，常用。
  - 可流式传输的 HTTP 传输：使用 HTTP POST 协议发送客户端到服务器的消息，并可选用服务器发送事件来实现流式传输功能。

`MCP`全量代码参考[MCP 代码](../mcp)，这里只介绍主要部分。

## 建构`MCP 服务器`

`MCP 服务器`写起来非常简单，有点像`fastapi`，一个`tool`注解`@mcp.tool`。

```python
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("weather")

@mcp.tool()
async def your_tools_function(para1: str, para2: float) -> str:
    pass

if __name__ == "__main__":
    mcp.run(transport='stdio')
```

## `MCP` 配置文件

```json
{
  "mcpServers": {
    "weather": {
      "timeout": 60,
      "command": "uv",
      "args": [
        "--directory",
        "your_program_dir\\weather_mcp_server",
        "run",
        "weather.py"
      ],
      "transportType": "stdio"
    }
  }
}
```

## 建构`MCP 客户端`

大部分的交互`mcp SDK`已经封装好了，调用时只需要使用`ClientSession`的`call_tool`函数即可。

```python

from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 读取 MCP 配置文件
mcp_server_name = key
tool_config = servers.get(key)
command = tool_config['command']
args = tool_config['args']

server_params = StdioServerParameters(
    command=command,
    args=args,
    env=None
)
stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
self.stdio, self.write = stdio_transport
session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

# 初始化之后就已经连接上
await session.initialize()
# 调用 MCP 服务器函数
await session.call_tool(tool_name, tool_args)
```
