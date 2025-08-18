import asyncio
from typing import Optional
from contextlib import AsyncExitStack
import ast
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent
from openai import OpenAI
from dotenv import load_dotenv
import json
import os
from prompt_template import react_system_prompt_template
from string import Template
import re
from typing import List, Callable, Tuple, Any
from datetime import date


load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self.tools = {}  # function -> mcp_name
        self.available_tools = []
        self.llm_model = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('BASE_URL')
        )

    async def connect_to_server(self, server_script_path: str):
        """连接 MCP 服务器

        Args:
            server_script_path: mcp 服务器配置文件
        """
        with open(server_script_path, mode='r') as f:
            servers = json.load(f)['mcpServers']

            for key in servers.keys():
                # 解析配置文件获取mcp的配置信息
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
                self.session[mcp_server_name] = session

                # 调用相关接口获取信息
                response = await session.list_tools()
                tools = response.tools
                for tool in tools:
                    self.tools[tool.name] = mcp_server_name
                self.tools[mcp_server_name] = tools
                self.available_tools.extend([{
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                } for tool in tools])

        for key in self.tools.keys():
            print(f"\nMCP服务: {key} 包括的工具: ",  self.tools[key])

    async def process_query(self, query: str) -> str:
        """使用大模型和工具处理用户查询请求"""

        messages: List[dict[str, Any]]=[
            {"role": "system", "content": self.render_system_prompt(react_system_prompt_template)},
            {"role": "user", "content": query}
        ]

        while True:
            content = self.call_model(messages)
            print(content)

            # Process response and handle tool calls
            tool_results = []
            final_text = []

            if "<final_answer>" in content:
                final_answer = re.search(r"<final_answer>(.*?)</final_answer>", content, re.DOTALL)
                if final_answer:
                    final_text.append(final_answer.group(1))
                    break

            # 检测 Action
            action_match = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
            if not action_match:
                raise RuntimeError("模型未输出 <action>")
            action = action_match.group(1)
            tool_name, args = self.parse_action(action)
            tool_args = {}
            for arg in args:
                s = arg.split("=")
                tool_args[s[0]] = s[1].strip('"')

            # Execute tool call
            session = self.session[self.tools[tool_name]]
            result = await session.call_tool(tool_name, tool_args)
            tool_results.append({"call": tool_name, "result": result})
            final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
            text = f"<observation>工具返回结果：{result.content[0].text}</observation>" if type(result.content[0]) is TextContent else "<observation>工具执行错误</observation>"

            print("工具返回结果：",text)

        return "\n".join(final_text)

    def call_model(self, messages):
        response = self.llm_model.chat.completions.create(
            model=os.getenv('MODEL_NAME', default='deepseek-chat'),
            messages=messages,
            temperature=0,
            max_tokens=1024,
        )
        if response.choices[0].message.content:
            content = response.choices[0].message.content
            messages.append({"role": "assistant", "content": content})
            return content
        return "模型未返回任何信息"

    async def chat_loop(self):
        """智能助手"""
        print("输入问题或quit退出程序")

        while True:
            try:
                query = input("\nQuestion: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

    def render_system_prompt(self, system_prompt_template: str) -> str:
        return Template(system_prompt_template).substitute(
            tool_list=self.available_tools,
            current_date=date.today()
        )

    def parse_action(self, code_str: str) -> Tuple[str, List[str]]:
        match = re.match(r'(\w+)\((.*)\)', code_str, re.DOTALL)
        if not match:
            raise ValueError("Invalid function call syntax")

        func_name = match.group(1)
        args_str = match.group(2).strip()

        # 手动解析参数，特别处理包含多行内容的字符串
        args = []
        current_arg = ""
        in_string = False
        string_char = None
        i = 0
        paren_depth = 0
        
        while i < len(args_str):
            char = args_str[i]
            
            if not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    string_char = char
                    current_arg += char
                elif char == '(':
                    paren_depth += 1
                    current_arg += char
                elif char == ')':
                    paren_depth -= 1
                    current_arg += char
                elif char == ',' and paren_depth == 0:
                    # 遇到顶层逗号，结束当前参数
                    args.append(self._parse_single_arg(current_arg.strip()))
                    current_arg = ""
                else:
                    current_arg += char
            else:
                current_arg += char
                if char == string_char and (i == 0 or args_str[i-1] != '\\'):
                    in_string = False
                    string_char = None
            
            i += 1
        
        # 添加最后一个参数
        if current_arg.strip():
            args.append(self._parse_single_arg(current_arg.strip()))
        
        return func_name, args
    
    def _parse_single_arg(self, arg_str: str):
        """解析单个参数"""
        arg_str = arg_str.strip()
        
        # 如果是字符串字面量
        if (arg_str.startswith('"') and arg_str.endswith('"')) or \
           (arg_str.startswith("'") and arg_str.endswith("'")):
            # 移除外层引号并处理转义字符
            inner_str = arg_str[1:-1]
            # 处理常见的转义字符
            inner_str = inner_str.replace('\\"', '"').replace("\\'", "'")
            inner_str = inner_str.replace('\\n', '\n').replace('\\t', '\t')
            inner_str = inner_str.replace('\\r', '\r').replace('\\\\', '\\')
            return inner_str
        
        # 尝试使用 ast.literal_eval 解析其他类型
        try:
            return ast.literal_eval(arg_str)
        except (SyntaxError, ValueError):
            # 如果解析失败，返回原始字符串
            return arg_str

async def main():
    if len(sys.argv) < 2:
        print("Usage: client.py <file_to_mcp_server_config>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())
