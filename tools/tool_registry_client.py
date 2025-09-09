import asyncio
import argparse
import json
from contextlib import AsyncExitStack
from typing import Optional, Dict, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    async def connect_to_server(self, server_script_path: str):
        command = "python"
        server_params = StdioServerParameters(command=command, args=[server_script_path], env=None)
        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            read, write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await self.session.initialize()
        except Exception as e:
            raise ValueError(f"Failed to connect to server at '{server_script_path}': {e}")

    async def list_schemas(self):
        if not self.session:
            raise ValueError("Not connected to a server")
        response = await self.session.list_tools()
        for tool in response.tools:
            print(f"Tool: {tool.name}")
            print(f"Description: {tool.description}")
            schema_data = tool.schema() if callable(tool.schema) else tool.schema
            print(f"Schema: {json.dumps(schema_data, indent=2)}")
            print("-" * 40)

    async def call_tool(self, tool_name: str, params: Dict[str, Any]):
        if not self.session:
            raise ValueError("Not connected to a server")
        result = await self.session.call_tool(tool_name, params)
        return result.content

    async def close(self):
        await self.exit_stack.aclose()

async def main(args):
    client = MCPClient()
    try:
        await client.connect_to_server(args.server_path)
        
        if args.list_schemas:
            await client.list_schemas()
        elif args.tool:
            try:
                params = json.loads(args.params) if args.params else {}
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON for --params. Example: '{\"root\": \".\"}'")
            result = await client.call_tool(args.tool, params)
            # Handle potential ContentBlock / TextContent from SDK
            if isinstance(result, list):
                text_result = "\n".join([block.text for block in result if hasattr(block, 'text')])
                print(f"Result from {args.tool} tool:\n{text_result}")
            elif hasattr(result, 'text'):
                print(f"Result from {args.tool} tool:\n{result.text}")
            elif isinstance(result, Dict):
                print(f"Result from {args.tool} tool:\n{json.dumps(result, indent=2)}")
            else:
                print(f"Result from {args.tool} tool:\n{str(result)}")
        else:
            print("Use --list-schemas or --tool <name> --params <json>")
    finally:
        await client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP Client CLI")
    parser.add_argument("--list-schemas", action="store_true", help="List all tools and their schemas")
    parser.add_argument("--tool", help="Tool name to execute (e.g., 'find_dir_size')")
    parser.add_argument("--params", default="{}", help="JSON params (e.g., '{\"root\": \".\"}')")
    parser.add_argument("--server-path", default="tool_registry_mcp.py", help="Path to MCP server script")
    args = parser.parse_args()
    asyncio.run(main(args))
