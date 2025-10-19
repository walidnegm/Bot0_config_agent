import asyncio
from contextlib import AsyncExitStack
from typing import Optional, Dict, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server via STDIO."""
        command = "python"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        read, write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("Connected to server with tools:", [tool.name for tool in tools])

    async def call_tool(self, tool_name: str, params: Dict[str, Any]):
        """Call a specified tool on the server with given parameters."""
        if not self.session:
            raise ValueError("Not connected to a server")

        result = await self.session.call_tool(tool_name, params)
        return result.content  # Access the tool's output

    async def close(self):
        await self.exit_stack.aclose()

async def main():
    client = MCPClient()
    try:
        await client.connect_to_server("mcp_server.py")  # Path to your server script (update if renamed)
        
        # Invoke 'find_dir_size' with appropriate params (root defaults to "." if omitted)
        tool_name = "find_dir_size"
        params = {"root": "."}  # Example: Current directory; change to another path as needed
        result = await client.call_tool(tool_name, params)
        print(f"Result from {tool_name} tool: {result}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
