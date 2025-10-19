"""
mcp_client.py
Creates and manages an MCP client connection to the Bot0 MCP Server.
Compatible with mcp >= 0.4.0 (official SDK).
"""
import logging
from contextlib import asynccontextmanager
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters

logger = logging.getLogger(__name__)

@asynccontextmanager
async def mcp_client_context(server_params):
    """
    Connects to an MCP server subprocess via stdio.
    Accepts either:
      - a StdioServerParameters object, OR
      - a list like ["python", "mcp_server.py"]
    """
    # Normalize params if given as a list
    if isinstance(server_params, (list, tuple)):
        if len(server_params) == 0:
            raise ValueError("Empty server_params list.")
        command = server_params[0]
        args = list(server_params[1:])
        server_params = StdioServerParameters(command=command, args=args)
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            logger.info("[MCP Client] ✅ Connected and initialized.")
            yield session
