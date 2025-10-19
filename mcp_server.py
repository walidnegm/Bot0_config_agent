"""
Bot0 MCP Server — Pure MCP design (no ToolRegistry)

• Dynamically discovers all MCP-compatible tools in `tools/`
  (those exposing get_tool_definition() + run()).
• Exposes them via the MCP protocol (mcp==1.16.0+).
"""

import asyncio
import importlib
import inspect
import logging
import pkgutil
from pathlib import Path

import mcp.types as types
from mcp.server.lowlevel import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio

# -------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

server = Server("Bot0 MCP Server")
TOOLS_PATH = Path(__file__).parent / "tools"


# -------------------------------------------------------------------------
# Tool Discovery (MCP-aware)
# -------------------------------------------------------------------------
def discover_tools():
    """
    Dynamically import modules in `tools/` and collect MCP-compatible tools.
    A valid tool must implement both `get_tool_definition()` and `run(params)`.
    """
    found = {}
    for _, module_name, _ in pkgutil.iter_modules([str(TOOLS_PATH)]):
        try:
            module = importlib.import_module(f"tools.{module_name}")
            if hasattr(module, "get_tool_definition") and hasattr(module, "run"):
                tool_def = module.get_tool_definition()
                name = tool_def.get("name", module_name)
                found[name] = {
                    "run": module.run,
                    "definition": tool_def,
                }
                logger.info(f"[MCP Server] Loaded tool: {name}")
            elif hasattr(module, "main") and callable(module.main):
                # Legacy fallback for old tools
                found[module_name] = {
                    "run": module.main,
                    "definition": {
                        "name": module_name,
                        "description": f"Legacy auto-discovered tool: {module_name}",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                }
                logger.info(f"[MCP Server] Loaded legacy tool: {module_name}")
        except Exception as e:
            logger.warning(f"[MCP Server] Failed to import tool {module_name}: {e}")
    return found


TOOLS = discover_tools()
logger.info(f"[MCP Server] ✅ Loaded {len(TOOLS)} tool(s): {', '.join(TOOLS.keys())}")


# -------------------------------------------------------------------------
# MCP: List Tools
# -------------------------------------------------------------------------
@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Return all available tools to the MCP client."""
    tools_list = []
    for name, meta in TOOLS.items():
        definition = meta["definition"]
        tools_list.append(
            types.Tool(
                name=definition.get("name", name),
                description=definition.get("description", f"MCP tool {name}"),
                inputSchema=definition.get("inputSchema", {"type": "object", "properties": {}}),
            )
        )
    return tools_list


# -------------------------------------------------------------------------
# MCP: Call Tool
# -------------------------------------------------------------------------
# ---------------------------------------------------------------------
# MCP: Call Tool
# ---------------------------------------------------------------------
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Execute a discovered MCP tool (async-safe)."""
    tool_meta = TOOLS.get(name)
    if not tool_meta:
        return [types.TextContent(type="text", text=f"❌ Unknown tool: {name}")]

    tool_func = tool_meta["run"]

    try:
        # Handle async and sync tool functions
        if inspect.iscoroutinefunction(tool_func):
            result = await tool_func(arguments)
        else:
            result = tool_func(arguments)

        # ✅ If a coroutine or Task is returned, await it here
        if asyncio.iscoroutine(result) or isinstance(result, asyncio.Task):
            result = await result

        return [types.TextContent(type="text", text=str(result))]

    except Exception as e:
        logger.exception(f"[MCP Server] Error running tool '{name}': {e}")
        return [
            types.TextContent(
                type="text",
                text=f"❌ Error running tool '{name}': {e}",
            )
        ]


# -------------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------------
async def main():
    logger.info("[MCP Server] Starting Bot0 MCP Server (stdio mode)…")
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="Bot0 MCP Server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())

