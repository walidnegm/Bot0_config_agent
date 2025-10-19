"""
tools/tool_registry.py
MCP tool that lists all tools currently discoverable via the MCP session.
"""

from mcp.server.lowlevel.server import active_server

def get_tool_definition():
    return {
        "name": "tool_registry",
        "description": "Lists all tools currently registered with the MCP server.",
        "inputSchema": {"type": "object", "properties": {}},
    }

def run(params=None):
    if not active_server or not hasattr(active_server, "registered_tools"):
        return {"status": "error", "message": "No active server context", "result": None}

    tools = [
        {"name": t.name, "description": getattr(t, "description", "")}
        for t in active_server.registered_tools.values()
    ]

    return {
        "status": "success",
        "message": f"{len(tools)} tools registered with MCP.",
        "result": tools,
    }

