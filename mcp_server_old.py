import os
import importlib
from mcp.server.fastmcp import FastMCP

# Initialize the FastMCP server
mcp = FastMCP("Dynamic Tools Server")

# Directory containing tool modules
tools_dir = 'tools'

# Dynamically load and register tools from the tools/ directory
for filename in os.listdir(tools_dir):
    if filename.endswith('.py') and not filename.startswith('__') and not filename.startswith('tool_'):
        module_name = filename[:-3]  # Strip .py
        try:
            module = importlib.import_module(f"{tools_dir}.{module_name}")
            # Assume the tool function is named after the module (snake_case)
            tool_func_name = module_name.replace('-', '_')  # Handle any hyphens, though none in your ls
            tool_func = getattr(module, tool_func_name, None)
            if callable(tool_func):
                mcp.tool()(tool_func)  # Register the function as an MCP tool
                print(f"Registered tool: {module_name}")
            else:
                print(f"No callable tool function '{tool_func_name}' found in {module_name}")
        except ImportError as e:
            print(f"Failed to import {module_name}: {e}")

# Run the server in STDIO mode (for simplicity; can be adapted for HTTP)
if __name__ == "__main__":
    mcp.run(transport='stdio')
