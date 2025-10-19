"""
CLI script for Bot0 Config Agent (pure MCP design, fixed placeholder extraction)
"""

import sys
import os
import asyncio
from pathlib import Path
import argparse
import logging
import json
import yaml
from typing import List, Dict, Any

# -------------------------------------------------------------------------
# Project import setup
# -------------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent.core import AgentCore
from tools.tool_models import ToolResult, ToolResults
from utils.get_llm_api_keys import (
    get_openai_api_key,
    get_anthropic_api_key,
    get_google_api_key,
)
from configs.api_models import get_llm_provider
from configs.paths import MODEL_CONFIGS_YAML_FILE
from loaders.load_model_config import load_model_config
from agent.mcp_client import mcp_client_context

# -------------------------------------------------------------------------
# Pretty console
# -------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.markdown import Markdown

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = lambda: None


# -------------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# MCP Tool Adapter
# -------------------------------------------------------------------------
class MCPToolAdapter:
    """Adapter between AgentCore and MCP server."""

    def __init__(self, session):
        self.session = session

    async def get_all_tools(self) -> List[Dict[str, Any]]:
        tools_resp = await self.session.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema.get("properties", {}),
            }
            for tool in tools_resp.tools
        ]

    async def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = await self.session.call_tool(tool_name, params)
            content = getattr(result, "content", result)
            return {
                "step_id": "mcp_step",
                "tool": tool_name,
                "params": params,
                "message": str(content),
                "status": "success",
                "result": content,
                "state": "completed",
            }
        except Exception as e:
            logger.exception(f"Tool {tool_name} failed: {e}")
            return {
                "step_id": "mcp_step",
                "tool": tool_name,
                "params": params,
                "message": f"❌ MCP tool execution failed: {e}",
                "status": "error",
                "result": None,
                "state": "failed",
            }


# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------
def get_available_models(config_file: Path = MODEL_CONFIGS_YAML_FILE) -> List[str]:
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f) or {}
        return list(config.keys())
    except Exception as e:
        logger.error(f"Failed to load model configs from {config_file}: {e}")
        return []

def display_result(result_data: dict, console: Console = None):
    """Pretty-print tool results with smart unwrapping and parsing."""

    import ast, json, textwrap
    import ast, json, textwrap
    if RICH_AVAILABLE:
        from rich.markdown import Markdown
    else:
        Markdown = None
    
    def normalize(obj):
        if hasattr(obj, "text") and isinstance(obj.text, str):
            return obj.text
        if isinstance(obj, dict):
            return {k: normalize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [normalize(x) for x in obj]
        return obj

    safe_result = normalize(result_data.get("result"))
    safe_message = normalize(result_data.get("message"))

    # ✅ Attempt to parse escaped dict/JSON strings for prettier display
    def try_parse(value):
        if not isinstance(value, str):
            return value
        try:
            # handle Python-style dicts with single quotes
            return ast.literal_eval(value)
        except Exception:
            try:
                return json.loads(value)
            except Exception:
                return value  # fallback to raw string

    safe_result = try_parse(safe_result)
    safe_message = try_parse(safe_message)

    # 🖼️ Rich console formatting
    if console and RICH_AVAILABLE:
        console.print(f"\n🔧 [bold cyan]Tool:[/bold cyan] {result_data.get('tool')}")
        console.print(f"🗨️ [dim]Message:[/dim] {safe_message}")
        console.print("📌 [bold]Result:[/bold]")
        if isinstance(safe_result, (dict, list)):
            console.print(Markdown(f"```json\n{json.dumps(safe_result, indent=2, ensure_ascii=False)}\n```"))
        else:
            console.print(textwrap.fill(str(safe_result), width=100))
    else:
        print(f"\n🔧 Tool: {result_data.get('tool')}")
        print(f"🗨️ Message: {safe_message}")
        print("📌 Result:")
        if isinstance(safe_result, (dict, list)):
            print(json.dumps(safe_result, indent=2, ensure_ascii=False))
        else:
            print(textwrap.fill(str(safe_result), width=100))

# -------------------------------------------------------------------------
# Core MCP execution flow
# -------------------------------------------------------------------------
async def run_agent_with_mcp(
    instruction: str, agent: AgentCore, mcp_adapter: MCPToolAdapter, console: Console = None
):
    """Executes the planned MCP tool chain for a given user instruction."""

    tools_for_prompt = await mcp_adapter.get_all_tools()
    plan = await agent.planner.plan_async(instruction)
    logger.info(f"Planned steps: {len(plan.steps) if hasattr(plan, 'steps') else 'N/A'}")

    results: Dict[str, Any] = {}

    # --- Helper for unwrapping TextContent, dicts, JSON, etc.
    
    
    def unwrap_value(val):
        """Flatten nested TextContent, dicts, and text blobs into usable values."""
        import re, json, ast

        # Handle TextContent
        if hasattr(val, "text"):
            val = val.text

        # Handle lists
        if isinstance(val, list):
            return [unwrap_value(v) for v in val]

        # Handle dicts
        if isinstance(val, dict):
            return json.dumps(val, indent=2)

        # Handle strings
        if isinstance(val, str):
            # ✅ Detect literal list as string (e.g., "['a.yaml', 'b.yaml']")
            if val.strip().startswith("[") and val.strip().endswith("]"):
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    pass

            # ✅ Detect "result=['a.yaml','b.yaml']" pattern
            m = re.search(r"result=\[([^\]]+)\]", val)
            if m:
                inner = m.group(1)
                files = [
                    f.strip().strip("'\"")
                    for f in inner.split(",")
                    if f.strip().strip("'\"")
                ]
                return files

            # ✅ Detect JSON-like outputs from read_files
            if '"files"' in val or "'files'" in val:
                try:
                    parsed = json.loads(val.replace("'", '"'))
                    if isinstance(parsed, dict) and "result" in parsed:
                        files = parsed.get("result", {}).get("files", [])
                        if isinstance(files, list):
                            return "\n\n".join(f.get("content", "") for f in files)
                except Exception:
                    pass

        return val


    # --- Execute planned steps sequentially
    for i, step in enumerate(getattr(plan, "steps", [])):
        tool_name = step.tool
        params = step.params

        # 🔁 Resolve <step_n> placeholders dynamically
        for k, v in params.items():
            if isinstance(v, str) and v.startswith("<step_"):
                ref_val = unwrap_value(results.get(v.strip("<>"), ""))
                params[k] = ref_val
            elif isinstance(v, list):
                flattened = []
                for x in v:
                    resolved = (
                        unwrap_value(results.get(x.strip("<>"), x))
                        if isinstance(x, str) and x.startswith("<step_")
                        else x
                    )
                    if isinstance(resolved, list):
                        flattened.extend(resolved)
                    else:
                        flattened.append(resolved)
                params[k] = flattened
        for k, v in list(params.items()):
            if isinstance(v, list) and len(v) == 1 and isinstance(v[0], list):
                params[k] = v[0]
        
        # 🧠 NEW: Flatten file dicts → readable text (for LLM summary steps)
        for k, v in list(params.items()):
            if isinstance(v, list) and all(isinstance(x, dict) and "content" in x for x in v):
                params[k] = "\n\n---\n\n".join(
                    f"## {x.get('path')}\n{x.get('content')}" for x in v
                )
            elif isinstance(v, dict) and "content" in v:
                params[k] = f"## {v.get('path')}\n{v.get('content')}"
        
        for k, v in params.items():
            if isinstance(v, str) and "<step_" in v:
                for key, val in results.items():
                    if key in v:
                        unwrapped = unwrap_value(val)
                        if isinstance(unwrapped, list) and all(
                            isinstance(x, dict) and "content" in x for x in unwrapped
                        ):
                            file_text = "\n\n---\n\n".join(
                                f"## {x.get('path')}\n{x.get('content')}" for x in unwrapped
                            )
                            v = v.replace(f"<{key}>", file_text)
                        else:
                            v = v.replace(f"<{key}>", str(unwrapped))
                params[k] = v

        result_dict = await mcp_adapter.execute_tool(tool_name, params)
        display_result(result_dict, console)
        results[f"step_{i}"] = result_dict.get("result")

    # ✅ Build ToolResults
    tool_result_objects = [
        ToolResult(
            step_id=k,
            tool="mcp",
            params={},
            status="success",
            message=str(v),
            result=v,
        )
        for k, v in results.items()
    ]

    return ToolResults(results=tool_result_objects)


# -------------------------------------------------------------------------
# CLI Entry Point
# -------------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(description="Bot0 Config Agent CLI (pure MCP mode)")
    parser.add_argument("--local-model", help="Specify local model name")
    parser.add_argument("--api-model", help="Specify API model (e.g., gpt-4.1-mini)")
    parser.add_argument("--once", help="Run a single instruction and exit")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    console = Console() if RICH_AVAILABLE else None

    # --- Validate API model ---
    if args.api_model:
        provider = get_llm_provider(args.api_model)
        key_fetchers = {
            "openai": get_openai_api_key,
            "anthropic": get_anthropic_api_key,
            "gemini": get_google_api_key,
        }
        key = key_fetchers.get(provider, lambda: None)()
        if not key:
            logger.error(f"{provider.title()} API key missing in .env.")
            sys.exit(1)
        logger.info(f"Validated {provider} API key for model '{args.api_model}'.")

    # --- Validate local model ---
    if args.local_model:
        try:
            load_model_config(args.local_model)
        except Exception as e:
            logger.error(f"Invalid local model '{args.local_model}': {e}")
            sys.exit(1)

    # --- Start MCP client-server session ---
    mcp_server_script = Path(project_root) / "mcp_server.py"
    if not mcp_server_script.exists():
        logger.error(f"MCP server not found: {mcp_server_script}")
        sys.exit(1)

    server_params = ["python", str(mcp_server_script)]
    async with mcp_client_context(server_params) as session:
        logger.info("[MCP Client] ✅ Connected and initialized.")
        mcp_adapter = MCPToolAdapter(session)

        # ✅ Dynamically fetch tool inventory from MCP (authoritative source)
        tools_resp = await session.list_tools()
        tools = [t.model_dump() if hasattr(t, "model_dump") else t for t in tools_resp.tools]
        logger.info(f"MCP connected: {len(tools)} tools available.")

    # ✅ Inject these tools into your planner prompt dynamically
    from prompts.load_agent_prompts import load_planner_prompts
    planner_prompts = load_planner_prompts(
        user_task=args.once or "",
        tools=tools
    )

    # 🔧 Initialize the agent AFTER prompts are ready
    agent = AgentCore(
        local_model_name=args.local_model,
        api_model_name=args.api_model,
        planner_prompts=planner_prompts,  # ✅ pass to AgentCore if supported
    )

    if args.once:
        instruction = args.once.strip()
        logger.info("=" * 50)
        logger.info(f"User Instruction: {instruction}")
        try:
            await run_agent_with_mcp(instruction, agent, mcp_adapter, console)
            logger.info("=" * 50)
        except Exception as e:
            logger.error(f"Execution failed: {e}", exc_info=True)
            if console and RICH_AVAILABLE:
                console.print(f"❌ [bold red]Error:[/bold red] {e}")
            else:
                print(f"❌ Error: {e}")
    else:
        print("Interactive mode not implemented. Use --once.")


if __name__ == "__main__":
    asyncio.run(main())

