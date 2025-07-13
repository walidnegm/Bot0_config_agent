# agent/agent_core.py

from agent.executor import ToolExecutor
from agent.planner import Planner
from tools.tool_registry import ToolRegistry


class AgentCore:
    def __init__(self):
        print("[AgentCore] 🔧 Initializing ToolRegistry, Planner, and Executor…")
        self.registry = ToolRegistry()
        self.planner = Planner()
        self.executor = ToolExecutor()
        print("[AgentCore] ✅ Initialization complete.")

    def handle_instruction(self, instruction: str) -> list:
        print(f"\n🧠 [AgentCore] Received instruction:\n  → {instruction}")

        try:
            print("[AgentCore] 🧭 Calling planner.plan()…")
            plan = self.planner.plan(instruction)
            print("[AgentCore] ✅ Plan generated.")

            print("[AgentCore] 🚀 Executing plan:")
            results = self.executor.execute_plan(plan)
            print("[AgentCore] ✅ Execution complete.")
            return results

        except Exception as e:
            print(f"❌ [AgentCore] Planner error: {e}")
            return [{
                "tool": "planner",
                "status": "error",
                "message": str(e)
            }]

