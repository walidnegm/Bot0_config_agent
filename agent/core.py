# agent/core.py
from agent.executor import ToolExecutor
from tools.tool_registry import ToolRegistry
from agent.planner import Planner

class AgentCore:
    def __init__(self):
        self.registry = ToolRegistry()
        self.planner = Planner(self.registry)
        self.executor = ToolExecutor()

    def run(self, instruction: str):
        print(f"[AgentCore] Received instruction: {instruction}")
        plan = self.planner.plan(instruction)
        return self.executor.execute_plan(plan)

