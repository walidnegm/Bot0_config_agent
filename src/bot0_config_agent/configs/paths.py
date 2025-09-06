"""config/paths.py"""

from bot0_config_agent.utils.system.find_root_dir import find_project_root

# All paths are Path objects
ROOT_DIR = find_project_root()
BOT0_CONFIG_AGENT_DIR = ROOT_DIR / "src" / "bot0_config_agent"

CONFIGS_DIR = BOT0_CONFIG_AGENT_DIR / "configs"
PROMPTS_CONFIG = CONFIGS_DIR / "prompts_config.yaml"
MODEL_CONFIGS_YAML_FILE = CONFIGS_DIR / "model_configs.yaml"
LOCAL_OVERRIDE_MODEL_CONFIGS_YAML_FILE = (
    CONFIGS_DIR / "overrides" / "model_configs.local.yaml"
)

PROMPTS_DIR = BOT0_CONFIG_AGENT_DIR / "prompts"
AGENT_PROMPTS = PROMPTS_DIR / "agent_prompts.yaml.j2"

TOOLS_DIR = BOT0_CONFIG_AGENT_DIR / "tools"
WORKBENCH_DIR = TOOLS_DIR / "workbench"
TOOL_REGISTRY = WORKBENCH_DIR / "tool_registry.json"
TOOL_TRANSFORMATION = WORKBENCH_DIR / "tool_transformation.json"
TOOL_TRANSFORMATION_FUNCTIONS = WORKBENCH_DIR / "tool_transformation.py"
