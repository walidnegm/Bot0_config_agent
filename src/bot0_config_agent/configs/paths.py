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

MODELS_DIR = BOT0_CONFIG_AGENT_DIR / "models"

PROMPTS_DIR = BOT0_CONFIG_AGENT_DIR / "prompts"
AGENT_PROMPTS = PROMPTS_DIR / "agent_prompts.yaml.j2"

TOOLS_DIR = BOT0_CONFIG_AGENT_DIR / "tools"
TOOL_CONFIGS_FILE = TOOLS_DIR / "configs"
TOOL_REGISTRY = TOOL_CONFIGS_FILE / "tool_registry.json"
TOOL_TRANSFORMATION = TOOL_CONFIGS_FILE / "tool_transformation.json"
TOOL_TRANSFORMATION_FUNCTIONS = TOOL_CONFIGS_FILE / "tool_transformation.py"

###########################
OFFLOAD_DIR = ROOT_DIR / "offload"
