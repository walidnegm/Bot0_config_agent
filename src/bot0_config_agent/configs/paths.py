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

# ------------------------------- Pydantic models Directory -------------------------------

MODELS_DIR = BOT0_CONFIG_AGENT_DIR / "models"

PROMPTS_DIR = BOT0_CONFIG_AGENT_DIR / "prompts"
AGENT_PROMPTS = PROMPTS_DIR / "agent_prompts.yaml.j2"

# ------------------------------- Tools directory -------------------------------

TOOLS_DIR = BOT0_CONFIG_AGENT_DIR / "tools"

# tools/configs directory
TOOLS_CONFIGS_DIR = TOOLS_DIR / "configs"
TOOL_MODELS_MODULE = TOOLS_CONFIGS_DIR / "tool_models.py"
TOOL_TRANSFORMATION_JSON_FILE = TOOLS_CONFIGS_DIR / "tool_transformation.json"
TOOL_TRANSFORMATION_FUNCTIONS_MODULE = TOOLS_CONFIGS_DIR / "tool_transformation.py"
TOOL_REGISTRY_JSON_FILE = TOOLS_CONFIGS_DIR / "tool_registry.json"

# tools/registry directory
TOOLS_REGISTRY_DIR = TOOLS_DIR / "registry"
TOOLS_MANIFEST_YAML_FILE = TOOLS_REGISTRY_DIR / "_manifest.yaml"
TOOL_TRANSFORMATION_JSON_FILE = TOOLS_CONFIGS_DIR / "tool_transformation.json"
TOOL_TRANSFORMATION_FUNCTIONS_MODULE = TOOLS_CONFIGS_DIR / "tool_transformation.py"

###########################
OFFLOAD_DIR = ROOT_DIR / "offload"
