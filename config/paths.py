"""config/paths.py"""

from utils.find_root_dir import find_project_root

# All paths are Path objects
ROOT_DIR = find_project_root()

CONFIG_DIR = ROOT_DIR / "config"
PROMPTS_CONFIG = CONFIG_DIR / "prompts_config.yaml"
MODEL_CONFIG_YAML_FILE = CONFIG_DIR / "model_configs.yaml"

PROMPTS_DIR = ROOT_DIR / "prompts"
TOOL_AGENT_PROMPTS = PROMPTS_DIR / "tool_agent_prompt.yaml"
