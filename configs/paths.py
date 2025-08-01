"""config/paths.py"""

from utils.find_root_dir import find_project_root

# All paths are Path objects
ROOT_DIR = find_project_root()

CONFIGS_DIR = ROOT_DIR / "configs"
PROMPTS_CONFIG = CONFIGS_DIR / "prompts_config.yaml"
MODEL_CONFIGS_YAML_FILE = CONFIGS_DIR / "model_configs.yaml"

PROMPTS_DIR = ROOT_DIR / "prompts"
AGENT_PROMPTS = PROMPTS_DIR / "agent_prompts.yaml.j2"
