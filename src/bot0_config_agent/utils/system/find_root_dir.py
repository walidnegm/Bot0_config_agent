from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def find_project_root(
    markers=(".git", "requirements.txt", "pyproject.toml", "README.md")
) -> Path:
    """Search up parent directories for a project root marker file."""
    path = Path(__file__).resolve().parent
    for parent in [path] + list(path.parents):
        if any((parent / marker).exists() for marker in markers):
            logger.info(f"🔍 Project root found: {parent}")
            return parent
    logger.warning("⚠️ Project root not found, using CWD.")
    return Path.cwd()
