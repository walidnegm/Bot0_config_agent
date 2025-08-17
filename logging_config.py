"""logging_config.py"""

import logging
import logging.handlers
import os
from pathlib import Path
import getpass  # To get the username
from datetime import datetime  # To add date and time to the log file name


def find_project_root(
    starting_path=None,
    marker=(".git", "requirements.txt", "pyproject.toml", "README.md"),
):
    """
    Recursively find the root directory of the project by looking for
    a specific marker.

    Args:
        starting_path (str or Path): The starting path to begin the search.
            Defaults to the current script's directory.
        marker (str): The marker to look for (e.g., '.git', 'setup.py', 'README.md').

    Returns:
        Path: The Path object pointing to the root directory of the project,
        or None if not found.
    """
    # Start from the directory of the current file if not specified
    if starting_path is None:
        starting_path = Path(__file__).resolve().parent

    # Convert starting_path to a Path object if it's not already
    starting_path = Path(starting_path)

    # Traverse up the directory tree
    for parent in [starting_path] + list(starting_path.parents):
        # Check if the marker exists in the current directory
        markers = marker if isinstance(marker, (list, tuple)) else [marker]
        for m in markers:
            if (parent / m).exists():
                return parent

    return None  # Return None if the marker is not found


def get_log_file_path(logs_dir):
    """
    Constructs the log file path with username, date, and time.

    Args:
        logs_dir (str): Directory path for logs.

    Returns:
        str: Log file path with username, date, and time.
    """
    username = getpass.getuser()
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(logs_dir, f"{username}_{current_time}_app.log")
    return log_file_path


# Ensure logs directory exists
root_dir = find_project_root()

if root_dir is None:
    raise RuntimeError(
        "Project root not found. Ensure the marker (e.g., .git) is present."
    )

logs_dir = os.path.join(root_dir, "logs")

# Ensure the logs directory exists
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
    logging.info(f"Created logs directory: {logs_dir}")

# Set up log file path with username, date, and time
log_file_path = get_log_file_path(logs_dir)

# Initialize the rotating file handler
file_handler = logging.handlers.RotatingFileHandler(
    log_file_path,
    maxBytes=100 * 1024 * 1024,
    backupCount=5,
)

# Configure file handler log format and level
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)

# Create a console handler with a specific log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(console_formatter)

# Get the root logger and attach handlers directly
root_logger = logging.getLogger()

# Add both the file handler and console handler to the root logger
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Set the overall logging level (root level)
root_logger.setLevel(logging.DEBUG)


def log_and_flush(message: str, level: str = "info") -> None:
    """
    Log a message and immediately flush the log handlers.

    Args:
        message (str): The log message.
        level (str): Log level ('info', 'debug', 'warning', 'error',
            'critical').
    """
    log_func = getattr(logging, level.lower(), logging.info)
    log_func(message)

    # Explicitly flush each handler
    for handler in logging.getLogger().handlers:
        handler.flush()
