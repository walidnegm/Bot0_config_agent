from enum import Enum


class StepStatus(str, Enum):
    """
    Enumeration of possible status for each step in an agent tool chain.
    """

    SUCCESS = "success"  # Step completed successfully
    ERROR = "error"  # Step execution failed
