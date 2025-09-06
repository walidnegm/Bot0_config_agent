from enum import Enum


class StepState(str, Enum):
    """
    Enumeration of possible execution states for each step in an agent tool chain.
    """

    PENDING = "pending"  # Have not yet started
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
