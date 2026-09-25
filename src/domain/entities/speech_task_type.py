"""Enumeration representing types of speech assessment tasks."""

from enum import Enum, auto


class SpeechTaskType(Enum):
    """Supported voice assessment protocol types."""

    READ_TEXT = auto()
    SPONTANEOUS_DIALOG = auto()
    SUSTAINED_VOWEL = auto()

    @classmethod
    def from_string(cls, task_name: str) -> "SpeechTaskType":
        """Convert a string representation to a SpeechTaskType enum member.

        Args:
            task_name: The name of the task as a string.

        Returns:
            The matching SpeechTaskType member.

        Raises:
            ValueError: If the task name does not match any known task.
        """
        normalized_name = task_name.strip().upper()
        for task in cls:
            if task.name == normalized_name:
                return task
        raise ValueError(f"Unknown speech task type: {task_name}")
