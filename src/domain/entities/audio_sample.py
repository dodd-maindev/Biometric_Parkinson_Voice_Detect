"""Entity representing an audio recording sample with metadata."""

from dataclasses import dataclass
from pathlib import Path
from src.domain.entities.speech_task_type import SpeechTaskType


@dataclass(frozen=True)
class AudioSample:
    """Immutable audio sample entity carrying subject and diagnosis metadata.

    Attributes:
        subject_id: Unique identifier for the individual speaker.
        file_path: Absolute or relative filesystem path to the audio file.
        is_parkinson: True if diagnosed with Parkinson's Disease (PD=1), False if Healthy Control (HC=0).
        task_type: The protocol used during recording (Read Text, Spontaneous Dialog).
        segment_index: Index of the segmented chunk if segmented, or 0 if raw.
    """

    subject_id: str
    file_path: Path
    is_parkinson: bool
    task_type: SpeechTaskType
    segment_index: int = 0

    @property
    def label(self) -> int:
        """Return binary integer label where 1 is Parkinson and 0 is Healthy Control."""
        return 1 if self.is_parkinson else 0
