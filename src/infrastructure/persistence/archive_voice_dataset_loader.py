"""Persistence loader for scanning and indexing Voice_Dataset archive files."""

from pathlib import Path
from typing import List, Optional
from src.domain.entities.audio_sample import AudioSample
from src.domain.entities.speech_task_type import SpeechTaskType


class ArchiveVoiceDatasetLoader:
    """Scans and parses Voice_Dataset directory structure into AudioSample entities."""

    def __init__(self, dataset_root_directory: Path) -> None:
        """Initialize with dataset root directory containing Healthy and Parkinsons."""
        self._root_dir = Path(dataset_root_directory)

    def load_samples(
        self,
        target_task_type: Optional[SpeechTaskType] = None,
    ) -> List[AudioSample]:
        """Scan directory and parse audio metadata into AudioSample entities."""
        if not self._root_dir.exists():
            return []

        if target_task_type and target_task_type != SpeechTaskType.SUSTAINED_VOWEL:
            return []

        audio_samples: List[AudioSample] = []
        audio_samples.extend(self._load_category("Healthy", is_parkinson=False))
        audio_samples.extend(self._load_category("Parkinsons", is_parkinson=True))
        return audio_samples

    def _load_category(self, folder_name: str, is_parkinson: bool) -> List[AudioSample]:
        """Load audio samples from a specific diagnosis directory."""
        category_dir = self._root_dir / folder_name
        if not category_dir.exists():
            return []

        samples: List[AudioSample] = []
        for file_path in sorted(category_dir.glob("*.wav")):
            subject_id = f"{folder_name.upper()}_{file_path.stem}"
            samples.append(
                AudioSample(
                    subject_id=subject_id,
                    file_path=file_path,
                    is_parkinson=is_parkinson,
                    task_type=SpeechTaskType.SUSTAINED_VOWEL,
                    segment_index=0,
                )
            )
        return samples
