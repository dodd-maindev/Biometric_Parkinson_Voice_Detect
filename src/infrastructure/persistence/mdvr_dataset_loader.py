"""Persistence loader for scanning and indexing MDVR-KCL audio files."""

from pathlib import Path
from typing import List, Optional
from src.domain.entities.audio_sample import AudioSample
from src.domain.entities.speech_task_type import SpeechTaskType


class MdvrDatasetLoader:
    """Scans and parses MDVR-KCL directory structure into AudioSample entities."""

    def __init__(self, dataset_root_directory: Path) -> None:
        """Initialize with dataset root directory."""
        self._root_dir = Path(dataset_root_directory)

    def load_samples(
        self,
        target_task_type: Optional[SpeechTaskType] = None,
    ) -> List[AudioSample]:
        """Scan directory and parse audio metadata into a list of AudioSample entities.

        Args:
            target_task_type: Optional filter for a specific speech task.

        Returns:
            A list of AudioSample entities parsed from discovered audio files.
        """
        if not self._root_dir.exists():
            return []

        audio_samples: List[AudioSample] = []
        for file_path in sorted(self._root_dir.rglob("*.wav")):
            file_name = file_path.stem.lower()

            # Parse subject id from convention like ID02_pd_...
            parts = file_name.split("_")
            subject_id = parts[0].upper() if len(parts) > 0 else "UNKNOWN"

            # Parse diagnosis (PD vs HC)
            is_parkinson = "_pd_" in file_name or "_pd" in file_name or "pd" in [p.lower() for p in file_path.parts]

            # Detect speech task type from path or filename
            if "dialog" in file_name or "conversation" in file_name or "spontaneous" in file_name:
                task = SpeechTaskType.SPONTANEOUS_DIALOG
            elif "vowel" in file_name:
                task = SpeechTaskType.SUSTAINED_VOWEL
            else:
                task = SpeechTaskType.READ_TEXT

            if target_task_type is not None and task != target_task_type:
                continue

            audio_samples.append(
                AudioSample(
                    subject_id=subject_id,
                    file_path=file_path,
                    is_parkinson=is_parkinson,
                    task_type=task,
                    segment_index=0,
                )
            )

        return audio_samples
