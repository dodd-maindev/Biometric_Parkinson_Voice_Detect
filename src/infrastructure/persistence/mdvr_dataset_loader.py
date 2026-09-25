"""Persistence loader for scanning and indexing MDVR-KCL audio files."""

import re
from pathlib import Path
from typing import List, Optional
from src.domain.entities.audio_sample import AudioSample
from src.domain.entities.speech_task_type import SpeechTaskType


class MdvrDatasetLoader:
    """Scans and parses MDVR-KCL directory structure into AudioSample entities."""

    _SUBJECT_ID_PATTERN = re.compile(r"(ID\d+)", re.IGNORECASE)

    def __init__(self, dataset_root_directory: Path) -> None:
        """Initialize with dataset root directory."""
        self._root_dir = Path(dataset_root_directory)

    def load_samples(
        self,
        target_task_type: Optional[SpeechTaskType] = None,
    ) -> List[AudioSample]:
        """Scan directory and parse audio metadata into AudioSample entities."""
        if not self._root_dir.exists():
            return []

        audio_samples: List[AudioSample] = []
        for file_path in sorted(self._root_dir.rglob("*.wav")):
            file_name = file_path.stem.lower()
            subject_id = self._extract_subject_id(file_name)
            is_parkinson = self._detect_parkinson_label(file_name, file_path)
            task = self._detect_task_type(file_name)

            if target_task_type is not None and task != target_task_type:
                continue

            audio_samples.append(AudioSample(
                subject_id=subject_id, file_path=file_path,
                is_parkinson=is_parkinson, task_type=task, segment_index=0,
            ))

        return audio_samples

    def _extract_subject_id(self, file_name: str) -> str:
        """Extract numeric subject ID (e.g. 'ID22') using regex, not underscore split."""
        match = self._SUBJECT_ID_PATTERN.search(file_name)
        return match.group(1).upper() if match else "UNKNOWN"

    @staticmethod
    def _detect_parkinson_label(file_name: str, file_path: Path) -> bool:
        """Determine PD diagnosis from filename convention (ID02_pd_...)."""
        if "_pd_" in file_name or "_pd" in file_name:
            return True
        path_parts_lower = [p.lower() for p in file_path.parts]
        return "pd" in path_parts_lower

    @staticmethod
    def _detect_task_type(file_name: str) -> SpeechTaskType:
        """Classify speech task type from filename keywords."""
        dialog_keywords = ("dialog", "conversation", "spontaneous")
        if any(keyword in file_name for keyword in dialog_keywords):
            return SpeechTaskType.SPONTANEOUS_DIALOG
        if "vowel" in file_name:
            return SpeechTaskType.SUSTAINED_VOWEL
        return SpeechTaskType.READ_TEXT
