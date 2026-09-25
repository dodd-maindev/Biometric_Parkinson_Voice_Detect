"""Interface for audio segmentation strategies."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from src.domain.entities.audio_sample import AudioSample


class IAudioSegmenter(ABC):
    """Abstract interface defining the audio segmentation contract."""

    @abstractmethod
    def segment(
        self,
        sample: AudioSample,
        output_directory: Path,
    ) -> List[AudioSample]:
        """Segment a raw audio sample into smaller chunks.

        Args:
            sample: The original parent AudioSample entity.
            output_directory: Directory where generated audio chunks will be saved.

        Returns:
            A list of new AudioSample entities corresponding to the chunks.
        """
        pass
