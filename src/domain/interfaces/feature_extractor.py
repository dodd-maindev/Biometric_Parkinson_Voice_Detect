"""Interface for feature extraction algorithms."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
from src.domain.entities.audio_sample import AudioSample


class IFeatureExtractor(ABC):
    """Abstract contract for extracting numerical feature vectors from audio."""

    @property
    @abstractmethod
    def feature_dimension(self) -> int:
        """Return the dimensionality of the generated feature vector."""
        pass

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """Return human-readable names for each feature column."""
        pass

    @abstractmethod
    def extract(self, sample: AudioSample) -> np.ndarray:
        """Extract a 1D numerical feature vector from a given audio sample.

        Args:
            sample: The target AudioSample to process.

        Returns:
            A 1D numpy array representing the extracted feature vector.
        """
        pass
