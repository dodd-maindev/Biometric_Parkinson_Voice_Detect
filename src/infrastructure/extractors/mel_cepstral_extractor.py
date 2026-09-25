"""Extractor for Mel-Frequency Cepstral Coefficients and first-order deltas."""

from typing import List
import librosa
import numpy as np
from src.domain.entities.audio_sample import AudioSample
from src.domain.interfaces.feature_extractor import IFeatureExtractor


class MelCepstralExtractor(IFeatureExtractor):
    """Extracts 13 MFCCs and 13 first-order delta derivatives (26 dimensions)."""

    def __init__(self, number_of_coefficients: int = 13) -> None:
        """Initialize number of static cepstral coefficients."""
        self._number_of_coefficients = number_of_coefficients

    @property
    def feature_dimension(self) -> int:
        """Return 26 dimensions (13 static MFCCs + 13 delta coefficients)."""
        return self._number_of_coefficients * 2

    @property
    def feature_names(self) -> List[str]:
        """Return names for static and delta MFCCs."""
        static_names = [f"mfcc_{i+1}_mean" for i in range(self._number_of_coefficients)]
        delta_names = [f"mfcc_delta_{i+1}_mean" for i in range(self._number_of_coefficients)]
        return static_names + delta_names

    def extract(self, sample: AudioSample) -> np.ndarray:
        """Extract mean static MFCCs and mean delta coefficients from audio."""
        waveform, sample_rate = librosa.load(str(sample.file_path), sr=None, mono=True)
        if len(waveform) == 0:
            return np.zeros(self.feature_dimension, dtype=np.float32)

        mfccs = librosa.feature.mfcc(
            y=waveform,
            sr=sample_rate,
            n_mfcc=self._number_of_coefficients,
        )
        delta_mfccs = librosa.feature.delta(mfccs, order=1)

        static_means = np.mean(mfccs, axis=1)
        delta_means = np.mean(delta_mfccs, axis=1)

        combined_vector = np.concatenate([static_means, delta_means]).astype(np.float32)
        return np.nan_to_num(combined_vector, nan=0.0, posinf=0.0, neginf=0.0)
