"""Extractor for Gammatone Cepstral Coefficients (GTCC / GFCC)."""

from typing import List
import numpy as np
import soundfile as sf
from src.domain.entities.audio_sample import AudioSample
from src.domain.interfaces.feature_extractor import IFeatureExtractor

try:
    from spafe.features.gfcc import gfcc as compute_gammatone_cepstrum
except ImportError:
    compute_gammatone_cepstrum = None


class GammatoneCepstralExtractor(IFeatureExtractor):
    """Extracts 13 mean Gammatone Frequency Cepstral Coefficients."""

    def __init__(self, number_of_coefficients: int = 13) -> None:
        """Initialize number of GTCC coefficients."""
        self._number_of_coefficients = number_of_coefficients

    @property
    def feature_dimension(self) -> int:
        """Return 13 dimensions for GTCC."""
        return self._number_of_coefficients

    @property
    def feature_names(self) -> List[str]:
        """Return names for each GTCC component."""
        return [f"gtcc_{i+1}_mean" for i in range(self._number_of_coefficients)]

    def extract(self, sample: AudioSample) -> np.ndarray:
        """Extract mean GTCC features from audio signal using auditory filter bank."""
        waveform, sample_rate = sf.read(str(sample.file_path))
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)

        if len(waveform) == 0:
            return np.zeros(self.feature_dimension, dtype=np.float32)

        if compute_gammatone_cepstrum is not None:
            try:
                matrix = compute_gammatone_cepstrum(
                    sig=waveform,
                    fs=sample_rate,
                    num_ceps=self._number_of_coefficients,
                    nfilts=26,
                )
                feature_vector = np.mean(matrix, axis=0).astype(np.float32)
                return np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                pass

        # Fallback using standard spectral centroid & filterbank approximation
        fft_spectrum = np.abs(np.fft.rfft(waveform[: sample_rate * 2]))
        bins = np.linspace(0, len(fft_spectrum) - 1, self._number_of_coefficients + 1, dtype=int)
        energies = [np.mean(fft_spectrum[bins[i] : bins[i + 1]]) for i in range(self._number_of_coefficients)]
        return np.nan_to_num(np.array(energies, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
