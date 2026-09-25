"""Extractor for 11 clinical acoustic features via Praat Parselmouth."""

from typing import List
import numpy as np
import parselmouth
from parselmouth.praat import call
from src.domain.entities.audio_sample import AudioSample
from src.domain.interfaces.feature_extractor import IFeatureExtractor


class PraatAcousticExtractor(IFeatureExtractor):
    """Extracts 11 phonatory and prosodic features used in Hossain et al. (2026)."""

    FEATURE_NAMES: List[str] = [
        "pitch_mean",
        "pitch_std",
        "pitch_min",
        "pitch_max",
        "jitter_local",
        "jitter_rap",
        "jitter_ppq5",
        "shimmer_local",
        "shimmer_apq3",
        "shimmer_apq5",
        "hnr_mean",
    ]

    @property
    def feature_dimension(self) -> int:
        """Return 11 as total number of acoustic features."""
        return len(self.FEATURE_NAMES)

    @property
    def feature_names(self) -> List[str]:
        """Return the list of 11 acoustic feature identifiers."""
        return self.FEATURE_NAMES.copy()

    def extract(self, sample: AudioSample) -> np.ndarray:
        """Extract pitch, jitter, shimmer, and HNR features from audio."""
        sound = parselmouth.Sound(str(sample.file_path))
        pitch = call(sound, "To Pitch", 0.0, 75.0, 600.0)

        pitch_mean = call(pitch, "Get mean", 0.0, 0.0, "Hertz") or 0.0
        pitch_std = call(pitch, "Get standard deviation", 0.0, 0.0, "Hertz") or 0.0
        pitch_min = call(pitch, "Get minimum", 0.0, 0.0, "Hertz", "Parabolic") or 0.0
        pitch_max = call(pitch, "Get maximum", 0.0, 0.0, "Hertz", "Parabolic") or 0.0

        point_process = call(sound, "To PointProcess (periodic, cc)", 75.0, 600.0)

        jitter_local = call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3) or 0.0
        jitter_rap = call(point_process, "Get jitter (rap)", 0.0, 0.0, 0.0001, 0.02, 1.3) or 0.0
        jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0.0, 0.0, 0.0001, 0.02, 1.3) or 0.0

        shimmer_local = call([sound, point_process], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6) or 0.0
        shimmer_apq3 = call([sound, point_process], "Get shimmer (apq3)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6) or 0.0
        shimmer_apq5 = call([sound, point_process], "Get shimmer (apq5)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6) or 0.0

        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 4.5)
        hnr_mean = call(harmonicity, "Get mean", 0.0, 0.0) or 0.0

        raw_vector = np.array([
            pitch_mean, pitch_std, pitch_min, pitch_max,
            jitter_local, jitter_rap, jitter_ppq5,
            shimmer_local, shimmer_apq3, shimmer_apq5,
            hnr_mean,
        ], dtype=np.float32)

        return np.nan_to_num(raw_vector, nan=0.0, posinf=0.0, neginf=0.0)
