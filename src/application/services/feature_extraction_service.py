"""Service orchestrating multiple feature extractors across audio samples."""

from typing import List, Tuple
import numpy as np
from tqdm import tqdm
from src.domain.entities.audio_sample import AudioSample
from src.domain.interfaces.feature_extractor import IFeatureExtractor


class FeatureExtractionService:
    """Extracts and concatenates feature representations from registered extractors."""

    def __init__(self, extractors: List[IFeatureExtractor]) -> None:
        """Initialize with one or more feature extractor implementations."""
        self._extractors = extractors

    @property
    def total_feature_dimension(self) -> int:
        """Return combined dimensionality of all registered extractors."""
        return sum(extractor.feature_dimension for extractor in self._extractors)

    def extract_dataset(
        self,
        samples: List[AudioSample],
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extract combined feature vectors, labels, and subject IDs for all samples.

        Args:
            samples: List of AudioSample entities to process.
            show_progress: Display tqdm progress bar if True.

        Returns:
            Tuple of (features_matrix [N, D], labels_vector [N], subject_ids [N]).
        """
        feature_rows: List[np.ndarray] = []
        labels: List[int] = []
        subject_ids: List[str] = []

        iterator = tqdm(samples, desc="Extracting features") if show_progress else samples

        for sample in iterator:
            sample_vectors: List[np.ndarray] = []
            for extractor in self._extractors:
                vector = extractor.extract(sample)
                sample_vectors.append(vector)

            concatenated_vector = np.concatenate(sample_vectors)
            feature_rows.append(concatenated_vector)
            labels.append(sample.label)
            subject_ids.append(sample.subject_id)

        features_matrix = np.array(feature_rows, dtype=np.float32)
        labels_vector = np.array(labels, dtype=np.int32)

        return features_matrix, labels_vector, subject_ids
