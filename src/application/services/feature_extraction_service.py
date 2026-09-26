from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from tqdm import tqdm
from src.application.services.diagnostic_logger_service import (
    DiagnosticLoggerService,
)
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
        return sum(e.feature_dimension for e in self._extractors)

    @property
    def all_feature_names(self) -> List[str]:
        """Return combined feature names from all registered extractors."""
        names: List[str] = []
        for extractor in self._extractors:
            names.extend(extractor.feature_names)
        return names

    def extract_dataset(
        self,
        samples: List[AudioSample],
        show_progress: bool = True,
        cache_file: Optional[Path] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extract or load cached combined features, labels, and subject IDs."""
        if cache_file is not None and cache_file.exists():
            data = np.load(cache_file)
            print(f"  Loaded {len(data['labels'])} cached features from {cache_file.name}")
            return data["features"], data["labels"], list(data["subjects"])

        feature_rows: List[np.ndarray] = []
        labels: List[int] = []
        subject_ids: List[str] = []
        iterator = tqdm(samples, desc="Extracting features") if show_progress else samples

        for sample in iterator:
            sample_vectors = [e.extract(sample) for e in self._extractors]
            feature_rows.append(np.concatenate(sample_vectors))
            labels.append(sample.label)
            subject_ids.append(sample.subject_id)

        features_matrix = np.array(feature_rows, dtype=np.float32)
        labels_vector = np.array(labels, dtype=np.int32)

        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_file, features=features_matrix,
                labels=labels_vector, subjects=np.array(subject_ids),
            )
            print(f"  Cached features to {cache_file.name}")

        DiagnosticLoggerService.log_feature_statistics(
            features_matrix, self.all_feature_names,
        )
        return features_matrix, labels_vector, subject_ids
