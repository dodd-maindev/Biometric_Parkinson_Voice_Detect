"""Interface for validation strategies ensuring unbiased model evaluation."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
from src.domain.entities.audio_sample import AudioSample
from src.domain.entities.evaluation_metrics import EvaluationMetrics
from src.domain.interfaces.classifier_model import IClassifierModel


class IValidationStrategy(ABC):
    """Abstract interface defining cross-validation execution protocol."""

    @abstractmethod
    def evaluate(
        self,
        classifier: IClassifierModel,
        features: np.ndarray,
        labels: np.ndarray,
        samples: List[AudioSample],
    ) -> EvaluationMetrics:
        """Execute validation across subject-partitioned folds.

        Args:
            classifier: An instance implementing IClassifierModel.
            features: 2D array of extracted feature vectors.
            labels: 1D array of ground truth labels.
            samples: List of AudioSample entities matching row order of features.

        Returns:
            An EvaluationMetrics entity with aggregated performance results.
        """
        pass
