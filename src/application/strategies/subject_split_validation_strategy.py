"""Subject-wise 70/30 Train/Test validation strategy matching paper Section 2.4.1."""

from typing import List
import numpy as np
from sklearn.model_selection import train_test_split
from src.application.services.metric_calculation_service import (
    MetricCalculationService,
)
from src.domain.entities.audio_sample import AudioSample
from src.domain.entities.evaluation_metrics import EvaluationMetrics
from src.domain.interfaces.classifier_model import IClassifierModel
from src.domain.interfaces.validation_strategy import IValidationStrategy


class SubjectSplitValidationStrategy(IValidationStrategy):
    """Evaluates performance using a subject-wise 70/30 train/test split."""

    def __init__(self, test_ratio: float = 0.30, random_seed: int = 42) -> None:
        """Initialize split ratio and random state."""
        self._test_ratio = test_ratio
        self._seed = random_seed

    def evaluate(
        self,
        classifier: IClassifierModel,
        features: np.ndarray,
        labels: np.ndarray,
        samples: List[AudioSample],
    ) -> EvaluationMetrics:
        """Partition by subject ID into 70% train and 30% test sets, then evaluate."""
        unique_subjects = list(dict.fromkeys([sample.subject_id for sample in samples]))
        subject_labels = {
            s.subject_id: s.label for s in samples
        }
        stratify_labels = [subject_labels[subj] for subj in unique_subjects]

        train_subjects, test_subjects = train_test_split(
            unique_subjects,
            test_size=self._test_ratio,
            random_state=self._seed,
            stratify=stratify_labels,
        )

        train_indices = [i for i, s in enumerate(samples) if s.subject_id in train_subjects]
        test_indices = [i for i, s in enumerate(samples) if s.subject_id in test_subjects]

        train_x, train_y = features[train_indices], labels[train_indices]
        test_x, test_y = features[test_indices], labels[test_indices]

        classifier.fit(train_x, train_y)
        predictions = classifier.predict(test_x)
        probabilities = classifier.predict_probability(test_x)

        return MetricCalculationService.calculate(
            ground_truth=test_y,
            predictions=predictions,
            probabilities=probabilities,
        )
