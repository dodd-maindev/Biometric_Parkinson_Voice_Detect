"""Leave-One-Subject-Out Cross-Validation strategy implementation."""

from typing import List
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from src.application.services.metric_calculation_service import (
    MetricCalculationService,
)
from src.domain.entities.audio_sample import AudioSample
from src.domain.entities.evaluation_metrics import EvaluationMetrics
from src.domain.interfaces.classifier_model import IClassifierModel
from src.domain.interfaces.validation_strategy import IValidationStrategy


class LeaveOneSubjectOutStrategy(IValidationStrategy):
    """Executes Leave-One-Subject-Out Cross-Validation (LOSOCV) to prevent leakage."""

    def evaluate(
        self,
        classifier: IClassifierModel,
        features: np.ndarray,
        labels: np.ndarray,
        samples: List[AudioSample],
    ) -> EvaluationMetrics:
        """Run LOSOCV across all distinct subject IDs and aggregate out-of-fold predictions."""
        subject_groups = [sample.subject_id for sample in samples]
        splitter = LeaveOneGroupOut()

        out_of_fold_predictions = np.zeros(len(labels), dtype=np.int32)
        out_of_fold_probabilities = np.zeros(len(labels), dtype=np.float32)

        for train_indices, test_indices in splitter.split(features, labels, groups=subject_groups):
            train_features = features[train_indices]
            train_labels = labels[train_indices]
            test_features = features[test_indices]

            # Fit classifier purely on training subjects
            classifier.fit(train_features, train_labels)

            fold_predictions = classifier.predict(test_features)
            fold_probabilities = classifier.predict_probability(test_features)

            out_of_fold_predictions[test_indices] = fold_predictions
            out_of_fold_probabilities[test_indices] = fold_probabilities

        return MetricCalculationService.calculate(
            ground_truth=labels,
            predictions=out_of_fold_predictions,
            probabilities=out_of_fold_probabilities,
        )
