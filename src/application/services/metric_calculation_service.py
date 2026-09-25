"""Service for computing standardized diagnostic classification metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from src.domain.entities.evaluation_metrics import EvaluationMetrics


class MetricCalculationService:
    """Calculates all diagnostic metrics reported in Hossain et al. (2026)."""

    @staticmethod
    def calculate(
        ground_truth: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> EvaluationMetrics:
        """Compute Accuracy, Sensitivity, Specificity, F1, ROC-AUC, and MCC.

        Args:
            ground_truth: 1D array of true binary integer labels (0: HC, 1: PD).
            predictions: 1D array of predicted binary integer labels.
            probabilities: 1D array of predicted probability scores for PD.

        Returns:
            An EvaluationMetrics entity.
        """
        accuracy = float(accuracy_score(ground_truth, predictions))

        conf_matrix = confusion_matrix(ground_truth, predictions, labels=[0, 1])
        true_negative = float(conf_matrix[0, 0])
        false_positive = float(conf_matrix[0, 1])
        false_negative = float(conf_matrix[1, 0])
        true_positive = float(conf_matrix[1, 1])

        actual_positives = true_positive + false_negative
        sensitivity = (true_positive / actual_positives) if actual_positives > 0 else 0.0

        actual_negatives = true_negative + false_positive
        specificity = (true_negative / actual_negatives) if actual_negatives > 0 else 0.0

        f1 = float(f1_score(ground_truth, predictions, zero_division=0.0))

        # Safeguard ROC-AUC if only one class exists in ground truth
        if len(np.unique(ground_truth)) > 1:
            auc = float(roc_auc_score(ground_truth, probabilities))
        else:
            auc = 0.5

        mcc = float(matthews_corrcoef(ground_truth, predictions))

        return EvaluationMetrics(
            accuracy=accuracy,
            sensitivity=sensitivity,
            specificity=specificity,
            f1_score=f1,
            roc_auc=auc,
            matthews_correlation_coefficient=mcc,
        )
