"""Evaluation metrics entity storing standardized diagnostic performance results."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class EvaluationMetrics:
    """Holds all clinical and statistical classification performance metrics.

    Attributes:
        accuracy: Percentage of correct classifications.
        sensitivity: Recall of the Parkinson positive class (True Positive Rate).
        specificity: True Negative Rate (Correct identification of Healthy Controls).
        f1_score: Harmonic mean of precision and recall for the positive class.
        roc_auc: Area Under the Receiver Operating Characteristic Curve.
        matthews_correlation_coefficient: Balanced correlation metric for binary classifications.
    """

    accuracy: float
    sensitivity: float
    specificity: float
    f1_score: float
    roc_auc: float
    matthews_correlation_coefficient: float

    def to_dictionary(self) -> Dict[str, float]:
        """Convert metrics to a readable dictionary format."""
        return {
            "Accuracy": round(self.accuracy, 4),
            "Sensitivity": round(self.sensitivity, 4),
            "Specificity": round(self.specificity, 4),
            "F1-Score": round(self.f1_score, 4),
            "ROC-AUC": round(self.roc_auc, 4),
            "MCC": round(self.matthews_correlation_coefficient, 4),
        }

    def format_summary(self) -> str:
        """Return formatted string summary of all metrics."""
        return (
            f"Accuracy: {self.accuracy * 100:.2f}% | "
            f"Sensitivity: {self.sensitivity * 100:.2f}% | "
            f"Specificity: {self.specificity * 100:.2f}% | "
            f"F1-Score: {self.f1_score * 100:.2f}% | "
            f"ROC-AUC: {self.roc_auc:.4f} | "
            f"MCC: {self.matthews_correlation_coefficient:.4f}"
        )
