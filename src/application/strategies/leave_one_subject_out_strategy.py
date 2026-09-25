"""Leave-One-Subject-Out Cross-Validation strategy implementation."""

from typing import List
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from src.application.services.diagnostic_logger_service import (
    DiagnosticLoggerService,
)
from src.application.services.experiment_logger_service import (
    ExperimentLoggerService,
)
from src.application.services.metric_calculation_service import (
    MetricCalculationService,
)
from src.domain.entities.audio_sample import AudioSample
from src.domain.entities.evaluation_metrics import EvaluationMetrics
from src.domain.interfaces.classifier_model import IClassifierModel
from src.domain.interfaces.validation_strategy import IValidationStrategy


class LeaveOneSubjectOutStrategy(IValidationStrategy):
    """Executes LOSOCV to prevent data leakage between subjects."""

    def evaluate(
        self,
        classifier: IClassifierModel,
        features: np.ndarray,
        labels: np.ndarray,
        samples: List[AudioSample],
    ) -> EvaluationMetrics:
        """Run LOSOCV across all subjects and aggregate out-of-fold predictions."""
        subject_groups = [sample.subject_id for sample in samples]
        splitter = LeaveOneGroupOut()
        unique_subjects = sorted(set(subject_groups))

        ExperimentLoggerService.log_section(
            f"LOSOCV ({len(unique_subjects)} folds)",
        )

        out_of_fold_preds = np.zeros(len(labels), dtype=np.int32)
        out_of_fold_probs = np.zeros(len(labels), dtype=np.float32)

        for fold_idx, (train_idx, test_idx) in enumerate(
            splitter.split(features, labels, groups=subject_groups),
        ):
            held_out_subject = subject_groups[test_idx[0]]
            held_out_label = "PD" if labels[test_idx[0]] == 1 else "HC"

            classifier.fit(features[train_idx], labels[train_idx])
            fold_preds = classifier.predict(features[test_idx])
            fold_probs = classifier.predict_probability(features[test_idx])

            correct = int(np.sum(fold_preds == labels[test_idx]))
            total = len(test_idx)
            fold_acc = correct / total * 100

            print(
                f"  Fold {fold_idx+1:2d}: Subject={held_out_subject} [{held_out_label}] "
                f"| {total} samples | {correct}/{total} correct ({fold_acc:.1f}%)"
            )

            out_of_fold_preds[test_idx] = fold_preds
            out_of_fold_probs[test_idx] = fold_probs

        DiagnosticLoggerService.log_confusion_matrix(labels, out_of_fold_preds)

        return MetricCalculationService.calculate(
            ground_truth=labels,
            predictions=out_of_fold_preds,
            probabilities=out_of_fold_probs,
        )
