"""Service extending experiment logger with feature and model diagnostics."""

from typing import List
import numpy as np
from src.application.services.experiment_logger_service import (
    ExperimentLoggerService,
)


class DiagnosticLoggerService:
    """Logs feature statistics, split details, and model diagnostics."""

    @staticmethod
    def log_feature_statistics(
        features: np.ndarray,
        feature_names: List[str],
    ) -> None:
        """Log per-feature min, max, mean, std, and NaN counts."""
        ExperimentLoggerService.log_section("FEATURE MATRIX DIAGNOSTICS")
        print(f"  Shape: {features.shape}  (samples x features)")
        nan_count = int(np.isnan(features).sum())
        inf_count = int(np.isinf(features).sum())
        print(f"  NaN values: {nan_count}  |  Inf values: {inf_count}")

        print(f"\n  {'Feature':<25} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
        print(f"  {'-'*65}")
        for i, name in enumerate(feature_names):
            column = features[:, i]
            print(
                f"  {name:<25} {column.min():>10.4f} {column.max():>10.4f} "
                f"{column.mean():>10.4f} {column.std():>10.4f}"
            )

    @staticmethod
    def log_train_test_split(
        train_subjects: List[str],
        test_subjects: List[str],
        train_labels: np.ndarray,
        test_labels: np.ndarray,
    ) -> None:
        """Log the composition of train and test sets."""
        ExperimentLoggerService.log_section("TRAIN / TEST SPLIT")
        train_pd = int(np.sum(train_labels == 1))
        train_hc = int(np.sum(train_labels == 0))
        test_pd = int(np.sum(test_labels == 1))
        test_hc = int(np.sum(test_labels == 0))

        print(f"  Train subjects ({len(train_subjects)}): {sorted(train_subjects)}")
        print(f"  Test  subjects ({len(test_subjects)}):  {sorted(test_subjects)}")
        print(f"  Train samples: {len(train_labels)} (PD={train_pd}, HC={train_hc})")
        print(f"  Test  samples: {len(test_labels)}  (PD={test_pd}, HC={test_hc})")

    @staticmethod
    def log_confusion_matrix(
        ground_truth: np.ndarray,
        predictions: np.ndarray,
    ) -> None:
        """Log the full confusion matrix and per-class precision/recall."""
        from sklearn.metrics import classification_report, confusion_matrix
        ExperimentLoggerService.log_section("CONFUSION MATRIX & CLASSIFICATION REPORT")
        cm = confusion_matrix(ground_truth, predictions, labels=[0, 1])
        print(f"               Predicted HC  Predicted PD")
        print(f"  Actual HC:       {cm[0,0]:>5d}         {cm[0,1]:>5d}")
        print(f"  Actual PD:       {cm[1,0]:>5d}         {cm[1,1]:>5d}")
        print()
        report = classification_report(
            ground_truth, predictions,
            target_names=["HC (Healthy)", "PD (Parkinson)"],
            zero_division=0.0,
        )
        print(report)

    @staticmethod
    def log_model_hyperparameters(params: dict) -> None:
        """Log selected model hyperparameters."""
        ExperimentLoggerService.log_subsection("Model Hyperparameters")
        for key, value in params.items():
            print(f"  {key}: {value}")
