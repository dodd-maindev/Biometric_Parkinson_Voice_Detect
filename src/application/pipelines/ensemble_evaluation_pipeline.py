"""Pipeline for evaluating Multi-Model Soft Voting Ensemble (Baseline + SSL)."""

from pathlib import Path
from typing import List, Optional
import numpy as np
from sklearn.model_selection import train_test_split
from src.application.pipelines.baseline_replication_pipeline import (
    BaselineReplicationPipeline,
)
from src.application.pipelines.self_supervised_evaluation_pipeline import (
    SelfSupervisedEvaluationPipeline,
)
from src.application.services.diagnostic_logger_service import (
    DiagnosticLoggerService,
)
from src.application.services.experiment_logger_service import (
    ExperimentLoggerService,
)
from src.application.services.metric_calculation_service import (
    MetricCalculationService,
)
from src.application.strategies.subject_split_validation_strategy import (
    SubjectSplitValidationStrategy,
)
from src.domain.entities.audio_sample import AudioSample
from src.domain.entities.evaluation_metrics import EvaluationMetrics
from src.domain.interfaces.validation_strategy import IValidationStrategy
from src.infrastructure.models.support_vector_classifier import (
    SupportVectorClassifier,
)


class EnsembleEvaluationPipeline:
    """Ensembles probabilistic predictions of Baseline and Foundation SSL models."""

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        layer_index: Optional[int] = 6,
        baseline_weight: float = 0.60,
        segmentation_cache_directory: Path = Path("./data/processed/segments"),
        validation_strategy: Optional[IValidationStrategy] = None,
        output_model_path: Optional[Path] = None,
    ) -> None:
        """Initialize ensemble pipeline with models, weights, and cache path."""
        self._model_name = model_name
        self._layer_index = layer_index
        self._weight = baseline_weight
        self._cache_dir = Path(segmentation_cache_directory)
        self._validator = validation_strategy or SubjectSplitValidationStrategy()
        self._output_model_path = output_model_path
        self._baseline_pipe = BaselineReplicationPipeline(self._cache_dir, self._validator)
        self._ssl_pipe = SelfSupervisedEvaluationPipeline(
            model_name=model_name, layer_index=layer_index,
            segmentation_cache_directory=self._cache_dir, validation_strategy=self._validator,
        )

    def run(self, raw_samples: List[AudioSample]) -> EvaluationMetrics:
        """Execute soft voting ensemble between Baseline and SSL representations."""
        title = f"SOFT VOTING: Baseline ({self._weight:.2f}) + {self._model_name} ({1-self._weight:.2f})"
        ExperimentLoggerService.log_section(title)
        base_samples = self._baseline_pipe._segment_audio(raw_samples, use_segmentation=True)
        base_feat, base_labels, samples = self._baseline_pipe._extract_features(base_samples)
        ssl_feat, _, _ = self._ssl_pipe._extract_ssl_features(base_samples)

        train_idx, test_idx = self._get_split_indices(samples)
        base_probs = self._fit_and_predict_proba(base_feat, base_labels, samples, train_idx, test_idx)
        ssl_probs = self._fit_and_predict_proba(ssl_feat, base_labels, samples, train_idx, test_idx)

        fused_probs = self._weight * base_probs + (1.0 - self._weight) * ssl_probs
        fused_preds = (fused_probs >= 0.5).astype(np.int32)
        test_labels = base_labels[test_idx]

        DiagnosticLoggerService.log_confusion_matrix(test_labels, fused_preds)
        return MetricCalculationService.calculate(test_labels, fused_preds, fused_probs)

    def _get_split_indices(self, samples: List[AudioSample]):
        """Derive identical train/test indices using subject-level stratification."""
        unique_subjects = list(dict.fromkeys([s.subject_id for s in samples]))
        subject_labels = {s.subject_id: s.label for s in samples}
        train_subjs, test_subjs = train_test_split(
            unique_subjects, test_size=0.30, random_state=42,
            stratify=[subject_labels[s] for s in unique_subjects],
        )
        train_idx = [i for i, s in enumerate(samples) if s.subject_id in train_subjs]
        test_idx = [i for i, s in enumerate(samples) if s.subject_id in test_subjs]
        return train_idx, test_idx

    def _fit_and_predict_proba(self, feat, labels, samples, train_idx, test_idx):
        """Fit a tuned SVM classifier on training split and return test probabilities."""
        train_x, train_y = feat[train_idx], labels[train_idx]
        test_x = feat[test_idx]
        train_groups = [samples[i].subject_id for i in train_idx]
        clf = SupportVectorClassifier(enable_grid_search=True)
        clf.fit(train_x, train_y, subject_groups=train_groups)
        return clf.predict_probability(test_x)
