"""Tri-Modal Soft Voting Ensemble combining Baseline, WavLM, and Wav2Vec2."""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from src.application.pipelines.baseline_replication_pipeline import BaselineReplicationPipeline
from src.application.pipelines.self_supervised_evaluation_pipeline import SelfSupervisedEvaluationPipeline
from src.application.services.diagnostic_logger_service import DiagnosticLoggerService
from src.application.services.experiment_logger_service import ExperimentLoggerService
from src.application.services.metric_calculation_service import MetricCalculationService
from src.application.strategies.subject_split_validation_strategy import SubjectSplitValidationStrategy
from src.domain.entities.audio_sample import AudioSample
from src.domain.entities.evaluation_metrics import EvaluationMetrics
from src.domain.interfaces.validation_strategy import IValidationStrategy
from src.infrastructure.models.support_vector_classifier import SupportVectorClassifier


class TriModalEnsembleEvaluationPipeline:
    """Ensembles probabilistic predictions of Baseline, WavLM, and Wav2Vec2."""

    def __init__(
        self, weights: Tuple[float, float, float] = (0.50, 0.30, 0.20),
        baseline_checkpoint: Optional[str] = None, threshold: float = 0.50,
        cache_directory: Path = Path("./data/processed/segments"),
        validation_strategy: Optional[IValidationStrategy] = None,
        output_model_path: Optional[Path] = None,
    ) -> None:
        """Initialize tri-modal ensemble with weights and sub-pipelines."""
        self._weights = weights
        self._base_ckpt = Path(baseline_checkpoint) if baseline_checkpoint else None
        self._threshold = threshold
        self._cache_dir = Path(cache_directory)
        self._validator = validation_strategy or SubjectSplitValidationStrategy()
        self._output_model_path = output_model_path
        self._baseline_pipe = BaselineReplicationPipeline(self._cache_dir, self._validator)
        self._wavlm_pipe = SelfSupervisedEvaluationPipeline(
            "microsoft/wavlm-base-plus", 6, segmentation_cache_directory=self._cache_dir,
        )
        self._w2v2_pipe = SelfSupervisedEvaluationPipeline(
            "facebook/wav2vec2-base", 6, segmentation_cache_directory=self._cache_dir,
        )

    def run(self, raw_samples: List[AudioSample]) -> EvaluationMetrics:
        """Execute tri-modal soft voting ensemble across all 3 representation spaces."""
        w1, w2, w3 = self._weights
        ExperimentLoggerService.log_section(f"TRI-MODAL ENSEMBLE: Base({w1:.2f}) + WavLM({w2:.2f}) + W2V2({w3:.2f})")
        base_samples = self._baseline_pipe._segment_audio(raw_samples, use_segmentation=True)
        feat_base, labels, samples = self._baseline_pipe._extract_features(base_samples)
        feat_wavlm, _, _ = self._wavlm_pipe._extract_ssl_features(base_samples)
        feat_w2v2, _, _ = self._w2v2_pipe._extract_ssl_features(base_samples)

        train_idx, test_idx = self._get_split_indices(samples)
        p_base = self._get_base_probs(feat_base, labels, samples, train_idx, test_idx)
        p_wavlm = self._fit_ssl(feat_wavlm, labels, samples, train_idx, test_idx)
        p_w2v2 = self._fit_ssl(feat_w2v2, labels, samples, train_idx, test_idx)

        fused_probs = w1 * p_base + w2 * p_wavlm + w3 * p_w2v2
        test_labels = labels[test_idx]
        self._log_threshold_diagnostics(test_labels, fused_probs)

        fused_preds = (fused_probs >= self._threshold).astype(np.int32)
        DiagnosticLoggerService.log_confusion_matrix(test_labels, fused_preds)
        return MetricCalculationService.calculate(test_labels, fused_preds, fused_probs)

    def _get_split_indices(self, samples: List[AudioSample]):
        """Derive identical train/test indices using subject-level stratification."""
        subjects = list(dict.fromkeys([s.subject_id for s in samples]))
        labels = {s.subject_id: s.label for s in samples}
        train_s, test_s = train_test_split(subjects, test_size=0.30, random_state=42, stratify=[labels[s] for s in subjects])
        return ([i for i, s in enumerate(samples) if s.subject_id in train_s],
                [i for i, s in enumerate(samples) if s.subject_id in test_s])

    def _get_base_probs(self, feat, labels, samples, train_idx, test_idx):
        """Obtain baseline probabilities from pre-trained checkpoint or sweet-spot fit."""
        if self._base_ckpt and self._base_ckpt.exists():
            return SupportVectorClassifier.load(self._base_ckpt).predict_probability(feat[test_idx])
        grid = {"C": [1.0, 3.0, 5.0, 7.0, 10.0], "gamma": ["scale", "auto", 0.02, 0.04, 0.05, 0.06]}
        return self._fit_ssl(feat, labels, samples, train_idx, test_idx, grid)

    def _fit_ssl(self, feat, labels, samples, train_idx, test_idx, grid=None):
        """Fit a tuned SVM on training split and return test prediction probabilities."""
        groups = [samples[i].subject_id for i in train_idx]
        grid = grid or {"C": [0.5, 1.0, 2.0, 5.0], "gamma": ["scale", "auto", 0.0005, 0.001, 0.005]}
        clf = SupportVectorClassifier(enable_grid_search=True, parameter_grid=grid)
        clf.fit(feat[train_idx], labels[train_idx], subject_groups=groups)
        return clf.predict_probability(feat[test_idx])

    def _log_threshold_diagnostics(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Scan candidate decision thresholds and log sensitivity-specificity trade-offs."""
        print("\n  --- THRESHOLD SCANNING DIAGNOSTICS ---")
        for th in [0.48, 0.50, 0.52, 0.54, 0.56]:
            preds = (y_prob >= th).astype(np.int32)
            m = MetricCalculationService.calculate(y_true, preds, y_prob)
            marker = " <-- ACTIVE" if abs(th - self._threshold) < 0.001 else ""
            print(f"  Threshold {th:.2f} | Acc: {m.accuracy*100:.2f}% | Sens: {m.sensitivity*100:.2f}% | Spec: {m.specificity*100:.2f}% | F1: {m.f1_score*100:.2f}%{marker}")
