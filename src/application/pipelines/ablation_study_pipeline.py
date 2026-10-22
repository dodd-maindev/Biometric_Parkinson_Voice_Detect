"""Pipeline for running ablation studies across tri-modal constituent models."""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from src.application.pipelines.tri_modal_ensemble_evaluation_pipeline import (
    TriModalEnsembleEvaluationPipeline,
)
from src.application.services.experiment_logger_service import ExperimentLoggerService
from src.application.services.metric_calculation_service import MetricCalculationService
from src.domain.entities.audio_sample import AudioSample
from src.domain.entities.evaluation_metrics import EvaluationMetrics
from src.domain.interfaces.validation_strategy import IValidationStrategy


class AblationStudyPipeline:
    """Executes systematic ablation study on Baseline, WavLM, and Wav2Vec2."""

    ABLATION_CONFIGURATIONS = [
        ("Baseline (Acoustic + GTCC)", (1.0, 0.0, 0.0), 0.50),
        ("WavLM Alone (Layer 6)", (0.0, 1.0, 0.0), 0.50),
        ("Wav2Vec2 Alone (Layer 6)", (0.0, 0.0, 1.0), 0.50),
        ("Dual: Baseline + WavLM", (0.60, 0.40, 0.0), 0.50),
        ("Dual: Baseline + Wav2Vec2", (0.70, 0.0, 0.30), 0.50),
        ("Dual: WavLM + Wav2Vec2", (0.0, 0.60, 0.40), 0.50),
        ("Tri-Modal (Default Threshold)", (0.50, 0.30, 0.20), 0.50),
        ("Tri-Modal (Calibrated Threshold)", (0.50, 0.30, 0.20), 0.56),
    ]

    def __init__(
        self,
        cache_directory: Path = Path("./data/processed/mdvr/segments"),
        baseline_checkpoint: Optional[str] = None,
        validation_strategy: Optional[IValidationStrategy] = None,
    ) -> None:
        """Initialize ablation study with cache directory and validator."""
        self._cache_dir = Path(cache_directory)
        self._base_ckpt = baseline_checkpoint
        self._validator = validation_strategy

    def run(self, raw_samples: List[AudioSample]) -> Dict[str, EvaluationMetrics]:
        """Execute all ablation configurations and return metrics mapping."""
        ExperimentLoggerService.log_section("ABLATION STUDY: CONSTITUENT MODEL CONTRIBUTIONS")
        results: Dict[str, EvaluationMetrics] = {}

        # Re-use tri-modal pipeline to extract features and obtain test probabilities
        pipe = TriModalEnsembleEvaluationPipeline(
            weights=(0.50, 0.30, 0.20), baseline_checkpoint=self._base_ckpt,
            threshold=0.56, cache_directory=self._cache_dir,
            validation_strategy=self._validator,
        )
        base_samples = pipe._baseline_pipe._segment_audio(raw_samples, use_segmentation=True)
        feat_base, labels, samples = pipe._baseline_pipe._extract_features(base_samples)
        feat_wavlm, _, _ = pipe._wavlm_pipe._extract_ssl_features(base_samples)
        feat_w2v2, _, _ = pipe._w2v2_pipe._extract_ssl_features(base_samples)

        train_idx, test_idx = pipe._get_split_indices(samples)
        _, p_base = pipe._get_base_probs(feat_base, labels, samples, train_idx, test_idx)
        _, p_wavlm = pipe._fit_ssl(feat_wavlm, labels, samples, train_idx, test_idx)
        _, p_w2v2 = pipe._fit_ssl(feat_w2v2, labels, samples, train_idx, test_idx)
        test_labels = labels[test_idx]

        for name, (w1, w2, w3), th in self.ABLATION_CONFIGURATIONS:
            p_fused = w1 * p_base + w2 * p_wavlm + w3 * p_w2v2
            preds = (p_fused >= th).astype(np.int32)
            results[name] = MetricCalculationService.calculate(test_labels, preds, p_fused)

        self._log_summary_table(results)
        return results

    def _log_summary_table(self, results: Dict[str, EvaluationMetrics]) -> None:
        """Display formatted markdown table comparing ablation metrics."""
        print("\n" + "=" * 80)
        print(f"{'Configuration':<35} | {'Acc (%)':<8} | {'Sens (%)':<9} | {'Spec (%)':<9} | {'F1 (%)':<8} | {'AUC':<6}")
        print("-" * 80)
        for name, m in results.items():
            print(f"{name:<35} | {m.accuracy*100:>7.2f}% | {m.sensitivity*100:>8.2f}% | {m.specificity*100:>8.2f}% | {m.f1_score*100:>7.2f}% | {m.roc_auc:>6.4f}")
        print("=" * 80 + "\n")
