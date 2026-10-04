"""Multi-domain routing pipeline orchestrating domain-specialized tri-modal models."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
from src.application.services.diagnostic_logger_service import DiagnosticLoggerService
from src.application.services.experiment_logger_service import ExperimentLoggerService
from src.application.services.metric_calculation_service import MetricCalculationService
from src.domain.entities.audio_sample import AudioSample
from src.domain.entities.evaluation_metrics import EvaluationMetrics
from src.infrastructure.models.acoustic_domain_router import AcousticDomainRouter


class MultiDomainRoutingPipeline:
    """Routes voice samples dynamically to domain-specific Tri-Modal ensemble models."""

    def __init__(
        self, kcl_checkpoint: Path, voice_checkpoint: Path,
        router_checkpoint: Optional[Path] = None,
    ) -> None:
        """Initialize routing pipeline with expert checkpoints and domain router."""
        self._kcl_ckpt, self._voice_ckpt = Path(kcl_checkpoint), Path(voice_checkpoint)
        self._router_ckpt = Path(router_checkpoint) if router_checkpoint else None
        self._router = AcousticDomainRouter()

    def train_router(
        self, feat_kcl: np.ndarray, feat_voice: np.ndarray, output_path: Path,
    ) -> None:
        """Train domain router on acoustic features from both datasets."""
        ExperimentLoggerService.log_section("TRAINING ACOUSTIC DOMAIN ROUTER")
        x_all = np.vstack([feat_kcl, feat_voice])
        y_domains = np.concatenate([np.zeros(len(feat_kcl), dtype=np.int32),
                                    np.ones(len(feat_voice), dtype=np.int32)])
        self._router.fit(x_all, y_domains)
        self._router.save(output_path)

    def evaluate_routing(
        self, features_dict: Dict[str, np.ndarray], labels: np.ndarray,
        expected_domain: int,
    ) -> EvaluationMetrics:
        """Evaluate routed inference across Tri-Modal expert ensembles."""
        ExperimentLoggerService.log_section(f"ROUTED INFERENCE (Expected Domain: {expected_domain})")
        kcl_model = joblib.load(self._kcl_ckpt)
        voice_model = joblib.load(self._voice_ckpt)
        router = AcousticDomainRouter.load(self._router_ckpt) if self._router_ckpt else self._router

        p_domains = router.predict_domain_probability(features_dict["base"])
        p_kcl_model = self._predict_tri_modal(kcl_model, features_dict)
        p_voice_model = self._predict_tri_modal(voice_model, features_dict)

        fused_probs = p_domains[:, 0] * p_kcl_model + p_domains[:, 1] * p_voice_model
        threshold = kcl_model["threshold"] if expected_domain == 0 else voice_model["threshold"]
        preds = (fused_probs >= threshold).astype(np.int32)

        predicted_domains = np.argmax(p_domains, axis=1)
        router_acc = np.mean(predicted_domains == expected_domain) * 100
        print(f"  Router Dispatched Accuracy to correct domain: {router_acc:.2f}%")
        DiagnosticLoggerService.log_confusion_matrix(labels, preds)
        return MetricCalculationService.calculate(labels, preds, fused_probs)

    def _predict_tri_modal(self, model_dict: dict, feat: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute fused probability prediction from constituent ensemble models."""
        w1, w2, w3 = model_dict["weights"]
        p_b = model_dict["baseline"].predict_probability(feat["base"])
        p_w = model_dict["wavlm"].predict_probability(feat["wavlm"])
        p_v = model_dict["w2v2"].predict_probability(feat["w2v2"])
        return w1 * p_b + w2 * p_w + w3 * p_v
