"""Pipeline for evaluating Hybrid Multimodal Fusion (Acoustic + GTCC + SSL)."""

from pathlib import Path
from typing import List, Optional
import numpy as np
from sklearn.decomposition import PCA
from src.application.pipelines.baseline_replication_pipeline import (
    BaselineReplicationPipeline,
)
from src.application.pipelines.self_supervised_evaluation_pipeline import (
    SelfSupervisedEvaluationPipeline,
)
from src.application.services.experiment_logger_service import (
    ExperimentLoggerService,
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


class HybridFusionEvaluationPipeline:
    """Fuses handcrafted acoustic-cepstral features with foundation SSL representations."""

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        layer_index: Optional[int] = None,
        ssl_pca_components: int = 32,
        segmentation_cache_directory: Path = Path("./data/processed/segments"),
        validation_strategy: Optional[IValidationStrategy] = None,
        output_model_path: Optional[Path] = None,
    ) -> None:
        """Initialize hybrid pipeline with SSL backbone and PCA configuration."""
        self._model_name = model_name
        self._layer_index = layer_index
        self._n_components = ssl_pca_components
        self._cache_dir = Path(segmentation_cache_directory)
        self._validator = validation_strategy or SubjectSplitValidationStrategy()
        self._output_model_path = output_model_path
        self._baseline_pipe = BaselineReplicationPipeline(self._cache_dir, self._validator)
        self._ssl_pipe = SelfSupervisedEvaluationPipeline(
            model_name=model_name, layer_index=layer_index,
            segmentation_cache_directory=self._cache_dir, validation_strategy=self._validator,
        )

    def run(self, raw_samples: List[AudioSample]) -> EvaluationMetrics:
        """Extract baseline and SSL features, fuse representations, and evaluate."""
        title = f"HYBRID FUSION: Baseline (24) + {self._model_name} (PCA-{self._n_components})"
        ExperimentLoggerService.log_section(title)
        base_samples = self._baseline_pipe._segment_audio(raw_samples, use_segmentation=True)
        base_feat, base_labels, samples = self._baseline_pipe._extract_features(base_samples)
        ssl_feat, _, _ = self._ssl_pipe._extract_ssl_features(base_samples)

        pca = PCA(n_components=self._n_components, random_state=42)
        ssl_reduced = pca.fit_transform(ssl_feat)
        explained = np.sum(pca.explained_variance_ratio_) * 100
        print(f"  PCA reduced SSL: {ssl_feat.shape[1]} -> {self._n_components} dims ({explained:.1f}% var)")

        fused_features = np.hstack([base_feat, ssl_reduced])
        print(f"  Fused matrix: {fused_features.shape} (24 Acoustic/GTCC + {self._n_components} SSL)")

        ExperimentLoggerService.log_section("HYBRID MODEL EVALUATION")
        classifier = SupportVectorClassifier(enable_grid_search=True)
        metrics = self._validator.evaluate(classifier, fused_features, base_labels, samples)
        if self._output_model_path is not None:
            classifier.save(self._output_model_path)
        return metrics
