"""Pipeline for evaluating Self-Supervised Speech Representations."""

from pathlib import Path
from typing import List, Optional
from src.application.services.experiment_logger_service import ExperimentLoggerService
from src.application.services.feature_extraction_service import FeatureExtractionService
from src.application.strategies.subject_split_validation_strategy import (
    SubjectSplitValidationStrategy,
)
from src.domain.entities.audio_sample import AudioSample
from src.domain.entities.evaluation_metrics import EvaluationMetrics
from src.domain.interfaces.validation_strategy import IValidationStrategy
from src.infrastructure.audio.silence_audio_segmenter import SilenceAudioSegmenter
from src.infrastructure.extractors.frozen_speech_encoder import FrozenSpeechEncoder
from src.infrastructure.extractors.layer_weighted_encoder import LayerWeightedEncoder
from src.infrastructure.models.support_vector_classifier import (
    SupportVectorClassifier,
)


class SelfSupervisedEvaluationPipeline:
    """Evaluates foundation speech models (Wav2Vec2, WavLM) on vocal segments."""

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        layer_index: Optional[int] = None,
        segmentation_cache_directory: Path = Path("./data/processed/segments"),
        validation_strategy: Optional[IValidationStrategy] = None,
        output_model_path: Optional[Path] = None,
    ) -> None:
        """Initialize SSL pipeline with encoder, cache paths, and validator."""
        self._model_name = model_name
        self._layer_index = layer_index
        self._cache_dir = Path(segmentation_cache_directory)
        self._segmenter = SilenceAudioSegmenter()
        self._validator = validation_strategy or SubjectSplitValidationStrategy()
        self._output_model_path = output_model_path
        self._extractor = (
            LayerWeightedEncoder(model_name=model_name, selected_layer_index=layer_index)
            if layer_index is not None
            else FrozenSpeechEncoder(model_name=model_name)
        )

    def run(self, raw_samples: List[AudioSample]) -> EvaluationMetrics:
        """Execute vocal chunk segmentation, SSL feature extraction, and evaluation."""
        processed = self._segment_audio(raw_samples)
        features, labels, samples = self._extract_ssl_features(processed)
        return self._evaluate_classifier(features, labels, samples)

    def _segment_audio(self, raw_samples: List[AudioSample]) -> List[AudioSample]:
        """Segment raw audio or reuse cached vocal chunks."""
        ExperimentLoggerService.log_section("AUDIO SEGMENTATION (SSL)")
        processed: List[AudioSample] = []
        for sample in raw_samples:
            chunks = self._segmenter.segment(sample, self._cache_dir)
            processed.extend(chunks if chunks else [sample])
        print(f"  Processed {len(processed)} vocal chunks from {len(raw_samples)} recordings")
        return processed

    def _extract_ssl_features(self, samples: List[AudioSample]):
        """Extract SSL latent embeddings with disk caching."""
        ExperimentLoggerService.log_section(f"SSL EMBEDDING ({self._model_name})")
        model_slug = self._model_name.replace("/", "_")
        layer_slug = f"layer_{self._layer_index}" if self._layer_index is not None else "last_layer"
        cache_file = self._cache_dir.parent / f"features_ssl_{model_slug}_{layer_slug}.npz"
        service = FeatureExtractionService([self._extractor])
        features, labels, _ = service.extract_dataset(samples, cache_file=cache_file)
        return features, labels, samples

    def _evaluate_classifier(self, features, labels, samples) -> EvaluationMetrics:
        """Fit subject-independent SVM classifier on SSL latent representations."""
        ExperimentLoggerService.log_section("SSL MODEL EVALUATION")
        classifier = SupportVectorClassifier(enable_grid_search=True)
        metrics = self._validator.evaluate(classifier, features, labels, samples)
        if self._output_model_path is not None:
            classifier.save(self._output_model_path)
        return metrics
