"""Pipeline for replicating the MDVR-KCL baseline from Hossain et al. (2026)."""

import shutil
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
from src.infrastructure.extractors.gammatone_cepstral_extractor import (
    GammatoneCepstralExtractor,
)
from src.infrastructure.extractors.praat_acoustic_extractor import PraatAcousticExtractor
from src.infrastructure.models.support_vector_classifier import SupportVectorClassifier


class BaselineReplicationPipeline:
    """Executes Acoustic + GTCC extraction and evaluation on Read-Text task."""

    def __init__(
        self,
        segmentation_cache_directory: Path,
        validation_strategy: Optional[IValidationStrategy] = None,
        clear_segment_cache: bool = False,
    ) -> None:
        """Initialize pipeline with segment cache directory and validation strategy."""
        self._cache_dir = Path(segmentation_cache_directory)
        self._segmenter = SilenceAudioSegmenter()
        self._validator = validation_strategy or SubjectSplitValidationStrategy()
        if clear_segment_cache and self._cache_dir.exists():
            shutil.rmtree(self._cache_dir)
            print(f"  Cleared segment cache: {self._cache_dir}")

    def run(
        self, raw_samples: List[AudioSample], use_segmentation: bool = True,
    ) -> EvaluationMetrics:
        """Execute baseline: segmentation, Acoustic+GTCC extraction, evaluation."""
        processed = self._segment_audio(raw_samples, use_segmentation)
        self._log_dataset_overview(processed)
        features, labels, samples = self._extract_features(processed)
        return self._evaluate_classifier(features, labels, samples)

    def _segment_audio(
        self, raw_samples: List[AudioSample], use_segmentation: bool,
    ) -> List[AudioSample]:
        """Segment raw audio files into vocal chunks if enabled."""
        if not use_segmentation:
            return raw_samples
        ExperimentLoggerService.log_section("AUDIO SEGMENTATION")
        print(f"  Input: {len(raw_samples)} raw recordings | Target: ~808 vocal chunks")
        processed: List[AudioSample] = []
        for sample in raw_samples:
            chunks = self._segmenter.segment(sample, self._cache_dir)
            processed.extend(chunks if chunks else [sample])
        print(f"\n  Result: {len(processed)} chunks from {len(raw_samples)} files")
        print(f"  Average: {len(processed)/len(raw_samples):.1f} chunks/file")
        return processed

    def _log_dataset_overview(self, samples: List[AudioSample]) -> None:
        """Log dataset composition after segmentation."""
        subjects = [s.subject_id for s in samples]
        labels = [s.label for s in samples]
        ExperimentLoggerService.log_dataset_summary(len(samples), subjects, labels)
        ExperimentLoggerService.log_per_subject_distribution(subjects, labels)

    def _extract_features(self, samples: List[AudioSample]):
        """Extract Acoustic + GTCC features and filter unvoiced segments."""
        ExperimentLoggerService.log_section("FEATURE EXTRACTION (Acoustic + GTCC)")
        extractors = [
            PraatAcousticExtractor(),
            GammatoneCepstralExtractor(number_of_coefficients=13),
        ]
        features, labels, _ = FeatureExtractionService(extractors).extract_dataset(samples)
        valid_idx = [i for i in range(len(features)) if features[i, 0] > 0.0]
        if len(valid_idx) < len(features):
            filtered = len(features) - len(valid_idx)
            print(f"  Filtered {filtered} unvoiced segments (pitch_mean == 0)")
            features = features[valid_idx]
            labels = labels[valid_idx]
            samples = [samples[i] for i in valid_idx]
        return features, labels, samples

    def _evaluate_classifier(self, features, labels, samples) -> EvaluationMetrics:
        """Train and evaluate a balanced SVM with hyperparameter tuning."""
        ExperimentLoggerService.log_section("MODEL TRAINING & EVALUATION")
        classifier = SupportVectorClassifier(enable_grid_search=True)
        return self._validator.evaluate(classifier, features, labels, samples)
