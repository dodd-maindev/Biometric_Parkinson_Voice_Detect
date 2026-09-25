"""Pipeline for replicating the MDVR-KCL baseline from Hossain et al. (2026)."""

import shutil
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from src.application.services.experiment_logger_service import (
    ExperimentLoggerService,
)
from src.application.services.feature_extraction_service import (
    FeatureExtractionService,
)
from src.application.strategies.subject_split_validation_strategy import (
    SubjectSplitValidationStrategy,
)
from src.domain.entities.audio_sample import AudioSample
from src.domain.entities.evaluation_metrics import EvaluationMetrics
from src.domain.interfaces.validation_strategy import IValidationStrategy
from src.infrastructure.audio.silence_audio_segmenter import (
    SilenceAudioSegmenter,
)
from src.infrastructure.extractors.gammatone_cepstral_extractor import (
    GammatoneCepstralExtractor,
)
from src.infrastructure.extractors.praat_acoustic_extractor import (
    PraatAcousticExtractor,
)
from src.infrastructure.models.support_vector_classifier import (
    SupportVectorClassifier,
)


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
        self,
        raw_samples: List[AudioSample],
        use_segmentation: bool = True,
    ) -> EvaluationMetrics:
        """Execute baseline: segmentation, Acoustic+GTCC extraction, evaluation."""
        processed = self._segment_audio(raw_samples, use_segmentation)
        self._log_dataset_overview(processed)
        features, labels, _ = self._extract_features(processed)
        return self._evaluate_classifier(features, labels, processed)

    def _segment_audio(self, raw_samples, use_segmentation):
        """Segment raw audio files into vocal chunks if enabled."""
        if not use_segmentation:
            return raw_samples

        ExperimentLoggerService.log_section("AUDIO SEGMENTATION")
        print(f"  Input: {len(raw_samples)} raw recordings")
        print(f"  Target: ~808 vocal chunks (paper benchmark)")

        processed: List[AudioSample] = []
        for sample in raw_samples:
            chunks = self._segmenter.segment(sample, self._cache_dir)
            processed.extend(chunks if chunks else [sample])

        print(f"\n  Result: {len(processed)} chunks from {len(raw_samples)} files")
        print(f"  Average: {len(processed)/len(raw_samples):.1f} chunks/file")
        return processed

    def _log_dataset_overview(self, samples):
        """Log dataset composition after segmentation."""
        subjects = [s.subject_id for s in samples]
        labels = [s.label for s in samples]
        ExperimentLoggerService.log_dataset_summary(len(samples), subjects, labels)
        ExperimentLoggerService.log_per_subject_distribution(subjects, labels)

    def _extract_features(self, samples):
        """Extract Acoustic + GTCC features (24 dimensions)."""
        ExperimentLoggerService.log_section("FEATURE EXTRACTION (Acoustic + GTCC)")
        extractors = [
            PraatAcousticExtractor(),
            GammatoneCepstralExtractor(number_of_coefficients=13),
        ]
        service = FeatureExtractionService(extractors)
        return service.extract_dataset(samples)

    def _evaluate_classifier(self, features, labels, samples):
        """Train and evaluate a balanced SVM with hyperparameter tuning."""
        ExperimentLoggerService.log_section("MODEL TRAINING & EVALUATION")
        classifier = SupportVectorClassifier(enable_grid_search=True)
        return self._validator.evaluate(classifier, features, labels, samples)
