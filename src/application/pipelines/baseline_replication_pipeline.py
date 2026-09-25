"""Pipeline for replicating the MDVR-KCL baseline from Hossain et al. (2026)."""

from pathlib import Path
from typing import List, Optional
from src.application.services.feature_extraction_service import (
    FeatureExtractionService,
)
from src.application.strategies.leave_one_subject_out_strategy import (
    LeaveOneSubjectOutStrategy,
)
from src.domain.entities.audio_sample import AudioSample
from src.domain.entities.evaluation_metrics import EvaluationMetrics
from src.domain.entities.speech_task_type import SpeechTaskType
from src.infrastructure.audio.silence_audio_segmenter import (
    SilenceAudioSegmenter,
)
from src.infrastructure.extractors.gammatone_cepstral_extractor import (
    GammatoneCepstralExtractor,
)
from src.infrastructure.extractors.mel_cepstral_extractor import (
    MelCepstralExtractor,
)
from src.infrastructure.extractors.praat_acoustic_extractor import (
    PraatAcousticExtractor,
)
from src.infrastructure.models.support_vector_classifier import (
    SupportVectorClassifier,
)


class BaselineReplicationPipeline:
    """Executes the exact feature extraction and SVM evaluation on Read-Text task."""

    def __init__(self, segmentation_cache_directory: Path) -> None:
        """Initialize pipeline with segment cache directory."""
        self._cache_dir = Path(segmentation_cache_directory)
        self._segmenter = SilenceAudioSegmenter()
        self._validator = LeaveOneSubjectOutStrategy()

    def run(
        self,
        raw_samples: List[AudioSample],
        use_segmentation: bool = True,
    ) -> EvaluationMetrics:
        """Execute baseline pipeline: segmentation, feature extraction, and LOSOCV evaluation."""
        processed_samples: List[AudioSample] = []

        if use_segmentation:
            for sample in raw_samples:
                chunks = self._segmenter.segment(sample, self._cache_dir)
                processed_samples.extend(chunks if len(chunks) > 0 else [sample])
        else:
            processed_samples = raw_samples

        # Extract Acoustic + GTCC features (best combination for Read-Text in paper: 95.45%)
        extractors = [
            PraatAcousticExtractor(),
            GammatoneCepstralExtractor(number_of_coefficients=13),
            MelCepstralExtractor(number_of_coefficients=13),
        ]
        service = FeatureExtractionService(extractors)
        features, labels, _ = service.extract_dataset(processed_samples)

        classifier = SupportVectorClassifier(c_regularization=1.0, kernel_type="rbf")
        metrics = self._validator.evaluate(classifier, features, labels, processed_samples)
        return metrics
