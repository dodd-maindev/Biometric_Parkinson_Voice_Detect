"""Pipeline for replicating the MDVR-KCL baseline from Hossain et al. (2026)."""

from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from src.application.services.feature_extraction_service import (
    FeatureExtractionService,
)
from src.application.strategies.leave_one_subject_out_strategy import (
    LeaveOneSubjectOutStrategy,
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
    """Executes feature extraction (Acoustic + GTCC) and evaluation on Read-Text task."""

    def __init__(
        self,
        segmentation_cache_directory: Path,
        validation_strategy: Optional[IValidationStrategy] = None,
    ) -> None:
        """Initialize pipeline with segment cache directory and validation strategy."""
        self._cache_dir = Path(segmentation_cache_directory)
        self._segmenter = SilenceAudioSegmenter()
        self._validator = validation_strategy or SubjectSplitValidationStrategy()

    def run(
        self,
        raw_samples: List[AudioSample],
        use_segmentation: bool = True,
    ) -> EvaluationMetrics:
        """Execute baseline: segmentation, Acoustic+GTCC extraction, and evaluation."""
        processed_samples: List[AudioSample] = []

        if use_segmentation:
            print(f"Segmenting {len(raw_samples)} audio files into ~800 vocal chunks...")
            for sample in tqdm(raw_samples, desc="Segmenting speech audio"):
                chunks = self._segmenter.segment(sample, self._cache_dir)
                processed_samples.extend(chunks if len(chunks) > 0 else [sample])
            print(f"Generated {len(processed_samples)} valid vocal chunks from {len(raw_samples)} files.")
        else:
            processed_samples = raw_samples

        print(f"Extracting 24 Acoustic + GTCC features for {len(processed_samples)} segments...")
        extractors = [
            PraatAcousticExtractor(),
            GammatoneCepstralExtractor(number_of_coefficients=13),
        ]
        service = FeatureExtractionService(extractors)
        features, labels, _ = service.extract_dataset(processed_samples)

        print("Fitting balanced Support Vector Machine with hyperparameter optimization...")
        classifier = SupportVectorClassifier(enable_grid_search=True)
        metrics = self._validator.evaluate(classifier, features, labels, processed_samples)
        return metrics
