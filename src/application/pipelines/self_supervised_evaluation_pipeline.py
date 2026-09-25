"""Pipeline for evaluating Self-Supervised Speech Representations via LOSOCV."""

from typing import List, Optional
from src.application.services.feature_extraction_service import (
    FeatureExtractionService,
)
from src.application.strategies.leave_one_subject_out_strategy import (
    LeaveOneSubjectOutStrategy,
)
from src.domain.entities.audio_sample import AudioSample
from src.domain.entities.evaluation_metrics import EvaluationMetrics
from src.infrastructure.extractors.frozen_speech_encoder import (
    FrozenSpeechEncoder,
)
from src.infrastructure.extractors.layer_weighted_encoder import (
    LayerWeightedEncoder,
)
from src.infrastructure.models.support_vector_classifier import (
    SupportVectorClassifier,
)


class SelfSupervisedEvaluationPipeline:
    """Evaluates frozen foundation speech models to test pathological representation capacity."""

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        layer_index: Optional[int] = None,
    ) -> None:
        """Initialize pipeline with target SSL model and optional probed layer."""
        if layer_index is not None:
            self._extractor = LayerWeightedEncoder(
                model_name=model_name,
                selected_layer_index=layer_index,
            )
        else:
            self._extractor = FrozenSpeechEncoder(model_name=model_name)

        self._validator = LeaveOneSubjectOutStrategy()

    def run(self, samples: List[AudioSample]) -> EvaluationMetrics:
        """Execute SSL feature extraction and subject-independent classification."""
        service = FeatureExtractionService([self._extractor])
        features, labels, _ = service.extract_dataset(samples)

        classifier = SupportVectorClassifier(c_regularization=1.0, kernel_type="rbf")
        metrics = self._validator.evaluate(classifier, features, labels, samples)
        return metrics
