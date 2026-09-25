"""Domain interfaces package defining contracts for clean architecture."""

from src.domain.interfaces.audio_segmenter import IAudioSegmenter
from src.domain.interfaces.classifier_model import IClassifierModel
from src.domain.interfaces.feature_extractor import IFeatureExtractor
from src.domain.interfaces.validation_strategy import IValidationStrategy

__all__ = [
    "IAudioSegmenter",
    "IClassifierModel",
    "IFeatureExtractor",
    "IValidationStrategy",
]
