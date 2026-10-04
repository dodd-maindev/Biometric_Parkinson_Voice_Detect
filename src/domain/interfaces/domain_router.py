"""Interface defining domain routing operations for multi-domain speech screening."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import numpy as np


class IDomainRouter(ABC):
    """Abstract interface for domain identification and gating across audio sources."""

    @abstractmethod
    def fit(self, features: np.ndarray, domain_labels: np.ndarray) -> None:
        """Fit domain discriminator model using acoustic representation matrices."""

    @abstractmethod
    def predict_domain(self, features: np.ndarray) -> np.ndarray:
        """Predict integer domain identifier for each input audio feature vector."""

    @abstractmethod
    def predict_domain_probability(self, features: np.ndarray) -> np.ndarray:
        """Estimate class probability distributions across distinct acoustic domains."""

    @abstractmethod
    def save(self, file_path: Path) -> None:
        """Serialize trained domain router model state to specified disk destination."""

    @classmethod
    @abstractmethod
    def load(cls, file_path: Path) -> Any:
        """Restore trained domain router instance from persisted storage."""
