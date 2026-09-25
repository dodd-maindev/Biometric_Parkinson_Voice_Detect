"""Interface for classification models."""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class IClassifierModel(ABC):
    """Abstract contract for binary classification models."""

    @abstractmethod
    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fit the model to training features and ground truth labels.

        Args:
            features: 2D array of shape (number_of_samples, number_of_features).
            labels: 1D array of shape (number_of_samples,) with binary integer labels.
        """
        pass

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict binary discrete labels for given feature samples.

        Args:
            features: 2D array of shape (number_of_samples, number_of_features).

        Returns:
            1D array of predicted binary labels (0 or 1).
        """
        pass

    @abstractmethod
    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        """Predict probability scores for the positive class (Parkinson's Disease).

        Args:
            features: 2D array of shape (number_of_samples, number_of_features).

        Returns:
            1D array of continuous probability scores between 0.0 and 1.0.
        """
        pass
