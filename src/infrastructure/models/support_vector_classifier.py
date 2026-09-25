"""Support Vector Machine classifier wrapper implementing IClassifierModel."""

from typing import Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.domain.interfaces.classifier_model import IClassifierModel


class SupportVectorClassifier(IClassifierModel):
    """Encapsulates Scikit-learn Support Vector Classifier with standardization."""

    def __init__(
        self,
        c_regularization: float = 1.0,
        kernel_type: str = "rbf",
        gamma_parameter: str = "scale",
        random_seed: int = 42,
    ) -> None:
        """Initialize SVM hyperparameters and preprocessor."""
        self._scaler = StandardScaler()
        self._classifier = SVC(
            C=c_regularization,
            kernel=kernel_type,
            gamma=gamma_parameter,
            probability=True,
            random_state=random_seed,
        )
        self._is_fitted: bool = False

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Standardize features and fit the SVM decision boundary."""
        scaled_features = self._scaler.fit_transform(features)
        self._classifier.fit(scaled_features, labels)
        self._is_fitted = True

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict binary class labels for unseen features."""
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted prior to running predictions.")
        scaled_features = self._scaler.transform(features)
        return self._classifier.predict(scaled_features)

    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        """Return positive class probability scores for unseen features."""
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted prior to running predictions.")
        scaled_features = self._scaler.transform(features)
        # Class 1 corresponds to Parkinson's Disease
        return self._classifier.predict_proba(scaled_features)[:, 1]
