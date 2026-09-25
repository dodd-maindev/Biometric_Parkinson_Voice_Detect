"""Support Vector Machine with standardization, balanced weights, and tuning."""

from typing import Optional
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.domain.interfaces.classifier_model import IClassifierModel


class SupportVectorClassifier(IClassifierModel):
    """Encapsulates Scikit-learn SVC with class balancing and optional GridSearchCV."""

    def __init__(
        self,
        c_regularization: float = 10.0,
        kernel_type: str = "rbf",
        gamma_parameter: str = "scale",
        enable_grid_search: bool = True,
        random_seed: int = 42,
    ) -> None:
        """Initialize hyperparameters and preprocessor."""
        self._scaler = StandardScaler()
        self._enable_grid_search = enable_grid_search
        self._seed = random_seed
        self._default_c = c_regularization
        self._kernel = kernel_type
        self._gamma = gamma_parameter
        self._classifier: Optional[SVC] = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Standardize features and fit the SVM with balanced class weights."""
        scaled_features = self._scaler.fit_transform(features)

        if self._enable_grid_search and len(labels) >= 50:
            param_grid = {
                "C": [1.0, 10.0, 50.0, 100.0],
                "gamma": ["scale", "auto", 0.01, 0.05],
            }
            grid = GridSearchCV(
                SVC(kernel=self._kernel, probability=True, class_weight="balanced", random_state=self._seed),
                param_grid=param_grid,
                cv=5,
                scoring="f1",
                n_jobs=-1,
            )
            grid.fit(scaled_features, labels)
            self._classifier = grid.best_estimator_
        else:
            self._classifier = SVC(
                C=self._default_c,
                kernel=self._kernel,
                gamma=self._gamma,
                probability=True,
                class_weight="balanced",
                random_state=self._seed,
            )
            self._classifier.fit(scaled_features, labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict binary diagnostic class labels."""
        if self._classifier is None:
            raise RuntimeError("Classifier must be fitted prior to predicting.")
        scaled_features = self._scaler.transform(features)
        return self._classifier.predict(scaled_features)

    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        """Predict continuous probability scores for positive Parkinson class."""
        if self._classifier is None:
            raise RuntimeError("Classifier must be fitted prior to predicting.")
        scaled_features = self._scaler.transform(features)
        return self._classifier.predict_proba(scaled_features)[:, 1]
