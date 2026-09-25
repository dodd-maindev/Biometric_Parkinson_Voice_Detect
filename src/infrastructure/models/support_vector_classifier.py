"""Support Vector Machine with standardization, balanced weights, and tuning."""

from typing import Optional
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.domain.interfaces.classifier_model import IClassifierModel


class SupportVectorClassifier(IClassifierModel):
    """Encapsulates SVC with class balancing and optional GridSearchCV."""

    _PARAMETER_GRID = {
        "C": [0.1, 1.0, 10.0, 50.0, 100.0],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.05],
    }

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
        pd_count = int(np.sum(labels == 1))
        hc_count = int(np.sum(labels == 0))
        print(f"    SVM fit: {len(labels)} samples (PD={pd_count}, HC={hc_count})")

        if self._enable_grid_search and len(labels) >= 50:
            self._fit_with_grid_search(scaled_features, labels)
        else:
            self._fit_default(scaled_features, labels)

    def _fit_with_grid_search(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Perform grid search cross-validation and log best parameters."""
        grid = GridSearchCV(
            SVC(
                kernel=self._kernel, probability=True,
                class_weight="balanced", random_state=self._seed,
            ),
            param_grid=self._PARAMETER_GRID,
            cv=5, scoring="f1", n_jobs=-1,
        )
        grid.fit(features, labels)
        self._classifier = grid.best_estimator_
        print(f"    GridSearchCV best params: {grid.best_params_}")
        print(f"    GridSearchCV best F1 (CV): {grid.best_score_:.4f}")

    def _fit_default(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fit with default hyperparameters when grid search is disabled."""
        self._classifier = SVC(
            C=self._default_c, kernel=self._kernel, gamma=self._gamma,
            probability=True, class_weight="balanced", random_state=self._seed,
        )
        self._classifier.fit(features, labels)
        print(f"    SVM default fit: C={self._default_c}, gamma={self._gamma}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict binary diagnostic class labels."""
        if self._classifier is None:
            raise RuntimeError("Classifier must be fitted prior to predicting.")
        return self._classifier.predict(self._scaler.transform(features))

    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        """Predict continuous probability scores for positive Parkinson class."""
        if self._classifier is None:
            raise RuntimeError("Classifier must be fitted prior to predicting.")
        return self._classifier.predict_proba(self._scaler.transform(features))[:, 1]
