"""Support Vector Machine with standardization, balanced weights, and tuning."""

from pathlib import Path
from typing import List, Optional
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.domain.interfaces.classifier_model import IClassifierModel


class SupportVectorClassifier(IClassifierModel):
    """Encapsulates SVC with class balancing and GridSearchCV tuning."""

    _PARAMETER_GRID = {
        "C": [1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
        "gamma": ["scale", "auto", 0.02, 0.03, 0.04, 0.05, 0.06],
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
        self._default_c, self._kernel, self._gamma = c_regularization, kernel_type, gamma_parameter
        self._classifier: Optional[SVC] = None

    def fit(
        self, features: np.ndarray, labels: np.ndarray,
        subject_groups: Optional[List[str]] = None,
    ) -> None:
        """Standardize features and fit SVM with balanced class weights."""
        scaled = self._scaler.fit_transform(features)
        pd_count = int(np.sum(labels == 1))
        hc_count = int(np.sum(labels == 0))
        print(f"    SVM fit: {len(labels)} samples (PD={pd_count}, HC={hc_count})")
        if self._enable_grid_search and len(labels) >= 50:
            self._fit_with_grid_search(scaled, labels, subject_groups)
        else:
            self._fit_default(scaled, labels)

    def _fit_with_grid_search(
        self, features: np.ndarray, labels: np.ndarray,
        subject_groups: Optional[List[str]] = None,
    ) -> None:
        """Grid search with group-aware CV to ensure subject-independent tuning."""
        cv = (
            StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self._seed)
            if subject_groups is not None
            else StratifiedKFold(n_splits=10, shuffle=True, random_state=self._seed)
        )
        grid = GridSearchCV(
            SVC(kernel=self._kernel, probability=True, class_weight="balanced", random_state=self._seed),
            param_grid=self._PARAMETER_GRID, cv=cv, scoring="f1", n_jobs=-1,
        )
        grid.fit(features, labels, groups=subject_groups)
        self._classifier = grid.best_estimator_
        print(f"    GridSearchCV best params: {grid.best_params_}")
        print(f"    GridSearchCV best F1 (CV): {grid.best_score_:.4f}")

    def _fit_default(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fit with default hyperparameters."""
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

    def save(self, destination_path: Path) -> None:
        """Serialize scaler and fitted classifier to disk via joblib."""
        if self._classifier is None:
            raise RuntimeError("Cannot save an unfitted classifier.")
        target = Path(destination_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self._scaler, "classifier": self._classifier}, target)
        print(f"    Saved model checkpoint to: {target}")
