"""Extreme Gradient Boosting (XGBoost) classifier implementing IClassifierModel."""

from typing import Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from src.domain.interfaces.classifier_model import IClassifierModel


class ExtremeGradientBoostingClassifier(IClassifierModel):
    """Encapsulates XGBoost tree ensemble classifier with automatic feature scaling."""

    def __init__(
        self,
        number_of_estimators: int = 100,
        learning_rate: float = 0.1,
        max_tree_depth: int = 5,
        subsample_ratio: float = 0.8,
        random_seed: int = 42,
    ) -> None:
        """Initialize XGBoost hyperparameters."""
        self._scaler = StandardScaler()
        self._classifier = XGBClassifier(
            n_estimators=number_of_estimators,
            learning_rate=learning_rate,
            max_depth=max_tree_depth,
            subsample=subsample_ratio,
            random_state=random_seed,
            eval_metric="logloss",
        )
        self._is_fitted: bool = False

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Standardize inputs and fit the gradient boosted tree ensemble."""
        scaled_features = self._scaler.fit_transform(features)
        self._classifier.fit(scaled_features, labels)
        self._is_fitted = True

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict binary diagnostic class labels."""
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted prior to running predictions.")
        scaled_features = self._scaler.transform(features)
        return self._classifier.predict(scaled_features)

    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        """Compute estimated probabilities of Parkinson's Disease."""
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted prior to running predictions.")
        scaled_features = self._scaler.transform(features)
        return self._classifier.predict_proba(scaled_features)[:, 1]
