"""Acoustic domain router for discriminating input audio dataset distribution."""

from pathlib import Path
from typing import Optional
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.domain.interfaces.domain_router import IDomainRouter


class AcousticDomainRouter(IDomainRouter):
    """Calibrated discriminator predicting probability of acoustic domain origin."""

    def __init__(self, random_seed: int = 42) -> None:
        """Initialize scaler and logistic regression classifier with seed."""
        self._scaler = StandardScaler()
        self._seed = random_seed
        self._classifier: Optional[LogisticRegression] = None

    def fit(self, features: np.ndarray, domain_labels: np.ndarray) -> None:
        """Fit domain discriminator on scaled acoustic representations."""
        scaled = self._scaler.fit_transform(features)
        self._classifier = LogisticRegression(
            C=1.0, max_iter=1000, random_state=self._seed, solver="lbfgs"
        )
        self._classifier.fit(scaled, domain_labels)
        train_acc = self._classifier.score(scaled, domain_labels)
        print(f"  Domain Router fit: {len(domain_labels)} samples | Train Acc: {train_acc*100:.2f}%")

    def predict_domain(self, features: np.ndarray) -> np.ndarray:
        """Predict discrete domain index (0 for MDVR-KCL, 1 for Voice_Dataset)."""
        if self._classifier is None:
            raise RuntimeError("Domain router must be fitted before predicting.")
        return self._classifier.predict(self._scaler.transform(features))

    def predict_domain_probability(self, features: np.ndarray) -> np.ndarray:
        """Predict domain posterior probabilities matrix of shape (n_samples, n_domains)."""
        if self._classifier is None:
            raise RuntimeError("Domain router must be fitted before predicting.")
        return self._classifier.predict_proba(self._scaler.transform(features))

    def save(self, file_path: Path) -> None:
        """Save scaler and fitted router classifier state to joblib file."""
        if self._classifier is None:
            raise RuntimeError("Cannot save unfitted domain router.")
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self._scaler, "classifier": self._classifier}, target)
        print(f"  Saved domain router checkpoint to: {target}")

    @classmethod
    def load(cls, file_path: Path) -> "AcousticDomainRouter":
        """Restore domain router state from disk checkpoint."""
        data = joblib.load(file_path)
        instance = cls()
        instance._scaler = data["scaler"]
        instance._classifier = data["classifier"]
        print(f"  Loaded domain router checkpoint from: {file_path}")
        return instance
