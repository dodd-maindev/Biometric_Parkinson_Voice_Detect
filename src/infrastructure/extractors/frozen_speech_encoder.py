"""Frozen Self-Supervised Speech feature extractor (Wav2Vec2 / WavLM)."""

from typing import List, Optional
import numpy as np
import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModel
from src.domain.entities.audio_sample import AudioSample
from src.domain.interfaces.feature_extractor import IFeatureExtractor


class FrozenSpeechEncoder(IFeatureExtractor):
    """Extracts representation embeddings from a frozen foundation audio model."""

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        device: Optional[str] = None,
        target_sample_rate: int = 16000,
    ) -> None:
        """Initialize model backbone and feature processor."""
        self._target_sample_rate = target_sample_rate
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoFeatureExtractor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(self._device)
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad = False
        self._dimension = self._model.config.hidden_size

    @property
    def feature_dimension(self) -> int:
        """Return hidden embedding dimension."""
        return self._dimension

    @property
    def feature_names(self) -> List[str]:
        """Return generic names for latent embedding dimensions."""
        return [f"ssl_embed_{i+1}" for i in range(self._dimension)]

    def extract(self, sample: AudioSample) -> np.ndarray:
        """Extract pooled temporal representation vector from audio sample."""
        waveform, sample_rate = torchaudio.load(str(sample.file_path))
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != self._target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self._target_sample_rate)
            waveform = resampler(waveform)

        inputs = self._processor(
            waveform.squeeze(0).numpy(),
            sampling_rate=self._target_sample_rate,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(self._device)

        with torch.no_grad():
            outputs = self._model(input_values)
            # Global mean temporal pooling across sequence frames
            pooled = torch.mean(outputs.last_hidden_state, dim=1).squeeze(0)

        return pooled.cpu().numpy().astype(np.float32)
