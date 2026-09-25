"""Audio segmentation implementation replicating Hossain et al. (Neurol. Int. 2026)."""

from pathlib import Path
from typing import List
from pydub import AudioSegment
from pydub.silence import split_on_silence
from src.domain.entities.audio_sample import AudioSample
from src.domain.interfaces.audio_segmenter import IAudioSegmenter


class SilenceAudioSegmenter(IAudioSegmenter):
    """Segments speech recordings into vocal intervals (~800 chunks total)."""

    def __init__(
        self,
        minimum_silence_milliseconds: int = 500,
        silence_threshold_dbfs: int = -16,
        keep_silence_padding_milliseconds: int = 100,
        seek_step_milliseconds: int = 10,
    ) -> None:
        """Initialize silence threshold and search step parameters."""
        self._minimum_silence_ms = minimum_silence_milliseconds
        self._silence_threshold_dbfs = silence_threshold_dbfs
        self._padding_ms = keep_silence_padding_milliseconds
        self._seek_step_ms = seek_step_milliseconds

    def segment(
        self,
        sample: AudioSample,
        output_directory: Path,
    ) -> List[AudioSample]:
        """Split audio file on detected silences after loudness normalization."""
        output_directory.mkdir(parents=True, exist_ok=True)

        existing_chunks = sorted(output_directory.glob(f"{sample.file_path.stem}_seg_*.wav"))
        if len(existing_chunks) > 1:
            return [
                AudioSample(
                    subject_id=sample.subject_id,
                    file_path=chunk_file,
                    is_parkinson=sample.is_parkinson,
                    task_type=sample.task_type,
                    segment_index=idx,
                )
                for idx, chunk_file in enumerate(existing_chunks)
            ]

        # Normalization brings speech peaks to 0 dBFS so silence is reliably < -16 dBFS
        raw_audio = AudioSegment.from_file(str(sample.file_path)).normalize()
        chunks = split_on_silence(
            raw_audio,
            min_silence_len=self._minimum_silence_ms,
            silence_thresh=self._silence_threshold_dbfs,
            keep_silence=self._padding_ms,
            seek_step=self._seek_step_ms,
        )

        # Dynamic fallback threshold if recording had high background ambient noise
        if len(chunks) < 2:
            chunks = split_on_silence(
                raw_audio,
                min_silence_len=self._minimum_silence_ms,
                silence_thresh=int(raw_audio.dBFS - 12),
                keep_silence=self._padding_ms,
                seek_step=self._seek_step_ms,
            )

        segmented_samples: List[AudioSample] = []
        for index, chunk in enumerate(chunks):
            if len(chunk) < 400:
                continue

            chunk_filename = f"{sample.file_path.stem}_seg_{index:03d}.wav"
            chunk_path = output_directory / chunk_filename
            chunk.export(str(chunk_path), format="wav")

            segmented_samples.append(
                AudioSample(
                    subject_id=sample.subject_id,
                    file_path=chunk_path,
                    is_parkinson=sample.is_parkinson,
                    task_type=sample.task_type,
                    segment_index=index,
                )
            )

        return segmented_samples
