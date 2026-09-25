"""Audio segmentation replicating Hossain et al. (~800 chunks, ~11 per file)."""

from pathlib import Path
from typing import List
from pydub import AudioSegment
from pydub.silence import split_on_silence
from src.domain.entities.audio_sample import AudioSample
from src.domain.interfaces.audio_segmenter import IAudioSegmenter


class SilenceAudioSegmenter(IAudioSegmenter):
    """Segments speech recordings into vocal intervals matching the ~808 segments benchmark."""

    def __init__(
        self,
        minimum_silence_milliseconds: int = 500,
        silence_threshold_dbfs: int = -16,
        keep_silence_padding_milliseconds: int = 150,
        minimum_chunk_duration_milliseconds: int = 1500,
    ) -> None:
        """Initialize silence threshold and duration filter parameters."""
        self._minimum_silence_ms = minimum_silence_milliseconds
        self._silence_threshold_dbfs = silence_threshold_dbfs
        self._padding_ms = keep_silence_padding_milliseconds
        self._min_chunk_duration_ms = minimum_chunk_duration_milliseconds

    def segment(
        self,
        sample: AudioSample,
        output_directory: Path,
    ) -> List[AudioSample]:
        """Split audio file on natural pauses into robust, multi-second vocal chunks."""
        output_directory.mkdir(parents=True, exist_ok=True)

        existing_chunks = sorted(output_directory.glob(f"{sample.file_path.stem}_seg_*.wav"))
        if len(existing_chunks) >= 4:
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

        raw_audio = AudioSegment.from_file(str(sample.file_path)).normalize()
        # Detect pauses between spoken phrases
        chunks = split_on_silence(
            raw_audio,
            min_silence_len=self._minimum_silence_ms,
            silence_thresh=self._silence_threshold_dbfs,
            keep_silence=self._padding_ms,
            seek_step=10,
        )

        # Discard sub-second micro-clicks to ensure stable acoustic phonation
        valid_chunks = [c for c in chunks if len(c) >= self._min_chunk_duration_ms]
        if not valid_chunks:
            valid_chunks = [raw_audio]

        segmented_samples: List[AudioSample] = []
        for index, chunk in enumerate(valid_chunks):
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
