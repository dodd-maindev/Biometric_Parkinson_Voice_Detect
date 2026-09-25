"""Audio segmentation replicating Hossain et al. (~816 chunks from 73 files).

Paper method (Section 2.2.2): PyDub silence detection at -16 dBFS
on raw audio with 0.5s minimum silence gap. No normalization.
"""

from pathlib import Path
from typing import List
from pydub import AudioSegment
from pydub.silence import split_on_silence
from src.domain.entities.audio_sample import AudioSample
from src.domain.interfaces.audio_segmenter import IAudioSegmenter


class SilenceAudioSegmenter(IAudioSegmenter):
    """Segments speech recordings into vocal intervals via silence detection."""

    _FIXED_SILENCE_THRESHOLD_DBFS: int = -16
    _MINIMUM_SILENCE_MS: int = 500
    _KEEP_SILENCE_PADDING_MS: int = 150

    def segment(
        self, sample: AudioSample, output_directory: Path,
    ) -> List[AudioSample]:
        """Split audio on natural pauses using paper's fixed threshold."""
        output_directory.mkdir(parents=True, exist_ok=True)
        cached = sorted(output_directory.glob(f"{sample.file_path.stem}_seg_*.wav"))

        if cached:
            return self._build_from_cache(sample, cached)

        raw_audio = AudioSegment.from_file(str(sample.file_path))

        chunks = split_on_silence(
            raw_audio,
            min_silence_len=self._MINIMUM_SILENCE_MS,
            silence_thresh=self._FIXED_SILENCE_THRESHOLD_DBFS,
            keep_silence=self._KEEP_SILENCE_PADDING_MS,
            seek_step=10,
        )

        if not chunks:
            chunks = [raw_audio]

        self._log_segmentation_result(sample, raw_audio, chunks)
        return self._export_chunks(sample, chunks, output_directory)

    def _log_segmentation_result(
        self, sample: AudioSample, raw: AudioSegment,
        chunks: List[AudioSegment],
    ) -> None:
        """Print segmentation statistics for diagnostic purposes."""
        durations = [len(c) / 1000.0 for c in chunks]
        print(
            f"  {sample.file_path.name}: {len(raw)/1000:.1f}s -> "
            f"{len(chunks)} chunks (raw_dBFS={raw.dBFS:.1f}, "
            f"thresh={self._FIXED_SILENCE_THRESHOLD_DBFS}, "
            f"dur=[{min(durations):.1f}s-{max(durations):.1f}s])"
        )

    def _build_from_cache(
        self, original: AudioSample, cached_files: List[Path],
    ) -> List[AudioSample]:
        """Reconstruct AudioSample entities from previously cached segments."""
        return [
            AudioSample(
                subject_id=original.subject_id, file_path=f,
                is_parkinson=original.is_parkinson,
                task_type=original.task_type, segment_index=i,
            )
            for i, f in enumerate(cached_files)
        ]

    def _export_chunks(
        self, original: AudioSample,
        chunks: List[AudioSegment], output_dir: Path,
    ) -> List[AudioSample]:
        """Export audio chunks to disk and return AudioSample entities."""
        results: List[AudioSample] = []
        for idx, chunk in enumerate(chunks):
            path = output_dir / f"{original.file_path.stem}_seg_{idx:03d}.wav"
            chunk.export(str(path), format="wav")
            results.append(AudioSample(
                subject_id=original.subject_id, file_path=path,
                is_parkinson=original.is_parkinson,
                task_type=original.task_type, segment_index=idx,
            ))
        return results
