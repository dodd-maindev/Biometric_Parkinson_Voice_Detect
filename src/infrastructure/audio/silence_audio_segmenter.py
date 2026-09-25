"""Audio segmentation replicating Hossain et al. (~808 chunks from 73 files)."""

from pathlib import Path
from typing import List
from pydub import AudioSegment
from pydub.silence import split_on_silence
from src.domain.entities.audio_sample import AudioSample
from src.domain.interfaces.audio_segmenter import IAudioSegmenter


class SilenceAudioSegmenter(IAudioSegmenter):
    """Segments speech recordings into vocal intervals via silence detection."""

    _MINIMUM_SILENCE_MS: int = 500
    _KEEP_SILENCE_PADDING_MS: int = 150
    _MIN_CHUNK_DURATION_MS: int = 1500

    def segment(
        self,
        sample: AudioSample,
        output_directory: Path,
    ) -> List[AudioSample]:
        """Split audio on natural pauses into robust multi-second vocal chunks."""
        output_directory.mkdir(parents=True, exist_ok=True)
        cache_pattern = f"{sample.file_path.stem}_seg_*.wav"
        cached_chunks = sorted(output_directory.glob(cache_pattern))

        if cached_chunks:
            return self._build_samples_from_cache(sample, cached_chunks)

        raw_audio = AudioSegment.from_file(str(sample.file_path))
        normalized_audio = raw_audio.normalize()
        threshold_dbfs = self._compute_adaptive_threshold(normalized_audio)
        duration_seconds = len(raw_audio) / 1000.0

        chunks = split_on_silence(
            normalized_audio,
            min_silence_len=self._MINIMUM_SILENCE_MS,
            silence_thresh=threshold_dbfs,
            keep_silence=self._KEEP_SILENCE_PADDING_MS,
            seek_step=10,
        )

        valid_chunks = [c for c in chunks if len(c) >= self._MIN_CHUNK_DURATION_MS]
        if not valid_chunks:
            valid_chunks = [normalized_audio]

        chunk_durations = [len(c) / 1000.0 for c in valid_chunks]
        print(
            f"  {sample.file_path.name}: {duration_seconds:.1f}s -> "
            f"{len(valid_chunks)} chunks "
            f"(thresh={threshold_dbfs}dBFS, dBFS={normalized_audio.dBFS:.1f}, "
            f"chunk_dur=[{min(chunk_durations):.1f}s-{max(chunk_durations):.1f}s])"
        )

        return self._export_and_build_samples(sample, valid_chunks, output_directory)

    def _compute_adaptive_threshold(self, audio: AudioSegment) -> int:
        """Compute silence threshold relative to audio loudness."""
        mean_dbfs = audio.dBFS
        threshold = int(mean_dbfs - 16)
        return max(threshold, -60)

    def _build_samples_from_cache(
        self,
        original: AudioSample,
        cached_files: List[Path],
    ) -> List[AudioSample]:
        """Reconstruct AudioSample entities from previously cached segments."""
        return [
            AudioSample(
                subject_id=original.subject_id,
                file_path=chunk_file,
                is_parkinson=original.is_parkinson,
                task_type=original.task_type,
                segment_index=idx,
            )
            for idx, chunk_file in enumerate(cached_files)
        ]

    def _export_and_build_samples(
        self, original: AudioSample,
        chunks: List[AudioSegment], output_dir: Path,
    ) -> List[AudioSample]:
        """Export audio chunks to disk and return AudioSample entities."""
        results: List[AudioSample] = []
        for index, chunk in enumerate(chunks):
            chunk_path = output_dir / f"{original.file_path.stem}_seg_{index:03d}.wav"
            chunk.export(str(chunk_path), format="wav")
            results.append(AudioSample(
                subject_id=original.subject_id, file_path=chunk_path,
                is_parkinson=original.is_parkinson,
                task_type=original.task_type, segment_index=index,
            ))
        return results
