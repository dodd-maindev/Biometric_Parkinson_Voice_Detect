"""Audio segmentation replicating Hossain et al. (~816 chunks from 73 files)."""

from pathlib import Path
from typing import List
from pydub import AudioSegment
from pydub.silence import split_on_silence
from src.domain.entities.audio_sample import AudioSample
from src.domain.interfaces.audio_segmenter import IAudioSegmenter


class SilenceAudioSegmenter(IAudioSegmenter):
    """Segments speech recordings into vocal intervals via silence detection."""

    _MINIMUM_SILENCE_MS: int = 500
    _SILENCE_OFFSET_DB: int = 10
    _KEEP_SILENCE_PADDING_MS: int = 150
    _MIN_CHUNK_DURATION_MS: int = 3000

    def segment(
        self, sample: AudioSample, output_directory: Path,
    ) -> List[AudioSample]:
        """Split audio on natural pauses into multi-second vocal chunks."""
        output_directory.mkdir(parents=True, exist_ok=True)
        cached = sorted(output_directory.glob(f"{sample.file_path.stem}_seg_*.wav"))

        if cached:
            return self._build_from_cache(sample, cached)

        raw_audio = AudioSegment.from_file(str(sample.file_path))
        normalized = raw_audio.normalize()
        threshold = max(int(normalized.dBFS) - self._SILENCE_OFFSET_DB, -50)

        chunks = split_on_silence(
            normalized,
            min_silence_len=self._MINIMUM_SILENCE_MS,
            silence_thresh=threshold,
            keep_silence=self._KEEP_SILENCE_PADDING_MS,
            seek_step=10,
        )

        valid = [c for c in chunks if len(c) >= self._MIN_CHUNK_DURATION_MS]
        if not valid:
            valid = [normalized]

        durations = [len(c) / 1000.0 for c in valid]
        print(
            f"  {sample.file_path.name}: {len(raw_audio)/1000:.1f}s -> "
            f"{len(valid)} chunks (dBFS={normalized.dBFS:.1f}, "
            f"thresh={threshold}, dur=[{min(durations):.1f}s-{max(durations):.1f}s])"
        )

        return self._export_chunks(sample, valid, output_directory)

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
