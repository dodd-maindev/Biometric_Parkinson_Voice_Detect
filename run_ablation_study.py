"""CLI runner for systematically evaluating model contributions in Parkinson screening."""

import argparse
from pathlib import Path
from src.application.pipelines.ablation_study_pipeline import AblationStudyPipeline
from src.application.strategies.subject_split_validation_strategy import (
    SubjectSplitValidationStrategy,
)
from src.domain.entities.speech_task_type import SpeechTaskType
from src.infrastructure.persistence.archive_voice_dataset_loader import (
    ArchiveVoiceDatasetLoader,
)
from src.infrastructure.persistence.mdvr_dataset_loader import MdvrDatasetLoader


def main() -> None:
    """Parse arguments and execute systematic ablation study."""
    parser = argparse.ArgumentParser(description="Ablation Study for Tri-Modal Parkinson Detection")
    parser.add_argument("--dataset", type=str, default="mdvr", choices=["mdvr", "voice_dataset"])
    parser.add_argument("--data_dir", type=str, default="./data/raw/mdvr_kcl")
    parser.add_argument("--task", type=str, default="READ_TEXT", choices=["READ_TEXT", "SPONTANEOUS_DIALOG", "SUSTAINED_VOWEL"])
    parser.add_argument("--baseline_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    loader = ArchiveVoiceDatasetLoader(data_path) if args.dataset == "voice_dataset" else MdvrDatasetLoader(data_path)
    task_type = SpeechTaskType.from_string(args.task) if args.dataset == "mdvr" else None
    samples = loader.load_samples(target_task_type=task_type)
    print(f"Loaded {len(samples)} samples for dataset {args.dataset} from {data_path}")
    if not samples:
        return print("No samples found! Please verify the dataset directory path.")

    cache_dir = Path(f"./data/processed/{args.dataset}/segments")
    strategy = SubjectSplitValidationStrategy(random_seed=args.seed)
    pipeline = AblationStudyPipeline(
        cache_directory=cache_dir,
        baseline_checkpoint=args.baseline_checkpoint,
        validation_strategy=strategy,
    )
    pipeline.run(samples)


if __name__ == "__main__":
    main()
