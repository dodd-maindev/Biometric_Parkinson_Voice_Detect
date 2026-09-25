"""Command line interface runner for executing Parkinson detection experiments."""

import argparse
from pathlib import Path
from src.application.pipelines.baseline_replication_pipeline import (
    BaselineReplicationPipeline,
)
from src.application.pipelines.self_supervised_evaluation_pipeline import (
    SelfSupervisedEvaluationPipeline,
)
from src.domain.entities.speech_task_type import SpeechTaskType
from src.infrastructure.persistence.mdvr_dataset_loader import (
    MdvrDatasetLoader,
)


def main() -> None:
    """Parse arguments and dispatch experimental evaluation."""
    parser = argparse.ArgumentParser(description="Parkinson's Disease Voice Screening Benchmark")
    parser.add_argument("--data_dir", type=str, default="./data/raw/mdvr_kcl", help="Path to MDVR-KCL dataset")
    parser.add_argument("--experiment", type=str, default="baseline", choices=["baseline", "ssl_frozen", "ssl_probe"])
    parser.add_argument("--task", type=str, default="READ_TEXT", choices=["READ_TEXT", "SPONTANEOUS_DIALOG"])
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base", help="HuggingFace model ID")
    parser.add_argument("--layer_index", type=int, default=6, help="Layer index for probing (0-12)")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    loader = MdvrDatasetLoader(data_path)
    task_type = SpeechTaskType.from_string(args.task)
    samples = loader.load_samples(target_task_type=task_type)

    print(f"Loaded {len(samples)} samples for task {task_type.name} from {data_path}")
    if not samples:
        print("No samples found! Please verify the dataset directory path.")
        return

    if args.experiment == "baseline":
        print("\n--- Running EXP-0: Baseline Replication (Acoustic + GTCC + MFCC) ---")
        pipeline = BaselineReplicationPipeline(segmentation_cache_directory=Path("./data/processed/segments"))
        metrics = pipeline.run(samples, use_segmentation=True)
    elif args.experiment == "ssl_frozen":
        print(f"\n--- Running EXP-1: Frozen SSL Evaluation ({args.model_name}) ---")
        pipeline = SelfSupervisedEvaluationPipeline(model_name=args.model_name)
        metrics = pipeline.run(samples)
    else:
        print(f"\n--- Running EXP-2: Layer-wise Probing (Layer {args.layer_index} of {args.model_name}) ---")
        pipeline = SelfSupervisedEvaluationPipeline(model_name=args.model_name, layer_index=args.layer_index)
        metrics = pipeline.run(samples)

    print("\n================ FINAL EVALUATION RESULTS (LOSOCV) ================")
    print(metrics.format_summary())
    print("===================================================================\n")


if __name__ == "__main__":
    main()
