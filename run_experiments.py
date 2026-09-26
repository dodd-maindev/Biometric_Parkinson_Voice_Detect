"""Command line interface runner for executing Parkinson detection experiments."""

import argparse
from pathlib import Path
from src.application.pipelines.baseline_replication_pipeline import BaselineReplicationPipeline
from src.application.pipelines.ensemble_evaluation_pipeline import EnsembleEvaluationPipeline
from src.application.pipelines.hybrid_fusion_evaluation_pipeline import HybridFusionEvaluationPipeline
from src.application.pipelines.self_supervised_evaluation_pipeline import SelfSupervisedEvaluationPipeline
from src.application.strategies.leave_one_subject_out_strategy import LeaveOneSubjectOutStrategy
from src.application.strategies.subject_split_validation_strategy import SubjectSplitValidationStrategy
from src.domain.entities.speech_task_type import SpeechTaskType
from src.infrastructure.persistence.mdvr_dataset_loader import MdvrDatasetLoader


def main() -> None:
    """Parse arguments and dispatch experimental evaluation."""
    parser = argparse.ArgumentParser(description="Parkinson's Disease Voice Screening Benchmark")
    parser.add_argument("--data_dir", type=str, default="./data/raw/mdvr_kcl")
    parser.add_argument(
        "--experiment", type=str, default="baseline",
        choices=["baseline", "ssl_frozen", "ssl_probe", "hybrid_fusion", "ensemble"],
    )
    parser.add_argument("--task", type=str, default="READ_TEXT", choices=["READ_TEXT", "SPONTANEOUS_DIALOG"])
    parser.add_argument("--eval_strategy", type=str, default="split", choices=["split", "losocv"])
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--layer_index", type=int, default=6)
    parser.add_argument("--pca_components", type=int, default=32)
    parser.add_argument("--ensemble_weight", type=float, default=0.60)
    parser.add_argument(
        "--baseline_checkpoint", type=str,
        default="/content/drive/MyDrive/parkinson_svm_baseline_94_87.joblib",
    )
    parser.add_argument("--clear_cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_model_path", type=str, default="./checkpoints/parkinson_model.joblib")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    loader = MdvrDatasetLoader(data_path)
    task_type = SpeechTaskType.from_string(args.task)
    samples = loader.load_samples(target_task_type=task_type)

    print(f"Loaded {len(samples)} samples for task {task_type.name} from {data_path}")
    if not samples:
        return print("No samples found! Please verify the dataset directory path.")

    strategy = _build_strategy(args.eval_strategy, args.seed)
    metrics = _dispatch_experiment(args, samples, strategy)

    print(f"\n===== FINAL RESULTS ({args.eval_strategy.upper()}) =====")
    print(metrics.format_summary())
    print("=" * 55 + "\n")


def _build_strategy(eval_strategy: str, seed: int = 42):
    """Construct the appropriate validation strategy from CLI argument."""
    if eval_strategy == "split":
        return SubjectSplitValidationStrategy(random_seed=seed)
    return LeaveOneSubjectOutStrategy()


def _dispatch_experiment(args, samples, strategy):
    """Route to the correct pipeline based on experiment type."""
    save_path = Path(args.save_model_path) if args.save_model_path else None
    if args.experiment == "baseline":
        print(f"\n--- EXP-0: Baseline (Acoustic + GTCC, {args.eval_strategy.upper()}) ---")
        pipeline = BaselineReplicationPipeline(
            Path("./data/processed/segments"), strategy, args.clear_cache, save_path,
        )
        return pipeline.run(samples, use_segmentation=True)

    if args.experiment == "hybrid_fusion":
        print(f"\n--- EXP-2: Hybrid Fusion ({args.model_name}, PCA={args.pca_components}) ---")
        layer = args.layer_index if args.layer_index != 0 else None
        pipeline = HybridFusionEvaluationPipeline(
            args.model_name, layer, args.pca_components,
            validation_strategy=strategy, output_model_path=save_path,
        )
        return pipeline.run(samples)

    if args.experiment == "ensemble":
        print(f"\n--- EXP-3: Soft Voting Ensemble (Baseline + {args.model_name}) ---")
        layer = args.layer_index if args.layer_index != 0 else None
        pipeline = EnsembleEvaluationPipeline(
            args.model_name, layer, args.ensemble_weight,
            baseline_checkpoint=args.baseline_checkpoint,
            validation_strategy=strategy, output_model_path=save_path,
        )
        return pipeline.run(samples)

    print(f"\n--- EXP-1: SSL ({args.model_name}, {args.eval_strategy.upper()}) ---")
    layer = args.layer_index if args.experiment == "ssl_probe" else None
    pipeline = SelfSupervisedEvaluationPipeline(
        args.model_name, layer, validation_strategy=strategy, output_model_path=save_path,
    )
    return pipeline.run(samples)


if __name__ == "__main__":
    main()
