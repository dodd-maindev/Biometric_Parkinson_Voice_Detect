"""Command line interface runner for executing Parkinson detection experiments."""

import argparse
from pathlib import Path
from src.application.pipelines.baseline_replication_pipeline import BaselineReplicationPipeline
from src.application.pipelines.self_supervised_evaluation_pipeline import SelfSupervisedEvaluationPipeline
from src.application.pipelines.tri_modal_ensemble_evaluation_pipeline import TriModalEnsembleEvaluationPipeline
from src.application.services.multi_seed_aggregator_service import MultiSeedAggregatorService
from src.application.strategies.leave_one_subject_out_strategy import LeaveOneSubjectOutStrategy
from src.application.strategies.subject_split_validation_strategy import SubjectSplitValidationStrategy
from src.domain.entities.speech_task_type import SpeechTaskType
from src.infrastructure.persistence.mdvr_dataset_loader import MdvrDatasetLoader


def main() -> None:
    """Parse arguments and dispatch experimental evaluation across single or multiple seeds."""
    parser = argparse.ArgumentParser(description="Parkinson's Disease Voice Screening Benchmark")
    parser.add_argument("--data_dir", type=str, default="./data/raw/mdvr_kcl")
    parser.add_argument("--experiment", type=str, default="tri_modal", choices=["tri_modal", "baseline", "ssl_probe"])
    parser.add_argument("--task", type=str, default="READ_TEXT", choices=["READ_TEXT", "SPONTANEOUS_DIALOG"])
    parser.add_argument("--eval_strategy", type=str, default="split", choices=["split", "losocv"])
    parser.add_argument("--model_name", type=str, default="microsoft/wavlm-base-plus")
    parser.add_argument("--layer_index", type=int, default=6)
    parser.add_argument("--tri_weights", type=str, default="0.50,0.30,0.20")
    parser.add_argument("--decision_threshold", type=float, default=0.56)
    parser.add_argument("--baseline_checkpoint", type=str, default="/content/drive/MyDrive/parkinson_svm_baseline_94_87.joblib")
    parser.add_argument("--clear_cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for single run")
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated seeds e.g. '42,10,2024,7,99'")
    parser.add_argument("--num_seeds", type=int, default=0, help="Generate N distinct seeds including 42")
    parser.add_argument("--save_model_path", type=str, default="/content/drive/MyDrive/parkinson_tri_modal_97_44_best.joblib")
    args = parser.parse_args()

    samples = MdvrDatasetLoader(Path(args.data_dir)).load_samples(target_task_type=SpeechTaskType.from_string(args.task))
    if not samples: return print("No samples found! Please verify the dataset directory path.")

    if args.num_seeds > 0:
        seeds = [42] + [i for i in range(1, args.num_seeds + 1) if i != 42][:args.num_seeds - 1]
    elif args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [args.seed]
    results = []
    for s in seeds:
        print(f"\n>>>> EXECUTING EXPERIMENT [{args.experiment.upper()}] WITH SEED: {s} <<<<")
        strat = SubjectSplitValidationStrategy(random_seed=s) if args.eval_strategy == "split" else LeaveOneSubjectOutStrategy()
        m = _dispatch_experiment(args, samples, strat, s)
        results.append((s, m))
        if len(seeds) == 1:
            print(f"\n===== FINAL RESULTS ({args.eval_strategy.upper()}) =====\n{m.format_summary()}\n" + "=" * 55 + "\n")

    if len(seeds) > 1:
        print(MultiSeedAggregatorService.format_report(results))


def _dispatch_experiment(args, samples, strategy, seed: int = 42):
    """Route to the correct pipeline based on experiment type and random seed."""
    save_path = Path(args.save_model_path) if (args.save_model_path and seed == 42) else None
    if args.experiment == "tri_modal":
        w = tuple(float(x.strip()) for x in args.tri_weights.split(","))
        return TriModalEnsembleEvaluationPipeline(
            weights=w, baseline_checkpoint=args.baseline_checkpoint,
            threshold=args.decision_threshold, validation_strategy=strategy,
            output_model_path=save_path, random_seed=seed,
        ).run(samples)
    if args.experiment == "baseline":
        return BaselineReplicationPipeline(
            Path("./data/processed/segments"), strategy, args.clear_cache, save_path,
        ).run(samples, use_segmentation=True)
    layer = args.layer_index if args.layer_index != 0 else None
    return SelfSupervisedEvaluationPipeline(
        args.model_name, layer, validation_strategy=strategy, output_model_path=save_path,
    ).run(samples)


if __name__ == "__main__":
    main()
