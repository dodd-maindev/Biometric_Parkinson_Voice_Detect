"""Service for aggregating and formatting metrics across multiple experimental seeds."""

from typing import Dict, List, Tuple
import numpy as np
from src.domain.entities.evaluation_metrics import EvaluationMetrics


class MultiSeedAggregatorService:
    """Computes mean and standard deviation across seeds for publication benchmark."""

    @staticmethod
    def aggregate(
        seed_results: List[Tuple[int, EvaluationMetrics]],
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate mean and standard deviation for each diagnostic metric.

        Args:
            seed_results: List of tuples containing seed and its EvaluationMetrics.

        Returns:
            Dictionary mapping metric name to (mean, std).
        """
        metrics = {
            "accuracy": [m.accuracy for _, m in seed_results],
            "sensitivity": [m.sensitivity for _, m in seed_results],
            "specificity": [m.specificity for _, m in seed_results],
            "f1_score": [m.f1_score for _, m in seed_results],
            "roc_auc": [m.roc_auc for _, m in seed_results],
            "mcc": [m.matthews_correlation_coefficient for _, m in seed_results],
        }
        return {k: (float(np.mean(v)), float(np.std(v, ddof=1) if len(v) > 1 else 0.0)) for k, v in metrics.items()}

    @classmethod
    def format_report(cls, seed_results: List[Tuple[int, EvaluationMetrics]]) -> str:
        """Format a publication-grade markdown table of per-seed and aggregated statistics.

        Args:
            seed_results: List of tuples containing seed and its EvaluationMetrics.

        Returns:
            Formatted multi-line report string.
        """
        agg = cls.aggregate(seed_results)
        lines = [
            "\n" + "=" * 80,
            f"  MULTI-SEED EXPERIMENTAL EVALUATION REPORT (N = {len(seed_results)} SEEDS)",
            "=" * 80,
            f"{'Seed':<8}{'Accuracy':<13}{'Sensitivity':<14}{'Specificity':<14}{'F1-Score':<11}{'ROC-AUC':<10}{'MCC':<8}",
            "-" * 80,
        ]
        for seed, m in seed_results:
            lines.append(
                f"{seed:<8}{m.accuracy*100:>6.2f}%      {m.sensitivity*100:>6.2f}%       {m.specificity*100:>6.2f}%       "
                f"{m.f1_score*100:>6.2f}%   {m.roc_auc:>7.4f}  {m.matthews_correlation_coefficient:>7.4f}"
            )
        lines.append("-" * 80)
        lines.append(
            f"{'MEAN±STD':<8}"
            f"{agg['accuracy'][0]*100:>5.2f}±{agg['accuracy'][1]*100:<4.2f}% "
            f"{agg['sensitivity'][0]*100:>5.2f}±{agg['sensitivity'][1]*100:<4.2f}%  "
            f"{agg['specificity'][0]*100:>5.2f}±{agg['specificity'][1]*100:<4.2f}%  "
            f"{agg['f1_score'][0]*100:>5.2f}±{agg['f1_score'][1]*100:<4.2f}% "
            f"{agg['roc_auc'][0]:>6.4f}±{agg['roc_auc'][1]:<5.3f} "
            f"{agg['mcc'][0]:>5.4f}±{agg['mcc'][1]:<5.3f}"
        )
        lines.append("=" * 80)
        lines.append("Paper Target: Acc: 95.45% | Sens: 94.62% | Spec: 95.97% | ROC-AUC: 0.98 | MCC: 0.90")
        lines.append("=" * 80 + "\n")
        return "\n".join(lines)
