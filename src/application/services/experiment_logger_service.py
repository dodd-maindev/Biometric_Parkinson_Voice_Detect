"""Service providing structured experiment logging for diagnostic purposes."""

from typing import Dict, List
import numpy as np


class ExperimentLoggerService:
    """Formats and prints structured diagnostic logs for ML experiments."""

    @staticmethod
    def log_section(title: str) -> None:
        """Print a prominent section header."""
        separator = "=" * 60
        print(f"\n{separator}")
        print(f"  {title}")
        print(separator)

    @staticmethod
    def log_subsection(title: str) -> None:
        """Print a subsection header."""
        print(f"\n--- {title} ---")

    @staticmethod
    def log_dataset_summary(
        total_samples: int,
        subjects: List[str],
        labels: List[int],
    ) -> None:
        """Log dataset composition including per-subject breakdown."""
        pd_count = sum(1 for label in labels if label == 1)
        hc_count = total_samples - pd_count
        unique_subjects = list(dict.fromkeys(subjects))

        ExperimentLoggerService.log_section("DATASET SUMMARY")
        print(f"  Total samples: {total_samples}")
        print(f"  Unique subjects: {len(unique_subjects)}")
        print(f"  PD (Parkinson): {pd_count}  |  HC (Healthy): {hc_count}")
        print(f"  Class ratio (PD/HC): {pd_count/max(hc_count,1):.2f}")

    @staticmethod
    def log_per_subject_distribution(
        subjects: List[str],
        labels: List[int],
    ) -> None:
        """Log per-subject sample counts and diagnosis label."""
        ExperimentLoggerService.log_subsection("Per-Subject Distribution")
        subject_info: Dict[str, Dict] = {}
        for subj, label in zip(subjects, labels):
            if subj not in subject_info:
                subject_info[subj] = {"count": 0, "label": label}
            subject_info[subj]["count"] += 1

        for subj, info in sorted(subject_info.items()):
            diagnosis = "PD" if info["label"] == 1 else "HC"
            print(f"  {subj}: {info['count']:4d} segments  [{diagnosis}]")
