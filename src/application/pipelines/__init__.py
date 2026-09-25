"""Pipelines package orchestrating execution workflows."""

from src.application.pipelines.baseline_replication_pipeline import (
    BaselineReplicationPipeline,
)
from src.application.pipelines.self_supervised_evaluation_pipeline import (
    SelfSupervisedEvaluationPipeline,
)

__all__ = [
    "BaselineReplicationPipeline",
    "SelfSupervisedEvaluationPipeline",
]
