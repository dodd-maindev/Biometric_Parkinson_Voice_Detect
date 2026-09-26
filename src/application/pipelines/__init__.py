"""Pipelines package orchestrating execution workflows."""

from src.application.pipelines.baseline_replication_pipeline import (
    BaselineReplicationPipeline,
)
from src.application.pipelines.self_supervised_evaluation_pipeline import (
    SelfSupervisedEvaluationPipeline,
)
from src.application.pipelines.tri_modal_ensemble_evaluation_pipeline import (
    TriModalEnsembleEvaluationPipeline,
)

__all__ = [
    "BaselineReplicationPipeline",
    "SelfSupervisedEvaluationPipeline",
    "TriModalEnsembleEvaluationPipeline",
]
