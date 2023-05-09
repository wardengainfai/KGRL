from typing import List
from dataclasses import dataclass

from .embedding import KGEmbeddingModel


@dataclass
class ExperimentResults:
    config: str  # .json of the training config
    log: str  # path to the logging folder
    kg_embedding_model: KGEmbeddingModel  #
    checkpoints: List[str]  # list of paths to the saved policies at certain intervals, depending on the rl Library used
    model: str  # path to the final saved policy, depending on the rl Library used


@dataclass
class TrainingResult:
    # todo: These types are not final! change once we have a clear demand for the type
    model: str
    config: str
