from typing import ClassVar, Optional, List
from dataclasses import dataclass

from cuco import config_parser, Config

from .KGModelCheckpointConfig import KGModelCheckpointConfig
from .KGModelTrainConfig import KGModelTrainConfig
from .KGModelEvalConfig import KGModelEvalConfig
from .KGModelConfig import KGModelConfig
from .RevealGraph import RevealGraph


@config_parser(object_type='kg-wrapper-config', module_name='base.kg.wrapper')
@dataclass
class KGWrapperConfig(Config):
    """
    Config for the KG Wrapper
    """
    label: ClassVar[str] = None

    checkpoint: KGModelCheckpointConfig = KGModelCheckpointConfig()
    train: KGModelTrainConfig = KGModelTrainConfig()
    eval: KGModelEvalConfig = KGModelEvalConfig()
    model: KGModelConfig = KGModelConfig()

    reveal: Optional[RevealGraph] = RevealGraph.SUBGRAPH
    kg_path: Optional[str] = None
    observation_as_dict: Optional[bool] = True
    load_from_checkpoint: bool = False
    subgraph_depth: int = 2
    subgraph_directional: bool = True
    k_nn_k: int = 8
    rw_k: int = 8
    visualize_kg_obs: bool = False

    @staticmethod
    def load(
        checkpoint: KGModelCheckpointConfig = KGModelCheckpointConfig(),
        train: KGModelTrainConfig = None,
        eval: KGModelEvalConfig = KGModelEvalConfig(),
        model: KGModelConfig = KGModelConfig(),

        reveal: Optional[str] = RevealGraph.KNN_EMBEDDING.value,
        kg_path: Optional[str] = None,
        observation_as_dict: Optional[bool] = True,
        load_from_checkpoint: Optional[bool] = False,
        subgraph_depth: int = 2,
        subgraph_directional: bool = True,
        k_nn_k: int = 8,
        rw_k: int = 8,
        visualize_kg_obs: bool = False,
    ):
        if train is None:
            train = KGModelTrainConfig()

        checkpoint.model = model
        train.checkpoint = checkpoint

        if model.dim == 0:
            reveal = RevealGraph.RANDOM_WALK

        return KGWrapperConfig(
            checkpoint=checkpoint,
            train=train,
            eval=eval,
            model=model,
            reveal=RevealGraph(reveal),
            kg_path=kg_path,
            observation_as_dict=observation_as_dict,
            load_from_checkpoint=load_from_checkpoint,
            subgraph_depth=subgraph_depth,
            subgraph_directional=subgraph_directional,
            k_nn_k=k_nn_k,
            rw_k=rw_k,
            visualize_kg_obs=visualize_kg_obs
        )

    @property
    def compute_embeddings(self):
        return self.reveal in (RevealGraph.KNN_EMBEDDING, RevealGraph.SUBGRAPH_EMBEDDING, RevealGraph.RANDOM_WALK_EMBEDDING) or (self.load_from_checkpoint and self.checkpoint.path is not None)

    def get_observation_keys(self) -> List[str]:
        """Get the keys that are actually in the observation."""
        keys = [self.reveal.value, 'state']
        return keys
