from dataclasses import dataclass
from typing import Dict, Union

from cuco import config_parser, Config

from .KGEmbeddingModel import KGEmbeddingModel

from ....utils.config import replace_kwargs


@config_parser(object_type='kg-model-config', module_name='base.kg.model')
@dataclass
class KGModelConfig(Config):
    kwargs: dict = None
    dim: int = 8
    include_walls: bool = False
    architecture: KGEmbeddingModel = KGEmbeddingModel.TRANSE

    @property
    def as_dict(self):
        return replace_kwargs(self.kwargs, embedding_dim=self.dim)

    @staticmethod
    def load(
            dim: int = 8,
            include_walls: bool = False,
            architecture: Union[str, KGEmbeddingModel] = KGEmbeddingModel.TRANSE,
            kwargs: Dict = None,
    ):
        return KGModelConfig(
            dim=dim,
            include_walls=include_walls,
            architecture=KGEmbeddingModel(architecture),
            kwargs=kwargs,
        )

