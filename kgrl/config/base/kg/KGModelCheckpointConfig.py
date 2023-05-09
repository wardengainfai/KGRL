import os
from dataclasses import dataclass

from ....__init__ import config

from cuco import config_parser, Config

from .KGModelConfig import KGModelConfig


@config_parser(object_type = 'kg-model-checkpoint-config', module_name = 'base.kg.checkpoint')
@dataclass
class KGModelCheckpointConfig(Config):
    model: KGModelConfig = None
    label: str = None

    experiment_name: str = None

    path_: str = None

    @property
    def path(self):
        if self.path_ is None:
            self.path_ = config.kg_model_checkpoint.format(experiment_name = self.experiment_name)
        return self.path_

    @property
    def name(self):
        return f"{self.model.architecture.value}-{'' if self.label is None else self.label}-embedding_dim={self.model.dim}{'-with-walls' if self.model.include_walls else '-without-walls'}.pkl"
