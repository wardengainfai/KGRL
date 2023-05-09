import os
from dataclasses import dataclass

from cuco import config_parser, Config

from ..SplitConfig import SplitConfig
from .KGModelCheckpointConfig import KGModelCheckpointConfig

from ....utils.config import replace_kwargs


@config_parser(object_type = 'kg-model-train-config', module_name = 'base.kg.train')
@dataclass
class KGModelTrainConfig(Config):
    n_epochs: int = 800
    split: SplitConfig = SplitConfig(train = 0.8, test = 0.1, valid = 0.1)
    kwargs: dict = None
    use_tqdm: bool = True

    checkpoint: KGModelCheckpointConfig = None

    @property
    def checkpoint_path(self):
        return os.path.join(self.checkpoint.path, self.checkpoint.name)

    @property
    def as_dict(self):
        return replace_kwargs(self.kwargs, num_epochs = self.n_epochs, use_tqdm = self.use_tqdm, checkpoint_name = self.checkpoint_path)
