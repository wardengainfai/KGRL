from dataclasses import dataclass

from cuco import config_parser, Config

from ....utils.config import replace_kwargs


@config_parser(object_type = 'kg-model-eval-config', module_name = 'base.kg.eval')
@dataclass
class KGModelEvalConfig(Config):
    kwargs: dict = None
    use_tqdm: bool = True

    @property
    def as_dict(self):
        return replace_kwargs(self.kwargs, use_tqdm = self.use_tqdm)
