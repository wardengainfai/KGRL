from dataclasses import dataclass

from cuco import config_parser, Config


@config_parser(module_name = 'base.checkpoint')
@dataclass
class ExperimentCheckpointConfig(Config):
    frequency: int = -1
    path: str = None
