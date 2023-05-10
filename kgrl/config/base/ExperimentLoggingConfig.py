from dataclasses import dataclass

from cuco import config_parser, Config


@config_parser(module_name = 'base.logging')
@dataclass
class ExperimentLoggingConfig(Config):
    log_interval: int = 2
    tensorboard: bool = False
    wandb: bool = False
    path: str = None
