from typing import Union, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from cuco import config_parser, Config

class PrunerEnum(Enum):
    HALVING = 'halving'
    MEDIAN = 'median'
    HYPERBAND = 'hyperband'
    NONE = 'none'

class SamplerEnum(Enum):
    RANDOM = 'random'
    TPE_SAMPLER = 'tpe_sampler'
    SKOPT_SAMPLER = 'skopt_sampler'


@config_parser(module_name='base.optimize')
@dataclass
class ExperimentHyperparameterOptimizationConfig(Config):
    verbose: int = 1
    pruner: PrunerEnum = PrunerEnum.HALVING
    sampler: SamplerEnum = SamplerEnum.TPE_SAMPLER

    seed: int = 42
    n_startup_trials: int = 0
    max_total_trials: Optional[int] = None
    n_processes: int = 4
    n_trials: int = 1

    stop_reward: Optional[float] = None

    optimize_timesteps: bool = False

    average_evaluations: Union[int, str] = 3

    no_optim_plots: bool = True
    storage_heartbeat_interval: int = 1200

    @staticmethod
    def load(
            verbose: int = 1,
            pruner: Union[PrunerEnum, str] = PrunerEnum.HALVING,
            sampler: Union[SamplerEnum, str] = SamplerEnum.TPE_SAMPLER,
            seed: int = 42,
            n_startup_trials: int = 0,
            max_total_trials: Optional[int] = None,
            n_processes: int = 1,
            n_trials: int = 1,
            stop_reward: Optional[float] = None,
            optimize_timesteps: bool = False,
            average_evaluations: Union[int, str] = 3,
            no_optim_plots: bool = True,
    ):
        return ExperimentHyperparameterOptimizationConfig(
            verbose=verbose,
            pruner=PrunerEnum(pruner),
            sampler=SamplerEnum(sampler),
            seed=seed,
            n_startup_trials=n_startup_trials,
            max_total_trials=max_total_trials,
            n_processes=n_processes,
            n_trials=n_trials,
            stop_reward=stop_reward,
            optimize_timesteps=optimize_timesteps,
            average_evaluations=average_evaluations,
            no_optim_plots=no_optim_plots,
        )
