"""
Hyperparameter tuning with Optuna.
"""
from typing import Any, Dict, Optional, Union
import copy
import torch
import optuna
from optuna import Trial

from ..config.base.ExperimentConfig import ExperimentConfig
from ..config.base.Agent import Agent
from ..config.base.LearningRateSchedule import LearningRateSchedule
from ..config.base.LearningRate import LearningRate
from ..config import Frequency, FrequencyUnit


DEFAULT_SEARCH_SPACE = {
    'gamma': [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999],

}

NET_ARCH_DICT = {
    "small": [256, 256],
    "medium": [256, 256, 256],
    "big": [512, 512, 512]
}

TRAIN_FREQUENCY_DICT = {
    '100step': (100, 'step'),
    '300step': (300, 'step'),
    '1000step': (1000, 'step'),
    '2000step': (2000, 'step'),
    '1episode': (1, 'episode'),
    #'2episode': (2, 'episode'),
}

def dqn_params_sampler(trial: Trial, search_space: Optional[Union[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Sampler for DQN hyperparameters.
    todo: load search space from json
    """

    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999])
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 0.1)
    learning_rate_scheduler = trial.suggest_categorical("schedule", [member.value for member in LearningRateSchedule if member != LearningRateSchedule.CONSTANT])
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    buffer_size = trial.suggest_categorical("buffer_size", [1*int(1e5), 3*int(1e5), 1*int(1e6)])
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0, 0.5)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.3, 0.9)
    target_update_interval = trial.suggest_categorical("target_update_interval", [1000, 2500, 5000, 10000, 15000, 20000])
    learning_starts = trial.suggest_categorical("learning_starts", [20000, 50000])
    train_freq = trial.suggest_categorical("train_freq", list(TRAIN_FREQUENCY_DICT.keys()))
    gradient_steps = trial.suggest_categorical('gradient_steps', [1, 2, 4, 8, 16])

    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])

    net_arch = NET_ARCH_DICT[net_arch]
    train_freq = TRAIN_FREQUENCY_DICT[train_freq]

    hyperparams = {
        "gamma": gamma,
        "learning_rate": LearningRate.load(initial_value=learning_rate, schedule=learning_rate_scheduler),
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": Frequency.load(*train_freq),
        "exploration_fraction": exploration_fraction,
        "exploration_final_eps": exploration_final_eps,
        "gradient-steps": gradient_steps,
        "target_update_interval": target_update_interval,
        "learning_starts": learning_starts,
        "policy_kwargs": net_arch,
    }

    return hyperparams


def update_experiment_config(
        config: ExperimentConfig,
        update_dict: Dict[str, Any],
        in_place: bool = True,
        transform: bool = False,
) -> ExperimentConfig:
    """update an ExperimentConfig with entries in the update_dict"""
    if transform:
        # the best params are not stored as the values given in the 'suggest' methods.
        # we need to transform the composite fields in the config to get usable results
        update_dict['learning_rate'] = LearningRate.load(
            initial_value=update_dict['learning_rate'],
            schedule=update_dict['schedule'],
        )
        update_dict['train_freq'] = TRAIN_FREQUENCY_DICT[update_dict['train_freq']]
        if isinstance(update_dict['train_freq'], int):
            update_dict['train_freq'] = Frequency(update_dict['train_freq'], FrequencyUnit.EPISODE)
        elif isinstance(update_dict['train_freq'], tuple):
            update_dict['train_freq'] = Frequency.load(*update_dict['train_freq'])
        else:
            raise ValueError(f'the training frequency:{update_dict["train_freq"]} can\'t be parsed to a valid training frequency.')
        update_dict['net_arch'] = NET_ARCH_DICT[update_dict['net_arch']]
    if not in_place:
        config = copy.deepcopy(config)
    for key, val in update_dict.items():
        if key == 'gamma':
            config.train.gamma = val
        elif key == 'learning_rate':
            config.train.learning_rate = val
        elif key == 'batch_size':
            config.train.batch_size = val
        elif key == 'train_freq':
            config.train.frequency = val
        elif key == 'exploration_fraction':
            config.train.exploration_fraction = val
        elif key == 'exploration_final_eps':
            config.train.exploration_final_eps = val
        elif key == 'gradient_steps':
            config.train.steps = val
        elif key == 'target_update_interval':
            config.train.target_update_interval = val
        elif key == 'learning_starts':
            config.train.learning_starts = val
        elif key == 'net_arch':
            config.policy_net_arch = tuple(val)
    return config

HYPERPARAMS_SAMPLER = {
    Agent.DQN: dqn_params_sampler,
}


if __name__ == "__main__":
    pass
