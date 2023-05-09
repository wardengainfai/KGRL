import os

from typing import Union, Tuple, Optional, Mapping, Any
from dataclasses import dataclass

# from ray.rllib.agents import dqn
# from ray.rllib.agents.trainer import COMMON_CONFIG

from cuco import config_parser, Config

from ...__init__ import config

from .ExperimentLoggingConfig import ExperimentLoggingConfig
from .ExperimentCheckpointConfig import ExperimentCheckpointConfig
from .ExperimentTrainConfig import ExperimentTrainConfig
from .ExperimentEvalConfig import ExperimentEvalConfig
from .ExperimentHyperparameterOptimizationConfig import ExperimentHyperparameterOptimizationConfig

from .Agent import Agent
from .Library import Library

from .kg import KGWrapperConfig


def get_default_trainer_config(agent: Union[Agent, str] = "dqn", library: Union[Library, str] = "tune"):
    """get the agent default config."""
    # if library in {"rllib", "tune"} or isinstance(library, Library):
    #     if agent == Agent.DQN:
    #         config_ = dqn.DEFAULT_CONFIG.copy()
    #     else:
    #         config_ = COMMON_CONFIG.copy()
    #     return config_
    # else:
    #     raise NotImplementedError("This has not been implemented for{}".format(library))
    pass


@config_parser(module_name='base.experiment')
@dataclass
class ExperimentConfig(Config):
    logging: ExperimentLoggingConfig = ExperimentLoggingConfig()
    checkpoint: ExperimentCheckpointConfig = ExperimentCheckpointConfig()
    train: ExperimentTrainConfig = ExperimentTrainConfig()
    eval: ExperimentEvalConfig = ExperimentEvalConfig()
    kg_wrapper_config: Optional[Union[KGWrapperConfig, Mapping[str, Any]]] = KGWrapperConfig()
    hyperparameter_optimization: Optional[ExperimentHyperparameterOptimizationConfig] = ExperimentHyperparameterOptimizationConfig()

    experiment_name: str = "rlkg"
    library: Library = Library.STABLE_BASELINES3
    policy_net_arch: Tuple[int] = (256, 256)
    agent: Agent = Agent.DQN
    kg_wrapper_enabled: bool = False
    experiment_headless: bool = False
    experiment_env_name: str = None
    deterministic: bool = False

    max_env_steps: Optional[int] = None

    @staticmethod
    def load(
        logging: ExperimentLoggingConfig = None,
        checkpoint: ExperimentCheckpointConfig = None,
        train: ExperimentTrainConfig = None,
        eval: ExperimentEvalConfig = None,
        kg_wrapper_config: Optional[Union[KGWrapperConfig, Mapping[str, Any]]] = None,
        hyperparameter_optimization: Optional[ExperimentHyperparameterOptimizationConfig] = None,

        experiment_name: str = "rlkg",
        library: str = Library.STABLE_BASELINES3.value,
        policy_net_arch: Tuple[int] = (256, 256),
        agent: str = Agent.DQN.value,
        kg_wrapper_enabled: bool = False,
        experiment_headless: bool = False,
        experiment_env_name: str = None,
        deterministic: bool = False,

        max_env_steps: Optional[int] = None,
    ):
        if logging is None:
            logging = ExperimentLoggingConfig()

        if train is None:
            train = ExperimentTrainConfig()

        if eval is None:
            eval = ExperimentEvalConfig()

        if checkpoint is None:
            checkpoint = ExperimentCheckpointConfig()

        if kg_wrapper_config is None:
            kg_wrapper_config = KGWrapperConfig()

        if hyperparameter_optimization is None:
            hyperparameter_optimization = ExperimentHyperparameterOptimizationConfig()

        kg_wrapper_config.checkpoint.experiment_name = experiment_name

        library = Library(library)
        agent = Agent(agent)

        # if train.trainer_config is None:
        #     train.trainer_config = get_default_trainer_config(agent, library)

        if checkpoint.path is None:
            checkpoint.path = os.path.join(config.checkpoint, experiment_name)

        # if kg_wrapper_enabled:
        #     if isinstance(kg_wrapper_config, dict):
        #         kg_wrapper_config = KGWrapperConfig(**kg_wrapper_config)
        #     elif kg_wrapper_config is None:
        #         kg_wrapper_config = KGWrapperConfig()

        if kg_wrapper_config is not None:
            kg_wrapper_config.embedding_checkpoint_dir = os.path.join(checkpoint.path, "graph_embedding")

        if logging.path is None:
            logging.path = os.path.join(checkpoint.path, "logs")

        if eval.run_during_training:
            # train.trainer_config["evaluation_interval"] = checkpoint.frequency
            train.evaluation_interval = checkpoint.frequency

        return ExperimentConfig(
            logging=logging,
            checkpoint=checkpoint,
            train=train,
            eval=eval,
            kg_wrapper_config=kg_wrapper_config,
            hyperparameter_optimization=hyperparameter_optimization,
            experiment_name=experiment_name,
            library=library,
            policy_net_arch=tuple(policy_net_arch),
            agent=agent,
            kg_wrapper_enabled=kg_wrapper_enabled,
            experiment_headless=experiment_headless,
            experiment_env_name=experiment_env_name,
            deterministic=deterministic,
            max_env_steps=max_env_steps,
        )
    