import os
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Any, List
import warnings
import pickle
import logging
import copy
import json
import tqdm
import time

from joblib import Parallel, delayed

import torch
import gym.wrappers
import numpy as np

import stable_baselines3
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, CheckpointCallback
from stable_baselines3.common.logger import ERROR
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import HerReplayBuffer

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.trial import TrialState
from optuna.study import MaxTrialsCallback
from optuna.samplers import BaseSampler, RandomSampler, TPESampler
from optuna.pruners import BasePruner, MedianPruner, NopPruner, SuccessiveHalvingPruner, HyperbandPruner
from optuna.exceptions import ExperimentalWarning
warnings.filterwarnings("ignore", category=ExperimentalWarning, module="optuna")

from ...config import ExperimentConfig, Library, Agent, ExperimentTrainConfig, PrunerEnum, SamplerEnum
from ...logging import CustomLogger, configure
from ...dto.experiments import ExperimentResults

from .eval_callback import EpisodeEvalCallback, TrialEvalCallback, StopTrainingOnRewardThresholdAndSteps
from .EvaluationResults import EvaluationResults

from ...utils.hyperparametertuing_utils import HYPERPARAMS_SAMPLER, update_experiment_config


class Experimenter(ABC):
    def __init__(
            self,
            config: Optional[ExperimentConfig] = None,
            device: Optional[Union[torch.device, str]] = None,
    ):
        if not isinstance(config, ExperimentConfig):
            if config is not None:
                logging.warning("Provided config has to be of `ExperimentConfig` class.")
            logging.info("Experiment Config set to default values.")
            self.config = ExperimentConfig()
        else:
            self.config = config

        if not os.path.isdir(self.config.checkpoint.path):
            os.makedirs(self.config.checkpoint.path)
        if self.config.kg_wrapper_enabled:
            if (kg_wrapper_config := self.config.kg_wrapper_config) is not None and not os.path.isdir(kg_checkpoint_path := kg_wrapper_config.train.checkpoint.path):
                os.makedirs(kg_checkpoint_path)

        self.agent = None  # is set in reset_experiment
        self.env_version = None
        self.device = device if device is not None else "auto"

        self._register_env()
        self.reset_experiment()

        self.results_dict: Dict[int, Dict[str, Any]] = {}
        self.eval_dict = {}
        self.checkpoints = []
        self.study_name = None

    @property
    def experiment_id(self):
        return self.config.experiment_env_name

    @abstractmethod
    def _create_env(self, config: ExperimentConfig = None) -> Union[gym.Env]:
        pass

    @abstractmethod
    def run_trial(
            self,
            n_training_runs: int = 20,
            envs: Optional[Union[str, List[str]]] = None,
            log_name_suffix: str = "",
            **kwargs
    ) -> ExperimentResults:
        """Run a trial of serveral experiments with similar configurations."""

    @abstractmethod
    def evaluate_trial(self, path: str = None) -> None:
        """Evaluate the trial"""

    @abstractmethod
    def _create_env_force_headless(self, config: Optional[Any] = None):
        """Create environment enforcing headless option regardless of what is prescribed by config"""

    def _register_env(self, reregister: bool = False) -> None:
        """
                register the environment for rllib and gym.
                TODO: there is a bug where after using gym.make with the just registered
                  env once the same fails afterwards ?? moved everything to using the
                  _create_env()
                """
        if reregister:
            env_dict = gym.envs.registry.env_specs.copy()
            for env in env_dict:
                if self.config.experiment_env_name in env:
                    print("Removing gym registry entry {}".format(env))
                    del gym.envs.registry.env_specs[env]
        # if self.config.library.name in {"tune", "rllib"}:
        #     tune.register_env(
        #         name=self.config.experiment_env_name,
        #         env_creator=self._create_env_force_headless,  # Rendering for training will always be disabled
        #     )

        gym.envs.register(
            id=self.config.experiment_env_name,
            entry_point=self._create_env_force_headless,
            # entry_point=self._create_env_force_headless,
            # version=self.env_version,
        )

    def reset_experiment(self, checkpoint: str = None) -> None:
        self.results_dict = dict()
        self.eval_dict = dict()
        self.checkpoints = []
        if checkpoint is not None:
            self.agent = self._create_agent(verbose=0)
            self._restore_agent_from_checkpoint(checkpoint)
        else:
            self.agent = self._create_agent(verbose=0)

    def save_eval_dict(self, save_dir: str = None,
                       file_ending: str = "pkl") -> str:
        """save the evaluation dictionary of the experiment to save_dir/eval_results.`format`"""
        if save_dir is None:
            save_dir = self.config.checkpoint.path
        filename = os.path.join(save_dir, "eval_results.{}".format(file_ending))
        if file_ending in {"pkl", "pickle", "pl"}:
            with open(filename, "wb") as file:
                pickle.dump(self.eval_dict, file)
        elif file_ending in {"json"}:
            with open(filename, "w") as file:
                json.dump(self.eval_dict, file)
        else:
            raise ValueError("Specify format as pickle or json.")
        return filename

    def _restore_agent_from_checkpoint(self, checkpoint: str):
        if self.config.library == Library.TUNE:
            pass
        elif self.config.library == Library.RLLIB:
            self.agent.restore(checkpoint)
        else:
            raise NotImplementedError(
                "restoring checkpoint is not implemented for Library {}".format(
                    self.config.library))

    def _create_agent(
            self,
            config=None,  #: TrainerConfigDict
            library: Union[Library, str] = None,
            verbose: int = 1,
    ) -> Union[stable_baselines3.common.base_class.BaseAlgorithm]:
        """Provide the appropriate agent (depending on the library) to the class."""
        library = Library(
            library) if library is not None else self.config.library
        if config is None:
            config = self.config
        if library == Library.STABLE_BASELINES3:
            if config.agent == Agent.DQN:
                return stable_baselines3.DQN(
                    policy="MultiInputPolicy" if config.kg_wrapper_enabled and config.kg_wrapper_config.observation_as_dict or (
                                config.train.observation_type == 'dict') else "MlpPolicy",
                    env=self._create_env(force_headless=True),
                    # Rendering will be disabled during training  # self.config.experiment_env_name
                    verbose=verbose,
                    train_freq=config.train.frequency.as_tuple,
                    learning_rate=config.train.learning_rate.as_callable,
                    exploration_fraction=config.train.exploration_fraction,
                    exploration_final_eps=config.train.exploration_final_eps,
                    batch_size=config.train.batch_size,
                    buffer_size=config.train.buffer_size,
                    # todo: implement HER replay buffer for general environments. so far this des not work for any env not adhering to the gym GoalEnv schema.
                    # use the Hindsight Experience Replay buffer(sampling for subgoals
                    # replay_buffer_class=HerReplayBuffer,
                    # replay_buffer_kwargs=dict(
                    #     goal_selection_strategy='future',
                    #     n_sampled_goal=4,
                    #     max_episode_length=self.config.max_env_steps,
                    # ),
                    tensorboard_log=config.logging.path,
                    policy_kwargs=dict(net_arch=config.policy_net_arch),
                    gamma=config.train.gamma,
                    gradient_steps=config.train.steps,
                    target_update_interval=config.train.target_update_interval,
                    device=self.device,
                )

    def run_training(
            self,
            library: str = None,
            tb_log_name: str = None,
            verbose: int = 1,
            results: EvaluationResults = None,
            agent: Optional[stable_baselines3.common.base_class.BaseAlgorithm] = None,
            config: Optional[ExperimentConfig] = None,
            callbacks: Optional[List[BaseCallback]] = None,
            callbacks_append: bool = True,
    ):  # todo fix return type

        if config is None:
            config = self.config

        if agent is None:
            agent = self._create_agent(config=config, verbose=max(0, verbose-1))

        if callbacks is None:
            callbacks = []

        if library is None:
            library = config.library
        if tb_log_name is None:
            tb_log_name = config.experiment_name
        logging.info("Start Training with {}.".format(library.name))
        if library == Library.RLLIB:
            for i in tqdm.tqdm(range(1, config.train.steps + 1),
                               desc="Training"):
                result = agent.train()
                self.results_dict[i] = {"step": i, "result": result}
                if "evaluation" in result:
                    self.eval_dict[i] = result["evaluation"]
                if i % self.config.checkpoint.frequency == 0:
                    logging.info("Finished Training step {}:".format(i))
                    checkpoint = agent.save(checkpoint_dir=config.checkpoint.path)
                    self.results_dict[i]["checkpoint"] = checkpoint
                    self.save_eval_dict()
            return self.results_dict
        elif library == Library.STABLE_BASELINES3:

            # init some callbacks for evaluation and stoppage
            if callbacks_append:  # append callbacks to the callback list
                eval_callback = EpisodeEvalCallback(
                    eval_env=(eval_env := self._create_env(config=config)),  # todo: add the possibility to also optimize parameters of the wrappers
                    best_model_save_path=config.checkpoint.path,
                    log_path=os.path.join(config.checkpoint.path, "logs"),
                    eval_freq=config.eval.frequency.as_tuple,
                    eval_freq_max=config.eval.frequency_max.as_tuple if config.eval.frequency_max is not None else None,
                    eval_start=config.eval.start,
                    deterministic=config.deterministic,
                    render=not config.experiment_headless,
                    n_eval_episodes=config.eval.n_episodes,
                    verbose=verbose,
                    results=results,
                    max_episode_length=config.eval.max_episode_length,
                    rendering_step_delay=config.eval.step_delay
                )
                callbacks.append(eval_callback)

                # if not results.plotted_losses:
                if config.kg_wrapper_enabled and results is not None and self.config.kg_wrapper_config.compute_embeddings:
                    results.set_losses(
                        eval_env.graph_handler.embedding_results.pipeline_results.losses)

                if config.checkpoint.frequency > -1:
                    checkpoint_callback = CheckpointCallback(
                        save_freq=config.checkpoint.frequency,
                        save_path=config.checkpoint.path,
                        name_prefix=config.experiment_name + "ckp",
                    )
                    callbacks.append(checkpoint_callback)

            # enable stopping after a certain amount of episodes
            if config.train.max_episodes > -1:
                stop_callback = StopTrainingOnMaxEpisodes(
                    max_episodes=config.train.max_episodes,
                    verbose=1)
                callbacks.append(stop_callback)

            # change logging
            if verbose > 0:
                new_logger = configure(
                    os.path.join(config.logging.path, tb_log_name),
                    ["stdout", "csv", "tensorboard"],
                    level=ERROR,
                    init_logger=CustomLogger
                )
                agent.set_logger(new_logger)

            return agent.learn(
                total_timesteps=config.train.total_timesteps,
                log_interval=config.logging.log_interval if verbose > 0 else None,  # contrary to the documentation this is the number of episodes before a log!
                tb_log_name=tb_log_name,
                callback=callbacks,  # if callbacks contains an eval_callback setting eval_freq>0 will result in redundant evaluation!
                # eval_freq=20000,
                # n_eval_episodes=8,
                # eval_log_path=self.config.checkpoint_directory,
                # eval_env=self._create_env(),
            )
        else:
            raise NotImplementedError(
                "Training for the library {} is not implemented.".format(
                    library))

    # methods for hyperparameter tuning
    def _load_hyperparam_ss(self):
        """laod the search space configuration from a json file"""
        pass

    def _create_sampler(self):
        if self.config.hyperparameter_optimization.sampler == SamplerEnum.TPE_SAMPLER:
            sampler = TPESampler(
                n_startup_trials=self.config.hyperparameter_optimization.n_startup_trials,
                seed=self.config.hyperparameter_optimization.seed, multivariate=True
            )
        elif self.config.hyperparameter_optimization.sampler == SamplerEnum.SKOPT_SAMPLER:
            from optuna.integration.skopt import SkoptSampler

            # cf https://scikit-optimize.github.io/#skopt.Optimizer
            # GP: gaussian process
            # Gradient boosted regression: GBRT
            sampler = SkoptSampler(skopt_kwargs={"base_estimator": "GP", "acq_func": "gp_hedge"})
        elif self.config.hyperparameter_optimization.sampler == SamplerEnum.RANDOM:
            sampler = RandomSampler(seed=self.config.hyperparameter_optimization.seed)

        return sampler

    def _create_pruner(self):
        if self.config.hyperparameter_optimization.pruner == PrunerEnum.HALVING:
            pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
        elif pself.config.hyperparameter_optimization.pruner == PrunerEnum.MEDIAN:
            pruner = MedianPruner(n_startup_trials=self.n_startup_trials, n_warmup_steps=self.n_evaluations // 3)
        elif self.config.hyperparameter_optimization.pruner == PrunerEnum.NONE:
            # Do not prune
            pruner = NopPruner()
        elif self.config.hyperparameter_optimization.pruner == PrunerEnum.HYPERBAND:
            pruner = HyperbandPruner(
                min_resource=np.floor(self.config.train.total_timesteps/3),
                max_resource=self.config.train.total_timesteps,
                reduction_factor=3,
            )
        else:
            raise NotImplementedError('Pruner is not implemented.')
        return pruner

    def objective(self, trial: optuna.Trial) -> float:

        # create a copy of the current configuration
        config = copy.deepcopy(self.config)

        # sample candidate hyperparameters
        sampled_hyperparameters = HYPERPARAMS_SAMPLER[self.config.agent](trial)
        config = update_experiment_config(config, sampled_hyperparameters)

        # store the results
        results = EvaluationResults()

        # initialize callbacks for the optimization
        callbacks = []
        if self.config.hyperparameter_optimization.stop_reward is not None:
            callback_on_best = StopTrainingOnRewardThresholdAndSteps(
                reward_threshold=self.config.hyperparameter_optimization.stop_reward,
                min_training_steps=np.floor(self.config.train.total_timesteps / 4),
                verbose=config.hyperparameter_optimization.verbose,
                verbose_prefix='Trial ' + str(trial.number) + ': ',
                hold_reward_steps=np.floor(self.config.train.total_timesteps / 16)
            )
            callbacks.append(callback_on_best)
        else:
            callback_on_best = None

        eval_callback = TrialEvalCallback(
            eval_env=(eval_env := self._create_env(config=config)),
            is_multi_objective=self._study_is_multi_objective(),
            trial=trial,
            best_model_save_path=config.checkpoint.path,
            log_path=os.path.join(config.checkpoint.path, "logs"),
            n_eval_episodes=config.eval.n_episodes,
            results=results,
            verbose=config.hyperparameter_optimization.verbose,  # low output on evaluation
            verbose_prefix='Trial' + str(trial.number) + ': ',
            eval_freq=config.eval.frequency.as_tuple,
            eval_freq_max=config.eval.frequency_max.as_tuple if config.eval.frequency_max is not None else None,
            eval_start=config.eval.start,
            deterministic=config.deterministic,
            render=not config.experiment_headless,
            max_episode_length=config.eval.max_episode_length,
            rendering_step_delay=config.eval.step_delay,
            callback_on_new_best=callback_on_best,
            average_evaluations=self.config.hyperparameter_optimization.average_evaluations,
            total_timesteps=self.config.train.total_timesteps,
            warn=False,
        )
        # todo: add the possibility to also optimize parameters of the wrappers

        callbacks.append(eval_callback)

        try:
            self.run_training(
                verbose=config.hyperparameter_optimization.verbose - 1,  # reduce the verbosity for logging during training
                results=results,
                config=config,
                callbacks=callbacks,
                callbacks_append=False,
            )
            eval_env.close()
            # todo: do i need to close the model and env?
        except (AssertionError, ValueError) as e:
            eval_env.close()

            print(e)
            print("============")
            print("Sampled hyperparams:")
            print(sampled_hyperparameters)
            raise optuna.exceptions.TrialPruned()

        is_pruned = eval_callback.is_pruned
        reward = eval_callback.last_mean_reward

        del eval_env
        del results

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        if config.hyperparameter_optimization.average_evaluations == 'auc':
            auc = eval_callback.auc
            return auc
        return reward

    @delayed
    def _optimize(self, study_name: str, storage: str):
        study: optuna.Study = optuna.load_study(study_name=study_name, storage=self._load_storage(storage_url=storage))

        if self.config.hyperparameter_optimization.max_total_trials is not None:
            # Note: we count already running trials here otherwise we get
            #  (max_total_trials + number of workers) trials in total.
            counted_states = [
                TrialState.COMPLETE,
                TrialState.RUNNING,
                TrialState.PRUNED,
            ]
            completed_trials = len(study.get_trials(states=counted_states))
            if completed_trials < self.config.hyperparameter_optimization.max_total_trials:
                study.optimize(
                    self.objective,
                    #n_jobs=2,  # two jobs per optimize
                    callbacks=[
                        MaxTrialsCallback(
                            self.config.hyperparameter_optimization.max_total_trials,
                            states=counted_states,
                        )
                    ],
                )
        else:
            study.optimize(
                self.objective,
                #n_jobs=2,
                n_trials=self.config.hyperparameter_optimization.n_trials
            )

    def _set_study_name(self, study_name: str, study_name_suffix: str = ''):
        if study_name is None:
            study_name = 'optimize_' + self.config.experiment_name
        self.study_name = study_name + study_name_suffix
        return self.study_name

    def _study_is_multi_objective(self):
        return self.config.hyperparameter_optimization.optimize_timesteps

    def _study_direction(self):
        if self.config.hyperparameter_optimization.optimize_timesteps:
            return ['maximize', 'minimize']
        #elif self.config.hyperparameter_optimization.average_evaluations == 'auc':
        #    return ['maximize', 'maximize']
        else:
            return 'maximize'

    def _load_storage(self, heartbeat: bool = False, storage_url: str = 'sqlite:///optuna.db'):
        return optuna.storages.RDBStorage(
            url=storage_url,
            heartbeat_interval=self.config.hyperparameter_optimization.storage_heartbeat_interval,# if heartbeat else None,
            failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=3),
        )


    def hyperparameter_optimization(
            self,
            verbose: Optional[int] = None,
            search_space: Optional = None,
            log_path: Optional[str] = None,
            study_name: Optional[str] = None,
            study_name_suffix: str = '',
            n_processes: Optional[int] = None,
            storage_url: str = 'sqlite:///optuna.db'
    ):
        if verbose is not None:
            self.config.hyperparameter_optimization.verbose = verbose
        if self.config.hyperparameter_optimization.verbose > 0:
            print("Optimizing Hyperparameters")

        sampler = self._create_sampler()
        pruner = self._create_pruner()

        if self.config.hyperparameter_optimization.verbose > 0:
            print(f"Sampler: {self.config.hyperparameter_optimization.sampler} - Pruner: {self.config.hyperparameter_optimization.pruner}")
        study_name = self._set_study_name(study_name=study_name, study_name_suffix=study_name_suffix)

        # storage management
        storage = self._load_storage(storage_url=storage_url)

        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,  # in case an existing study was found this will still change the pruner to the given pruner!
            study_name=self.study_name,
            storage=storage,  # todo: fix this to a more general storage( e.g. using cloud gcp sql or some other sql db)
            load_if_exists=True,
            direction=self._study_direction(),
        )

        # fail trials marked as running without heartbeat these will then be restarted via failed_trial_callback
        optuna.storages.fail_stale_trials(study)

        if n_processes is None:
            n_processes = self.config.hyperparameter_optimization.n_processes

        try:
            r = Parallel(n_jobs=n_processes)([self._optimize(study_name, storage_url) for _ in range(n_processes)])  # two jobs per process to balance io load, , prefer="threads"
        except KeyboardInterrupt:
            print("Abort study! - keyboard interrupt")

        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("Value: ", trial.value)

        print("Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        report_name = (
            f"report_{study_name}-{self.config.hyperparameter_optimization.sampler}-{self.config.hyperparameter_optimization.pruner}_{int(time.time())}"
        )

        # log_path = os.path.join(self.log_folder, self.algo, report_name)
        if log_path is None:
            log_path = os.path.join('data', 'hyperparameter_optimization', report_name)
        if self.config.hyperparameter_optimization.verbose:
            print(f"Writing report to {log_path}")

        # Write report
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        study.trials_dataframe().to_csv(f"{log_path}.csv")

        # Save python object to inspect/re-use it later
        with open(f"{log_path}.pkl", "wb+") as f:
            pickle.dump(study, f)
        print('Report saved.')

        # Skip plots
        if self.config.hyperparameter_optimization.no_optim_plots:
            return

        # Plot optimization result
        try:
            fig1 = plot_optimization_history(study)
            fig2 = plot_param_importances(study)

            fig1.show()
            fig2.show()
        except (ValueError, ImportError, RuntimeError) as e:
            print(e)
            pass

    def enqueue_trial(
            self,
            force_parameters: Dict[str, Any],
            study_name: Optional[str] = None,
            study_name_suffix: Optional[str] = '',
            verbose: int = 1,
            from_best: bool = True,
            storage_url: str = 'sqlite:///optuna.db'
    ):
        study_name = self._set_study_name(study_name, study_name_suffix)
        study: optuna.Study = optuna.load_study(study_name=study_name, storage=self._load_storage(storage_url=storage_url))
        if from_best:
            enqueue_params = study.best_params
            enqueue_params.update(force_parameters)
        if verbose > 1:
            print(f'enqueueing Trial with parameters:{enqueue_params}')
        study.enqueue_trial(params=enqueue_params, skip_if_exists=True)
