import os
from enum import Enum
from typing import NamedTuple, Union, Tuple, Optional

import numpy as np
import optuna
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import sync_envs_normalization

from kgrl.base.experiments.evaluate_policy import evaluate_policy

class EvalFrequencyUnit(Enum):
    STEP = "step"
    EPISODE = "episode"


class EvalFreq(NamedTuple):
    frequency: Union[int, Tuple[int, int]]
    unit: EvalFrequencyUnit


def convert_to_eval_freq(eval_freq: Union[int, Tuple[int, str], EvalFreq]) -> EvalFreq:
    if not isinstance(eval_freq, EvalFreq):
        if not isinstance(eval_freq, tuple):
            eval_freq = (eval_freq, "episode")
        # type checking happens in the Enum class constructor
        return EvalFreq(eval_freq[0], EvalFrequencyUnit(eval_freq[1]))
    else:
        raise TypeError('Argument eval_freq does not match a supported type.')

class EpisodeEvalCallback(EvalCallback):
    def __init__(
            self,
            eval_freq: Union[int, Tuple[int, str], EvalFreq] = (4, "episode"),
            eval_freq_max: Optional[Union[int, Tuple[int, str], EvalFreq]] = None,
            eval_start: int = 5,
            max_episode_length: int = None,
            rendering_step_delay: float = None,
            log_episode_lengths_and_rewards: bool = False,
            verbose_prefix: str = '',
            **kwargs,
    ):
        self.max_episode_length = max_episode_length

        if 'results' in kwargs:
            self.results = kwargs.pop('results')
        else:
            self.results = None

        super().__init__(**kwargs)
        self.env = kwargs['eval_env']
        self.eval_freq = convert_to_eval_freq(eval_freq)
        if eval_freq_max is not None:
            self.eval_freq_max = convert_to_eval_freq(eval_freq_max)
        else:
            self.eval_freq_max = None
        self.eval_start = eval_start
        self.last_eval: int = -1  # timesteps finished at last evaluation
        self.last_eval_ep: int = eval_start - self.eval_freq.frequency if self.eval_freq.unit == EvalFrequencyUnit.EPISODE else -1
        self.n_eval = 0
        self.evaluations_episodes = []
        self.rendering_step_delay = rendering_step_delay
        self.log_episode_lengths_and_rewards = log_episode_lengths_and_rewards
        self.verbose_prefix = verbose_prefix

    def _convert_eval_freq(self):
        """Convert eval_freq to an EvalFreq instance."""
        if not isinstance(self.eval_freq, EvalFreq):
            eval_freq = self.eval_freq
            if not isinstance(eval_freq, tuple):
                eval_freq = (eval_freq, "episode")
            # type checking happens in the Enum class constructor
            self.eval_freq = EvalFreq(eval_freq[0], EvalFrequencyUnit(eval_freq[1]))

    def _on_step(self) -> bool:
        if self.n_calls > self.eval_start and self._do_evaluation():
            self._evaluate()
        return True

    def _evaluate(self):
        self.last_eval = self.num_timesteps
        self.last_eval_ep = self.model._episode_num
        self.n_eval += 1

        try:
            self.env.start_evaluation()
        except AttributeError:
            pass

        if self.model.get_vec_normalize_env() is not None:
            try:
                sync_envs_normalization(self.training_env, self.eval_env)
            except AttributeError:
                raise AssertionError(
                    "Training and eval env are not wrapped the same way, "
                    "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                    "and warning above."
                )

            # Reset success rate buffer
        self._is_success_buffer = []

        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            render=self.render,
            deterministic=self.deterministic,
            return_episode_rewards=True,
            warn=self.warn,
            callback=self._log_success_callback,
            max_episode_length=self.max_episode_length,
            evaluation_step_delay=self.rendering_step_delay,
            track_agent_actions_diversity=True,
        )

        assert len(episode_lengths) == len(episode_rewards)

        if self.log_episode_lengths_and_rewards and verbose > 0:

            for i, (episode_reward, episode_length) in enumerate(
                    zip(episode_rewards, episode_lengths)):
                print(
                    f'🔵 episode index = {i:-5d} \tepisode length = {episode_length:-10d} \tepisode reward = {episode_reward:-10.3f}'
                )
            print(
                f'🟢 n episodes = {len(episode_lengths):-5d}   ' +
                f'mean episode length = {np.mean(episode_lengths):-10.3f}        mean reward = {np.mean(episode_rewards):-10.3f}'
            )

        if self.results is not None:
            self.results.append_episode_lengths(episode_lengths)
            self.results.append_episode_rewards(episode_rewards)
            self.results.append_timesteps(self.num_timesteps)

        if self.log_path is not None:
            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(episode_rewards)
            self.evaluations_length.append(episode_lengths)
            self.evaluations_episodes.append(self.model._episode_num)

            kwargs = {}
            # Save success log if present
            if len(self._is_success_buffer) > 0:
                self.evaluations_successes.append(self._is_success_buffer)
                kwargs = dict(successes=self.evaluations_successes)

            np.savez(
                self.log_path,
                timesteps=self.evaluations_timesteps,
                results=self.evaluations_results,
                ep_lengths=self.evaluations_length,
                episdoes=self.evaluations_episodes,
                **kwargs,
            )

        mean_reward, std_reward = np.mean(episode_rewards), np.std(
            episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(
            episode_lengths)
        self.last_mean_reward = mean_reward

        if self.verbose > 0:
            print(
                self.verbose_prefix + f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            print(
                f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
        # Add to current Logger
        self.logger.record("eval/mean_reward", float(mean_reward))
        self.logger.record("eval/mean_ep_length", mean_ep_length)

        if len(self._is_success_buffer) > 0:
            success_rate = np.mean(self._is_success_buffer)
            if self.verbose > 0:
                print(f"Success rate: {100 * success_rate:.2f}%")
            self.logger.record("eval/success_rate", success_rate)

        # Dump log so the evaluation results are printed with the correct timestep
        self.logger.record("time/total_timesteps", self.num_timesteps,
                           exclude="tensorboard")
        # self.logger.dump(self.num_timesteps)

        try:
            self.env.stop_evaluation()
        except AttributeError:
            pass

        if mean_reward > self.best_mean_reward:
            if self.verbose > 0:
                print("New best mean reward!")
            if self.best_model_save_path is not None:
                self.model.save(
                    os.path.join(self.best_model_save_path, "best_model"))
            self.best_mean_reward = mean_reward
            # Trigger callback if needed
            if self.callback is not None:
                return self._on_event()

        # print("training...")

    def _do_evaluation(self,) -> bool:
        do = False
        if self.eval_freq.unit == EvalFrequencyUnit.STEP:
            do = self.eval_freq.frequency > 0 and self.n_calls % self.eval_freq.frequency == 0
        elif self.eval_freq.unit == EvalFrequencyUnit.EPISODE:
            if self.eval_freq.frequency > 0:
                # print('--', self.model._episode_num)
                do = (self.model._episode_num - self.eval_start + 1) // self.eval_freq.frequency >= self.n_eval and (self.model._episode_num - self.last_eval_ep) >= self.eval_freq.frequency

        if do and self.eval_freq_max is not None and self.eval_freq_max.unit == EvalFrequencyUnit.STEP:
            do = (self.num_timesteps - self.last_eval) >= self.eval_freq_max.frequency
        elif do and self.eval_freq_max is not None and self.eval_freq_max.unit == EvalFrequencyUnit.EPISODE:
            do = (self.model._episode_num - self.last_eval_ep) >= self.eval_freq_max.frequency

        return do


class TrialEvalCallback(EpisodeEvalCallback):
    def __init__(
            self,
            trial: optuna.Trial,
            average_evaluations: Union[int, str] = 3,
            is_multi_objective: bool = False,
            total_timesteps: int = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
        self.average_evaluations = average_evaluations
        self.report_reward = []
        self.pre_auc = 0
        self.auc = 0
        self._is_multi_objective = is_multi_objective
        self._total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        if self.n_calls > self.eval_start and self._do_evaluation():
            last_eval = 0 if self.last_eval == -1 else self.last_eval
            self._evaluate()
            if self.average_evaluations == 'auc':
                self.pre_auc += self.last_mean_reward * (self.num_timesteps - last_eval)
                self.auc = float(
                    (self.pre_auc + self.last_mean_reward * (self._total_timesteps - self.num_timesteps)) / self._total_timesteps
                )  # adjust the auc for the entire possible training timesteps (otherwise slow but steady would be preferred)
                self.trial.report(self.auc, self.num_timesteps)  # time normalized auc
            else:
                self.report_reward.append(self.last_mean_reward)
                if len(self.report_reward) >= self.average_evaluations:
                    if not self._is_multi_objective:  # report does not work with the multiobjective (at least it says so in the docs)
                        self.trial.report(float(np.mean(self.report_reward)), self.num_timesteps)
                    self.report_reward.pop(0)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


class StopTrainingOnRewardThresholdAndSteps(StopTrainingOnRewardThreshold):
    def __init__(self, reward_threshold: float, min_training_steps: int = 0, verbose: int = 0, hold_reward_steps: int = 0, verbose_prefix: str = ''):
        super().__init__(reward_threshold=reward_threshold, verbose=verbose)
        self.min_training_steps = min_training_steps
        self.achieved_reward_threshold_step: Optional[int] = None
        self.hold_reward_steps = hold_reward_steps
        self.verbose_prefix = verbose_prefix

    def _on_step(self) -> bool:
        if self.num_timesteps > (self.min_training_steps-self.hold_reward_steps):
            continue_training: bool = bool(self.parent.last_mean_reward < self.reward_threshold)
            if not continue_training and self.achieved_reward_threshold_step is None:
                self.achieved_reward_threshold_step = self.num_timesteps

            if self.achieved_reward_threshold_step is not None:
                if not continue_training and (self.num_timesteps >= self.achieved_reward_threshold_step + self.hold_reward_steps):
                    if self.verbose > 0:
                        print(
                            self.verbose_prefix + f"Stopping training because the mean reward {self.parent.best_mean_reward:.2f} "
                            f" is above the threshold {self.reward_threshold}"
                            f" and the additional {self.hold_reward_steps} have passed."
                        )
                    return False
                elif continue_training:
                    self.achieved_reward_threshold_step = None  # unset the achieved threshold steps if the threshold is not held
                    return True
                else:
                    return True
            return True
        else:
            return True
