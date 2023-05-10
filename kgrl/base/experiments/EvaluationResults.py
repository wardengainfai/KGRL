import os
from typing import List, Tuple

import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib

matplotlib.use("Agg")
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from .exceptions import EvaluationResultsInitializationException


class EvaluationResults:
    def __init__(self, label: str = None, plot_losses_in_separate_file: bool = False):
        self.label = 'foo' if label is None else label

        self.episode_lengths = None  # [[1.0, 1.1], [2.0, 3.0], [2.0, 3.0]]  # first dimension - evaluation step, second dimension - number of episode
        self.episode_rewards = None  # [[2.0, 1.1], [2.0, 6.0], [2.0, 3.0]]
        self.losses = None

        self.fig = None
        self.axs = None

        self.fig_losses = None
        self.ax_losses = None

        self.gs = None

        # self.experiment_label_to_episode_lengths = {}
        self.episode_length_history = None
        # self.experiment_label_to_episode_rewards = {}
        self.episode_reward_history = None

        #record the timestep done in the env up to evaluation
        self.timesteps = None
        self.timesteps_history = None

        # self.plotted_losses = False
        self.loss_history = None

        self.enable_losses = True
        self.plot_losses_in_separate_file = plot_losses_in_separate_file
        self._history_is_in_long_format = False

    # Push evaluation results

    def append_episode_lengths(self, lengths: List[int]):
        if self.episode_lengths is None:
            self.episode_lengths = [tuple(lengths)]
        else:
            self.episode_lengths.append(tuple(lengths))

    def append_episode_rewards(self, rewards: List[float]):
        if self.episode_rewards is None:
            self.episode_rewards = [tuple(rewards)]
        else:
            self.episode_rewards.append(tuple(rewards))

    def append_timesteps(self, timesteps: int):
        if self.timesteps is None:
            self.timesteps = [timesteps]
        else:
            self.timesteps.append(timesteps)

    def set_losses(self, losses: Tuple[float]):
        self.losses = losses

    # Convert evaluation results to numpy arrays

    @property
    def episode_lengths_as_tensor(self):
        return np.array(self.episode_lengths)

    @property
    def episode_rewards_as_tensor(self):
        return np.array(self.episode_rewards)

    # @property
    # def losses_as_tensor(self):
    #     return np.array(self.losses)

    # Convert evaluation results to data frames

    def episode_lengths_as_dataframe(self, label: str = None):
        if self.episode_lengths is None:
            items = ()
        else:
            items = (
                (label, i, j, episode_length)
                for i, lengths in enumerate(self.episode_lengths)
                for j, episode_length in enumerate(lengths)
            )
        return pd.DataFrame(items, columns=('label', 'step', 'episode', 'length'))

    def episode_rewards_as_dataframe(self, label: str = None):
        if self.episode_lengths is None:
            items = ()
        else:
            items = (
                (label, i, j, episode_reward)
                for i, rewards in enumerate(self.episode_rewards)
                for j, episode_reward in enumerate(rewards)
            )

        return pd.DataFrame(items, columns=('label', 'step', 'episode', 'reward'))

    def losses_as_dataframe(self, label: str = None):
        return pd.DataFrame(
            zip(self.losses),
            columns=(label,)
        )

    def timesteps_as_dataframe(self, label: str = None):
        if self.timesteps is None:
            items = ()
        else:
            items = (
                (label, step, timestep)
                for step, timestep in enumerate(self.timesteps)
            )
        return pd.DataFrame(items, columns=('label', 'step', 'timestep'))



    # Collect and visualize results

    def init_subplots(self, enable_losses: bool = True):
        if enable_losses and not self.plot_losses_in_separate_file:
            self.fig = fig = plt.figure(figsize=(25, 19))
            self.gs = gs = GridSpec(2, 2, figure=fig)
            self.axs = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :]))
        else:
            self.fig, self.axs = plt.subplots(ncols=2, figsize=(25, 8))
            self.enable_losses = enable_losses
            if enable_losses and self.plot_losses_in_separate_file:
                self.fig_losses, self.ax_losses = plt.subplots(figsize=(25, 8))

    def reset(self, label: str = None, long_format: bool = False):
        if label is not None and self.episode_lengths is not None and self.episode_rewards is not None and self.timesteps is not None:
            # self.experiment_label_to_episode_lengths[label] = self.episode_lengths
            # self.experiment_label_to_episode_rewards[label] = self.episode_rewards
            if self.episode_length_history is None:
                if long_format:
                    self.episode_length_history = self.episode_lengths_as_dataframe(label=label)
                    self._history_is_in_long_format = True
                else:
                    self.episode_length_history = pd.DataFrame((label, i, *lengths) for i, lengths in enumerate(self.episode_lengths))
            else:
                if long_format:
                    assert self._history_is_in_long_format, 'History is not in long format, cannot append current results to existing ones'
                    df = self.episode_lengths_as_dataframe(label=label)
                else:
                    df = pd.DataFrame((label, i, *lengths) for i, lengths in enumerate(self.episode_lengths))

                self.episode_length_history = self.episode_length_history.append(df, ignore_index=True)

            if self.episode_reward_history is None:
                if long_format:
                    self.episode_reward_history = self.episode_rewards_as_dataframe(label=label)
                else:
                    self.episode_reward_history = pd.DataFrame((label, i, *rewards) for i, rewards in enumerate(self.episode_rewards))
            else:
                if long_format:
                    df = self.episode_rewards_as_dataframe(label=label)
                else:
                    df = pd.DataFrame((label, i, *rewards) for i, rewards in enumerate(self.episode_rewards))

                self.episode_reward_history = self.episode_reward_history.append(df, ignore_index=True)

            if self.timesteps_history is None:
                if long_format:
                    self.timesteps_history = self.timesteps_as_dataframe(label=label)
                else:
                    self.timesteps_history = pd.DataFrame((label, i, timesteps) for i, timesteps in enumerate(self.timesteps))
            else:
                if long_format:
                    df = self.timesteps_as_dataframe(label=label)
                else:
                    df = pd.DataFrame((label, i, timesteps) for i, timesteps in enumerate(self.timesteps))
                self.timesteps_history = self.timesteps_history.append(df, ignore_index=True)

            if self.losses is not None and self.enable_losses:
                if self.loss_history is None:
                    self.loss_history = self.losses_as_dataframe(label=label)
                else:
                    self.loss_history = pd.concat((self.loss_history, self.losses_as_dataframe(label=label)), axis=1)

                self.losses = None

            self.episode_lengths = None
            self.episode_rewards = None
            self.timesteps = None

    def plot(self, label: str = None, with_confidence_interval: bool = False, xkey: str = 'timestep'):
        first_run = False

        if self.fig is None:
            first_run = True
            self.init_subplots(enable_losses=self.losses is not None)

        axs = self.axs

        if self.episode_lengths is None or self.episode_rewards is None:
            raise EvaluationResultsInitializationException('EvaluationResults is poorly initialized probably because of an early stop (episode lengths and corresponding rewards are missing)')

        if first_run or not with_confidence_interval and xkey == 'step':  # xticks are needed for initializing figure axis and drawing plots which don't need a confidence interval
            xticks = range(len(self.episode_lengths))
        elif first_run or not with_confidence_interval and xkey == 'timestep':
            xticks = self.timesteps
        else:
            xticks = range(1)

        def create_ax(ykey, ylabel, title, ax_num):
            if ykey == 'length':
                df = self.episode_lengths_as_dataframe(label=label)
            elif ykey == 'reward':
                df = self.episode_rewards_as_dataframe(label=label)

            if with_confidence_interval:
                ax = sns.lineplot(
                    data=df.merge(self.timesteps_as_dataframe(label=label), how='left', left_on=('label', 'step'), right_on=('label', 'step')),
                    x=xkey, y=ykey, ax=axs[ax_num], label=label)
            else:
                ax = sns.lineplot(x=xticks,
                                  y=self.episode_lengths_as_tensor.mean(axis=1) if ykey == 'length' else self.episode_rewards_as_tensor.mean(axis=1),
                                  ax=axs[ax_num], label=label)

            if first_run:
                ax.set(xlabel='evaluation step' if xkey == 'step' else xkey,
                       ylabel=ylabel, title=title)

        create_ax(ykey='length', ylabel='mean episode length', title='Episode length', ax_num=0)
        create_ax(ykey='reward', ylabel='mean reward', title='Cumulative reward', ax_num=1)

        # plot losses

        if self.losses is not None and self.enable_losses and not with_confidence_interval:  # and not self.plotted_losses:
            # ax = sns.lineplot(x = (xticks := range(len(self.losses))), y = self.losses, ax = axs[2])
            # ax.set(xlabel = 'training step', ylabel = 'loss', title = 'kg embeddings model training losses', xticks = xticks)
            ax = sns.lineplot(x = range(len(self.losses)), y = self.losses, ax = self.ax_losses if self.plot_losses_in_separate_file else axs[2], label = label)
            if first_run:
                ax.set(xlabel='training step', ylabel='loss', title='kg embeddings model training losses', xticks=[])
                # self.plotted_losses = True

    def _plot_losses_with_confidence_interval(self):
        df = self.loss_history.copy()
        df['index'] = df.index
        melted = pd.melt(df, id_vars=('index',), value_vars=df.columns)

        ax = sns.lineplot(
            data=melted,
            x='index',
            y='value',
            ax=self.ax_losses if self.plot_losses_in_separate_file else self.axs[2])
        ax.set(xlabel='training step', ylabel='loss', title='kg embeddings model training losses', xticks=[])

    def save(self, folder: str = 'data/images',
             dumps_folder: str = 'data/dumps', as_single_dump: bool = False,
             with_confidence_interval: bool = False):
        os.makedirs(folder, exist_ok=True)
        os.makedirs(dumps_folder, exist_ok=True)

        if self.enable_losses and with_confidence_interval and self.episode_length_history is not None:
            self._plot_losses_with_confidence_interval()

        if self.enable_losses and self.plot_losses_in_separate_file:
            self.fig.savefig(os.path.join(folder, f'{self.label}.rewards.png'), bbox_inches='tight')
            self.fig_losses.savefig(os.path.join(folder, f'{self.label}.losses.png'), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(folder, f'{self.label}.png'), bbox_inches = 'tight')

        if as_single_dump:
            pd.merge(
                self.episode_length_history, self.episode_reward_history,
                how='left', left_on=('label', 'step', 'episode'),
                right_on=('label', 'step', 'episode')
            ).merge(
                self.timesteps_history, how='left', left_on=('label', 'step'), right_on=('label', 'step')
            ).to_csv(os.path.join(dumps_folder, f'{self.label}.tsv'), sep='\t',
                     index=False)
        else:
            if self.episode_length_history is not None:
                self.episode_length_history.to_csv(
                    os.path.join(dumps_folder, f'{self.label}-lengths.tsv'),
                    sep='\t', index=False, header=False)
                self.episode_reward_history.to_csv(
                    os.path.join(dumps_folder, f'{self.label}-rewards.tsv'),
                    sep='\t', index=False, header=False)
                self.timesteps_history.to_csv(
                    os.path.join(dumps_folder, f'{self.label}-timesteps.tsv'),
                    sep="\t", index=False, header=False)


        if self.loss_history is not None and self.enable_losses:
            self.loss_history.to_csv(
                os.path.join(dumps_folder, f'{self.label}-losses.tsv'),
                sep='\t', index=False)
