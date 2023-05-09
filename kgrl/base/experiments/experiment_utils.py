import os
from typing import Optional, Tuple, List, Dict, Any, Union

import datetime
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


import click
"""
@click.group()
def main():
    pass

@main.command('accumulate')
@click.argument('results_dir', type=str)
@click.argument('prefix', type=str)
@click.option('--key', type=str, default='seed')
@click.option('--delimiter', type=str, default='\t')
"""
def accumulate_results(
        results_dir: str,
        prefix: Optional[str] = None,
        delimiter: Optional[str] = '\t',
        x_range: Optional[Tuple[int, int]] = None,
        y_range: Optional[List[Tuple[int, int]]] = None,
):
    """read in the dumps in results_dir with the given prefix and aggregate them."""
    if prefix is None:
        prefix = get_latest_dumps_prefix(results_dir)
    # read in the results
    lengths = pd.read_csv(
        os.path.join(results_dir, prefix+'-lengths.tsv'),
        delimiter=delimiter,
        index_col=[0, 1],
        header=None,
    )

    rewards = pd.read_csv(
        os.path.join(results_dir, prefix+'-rewards.tsv'),
        delimiter=delimiter,
        index_col=[0, 1],
        header=None,
    )

    timesteps = pd.read_csv(
        os.path.join(results_dir, prefix+'-timesteps.tsv'),
        delimiter=delimiter,
        index_col=[0, 1],
        header=None,
        names=['timesteps']
    )

    all_timesteps = np.unique(timesteps['timesteps'].values)
    if x_range is not None:
        all_timesteps = all_timesteps[np.logical_and(all_timesteps >= x_range[0], all_timesteps <= x_range[1])]
    all_keys = np.unique(timesteps.reset_index()['level_0'].values)
    index_lists = list(zip(*[k.split(';') for k in all_keys.tolist()]))
    index_lists = [list(set(item)) for item in index_lists]
    level_names = ['param_' + str(i) for i in range(len(all_keys[0].split(';')))]

    # get multiindex including all params in the keys
    new_index = pd.MultiIndex.from_product([*index_lists, all_timesteps], names=[*level_names, 'timesteps'])

    def process_df(df):
        # join timesteps into the dataframe
        df = df.join(timesteps, on=[0, 1])

        # set key and timesteps as index
        df = df.reset_index()
        df[level_names] = df[0].str.split(';', expand=True)  # split the id into the params
        df = df.drop(0, axis=1)  # drop the id axis
        df = df.set_index([*level_names, 'timesteps'])

        # reindex
        df = df.reindex(new_index)

        # interpolate linearly
        for value in list(set(zip(*[new_index.get_level_values(level) for level in level_names]))):  # create a list of tuples corresponding to the available choices of values in the first levels (level_names)of the multiindex
            # key is a tuple of sample values in the levels named in level_names
            df.loc[value, :].interpolate(method='index', inplace=True)

        # compute mean across one eval step (of different evaluation episodes)
        df['mean'] = df.drop(1, axis=1).mean(axis=1)
        return df

    rewards = process_df(rewards)
    lengths = process_df(lengths)
    return rewards, lengths, new_index, level_names


def plot_results(
        results_dir: str,
        prefix: Optional[str] = None,
        image_name: str = 'bb.png',
        key: str = 'seed',
        delimiter: Optional[str] = '\t',
        label: str = '',
        fig: matplotlib.figure.Figure = None,
        axs = None,
        x_range: Optional[Tuple[int, int]] = None,
        y_range: Optional[List[Tuple[int, int]]] = None,
        smoothing: Optional[int] = 20,
        confidence_interval: bool = True,
        **kwargs,
):
    rewards, lengths, new_index, level_names = accumulate_results(
        results_dir=results_dir,
        prefix=prefix,
        delimiter=delimiter,
        x_range=x_range,
        y_range=y_range,
    )
    # plot figure with confidence intervals
    # init plot
    # fig = plt.Figure()
    levels_without_key = [level_name for level_name in level_names if
                          key not in new_index.get_level_values(level_name)[0]]
    key_levels = [level_name for level_name in level_names if
                  level_name not in levels_without_key]

    # init figure
    if fig is None or axs is None:
        fig, axs = plt.subplots(ncols=2, figsize=(8, 3))

    assert isinstance(fig, matplotlib.figure.Figure) and isinstance(axs, np.ndarray), 'Figure has to be of type matplotlib.figure.Figure'

    sns.set(font_scale=kwargs.get('font_scale', 0.8))

    def create_ax(
            data: pd.DataFrame,
            plot_num: int,
            title: str = None,
            data_label: str = None,
            levels: Optional[Union[list[str], str]] = None
    ):
        if smoothing is not None:
            if confidence_interval:
                ax_temp = sns.lineplot(
                    data.groupby(level=levels, group_keys=False).apply(lambda x: x.rolling(smoothing).mean()),
                    x='timesteps',
                    y='mean',
                    ax=axs[plot_num],
                    label=data_label,)
            else:
                ax_temp = sns.lineplot(
                    data=data.rolling(smoothing).mean(),
                    x='timesteps',
                    y='mean',
                    ax=axs[plot_num],
                    label=data_label,)
        else:
            ax_temp = sns.lineplot(
                data=data,
                x='timesteps',
                y='mean',
                ax=axs[plot_num],
                label=data_label)
        ax_temp.set(title=title)

    if len(levels_without_key) > 0:
        value_list = list(set(zip(*[new_index.get_level_values(level) for level in
                          levels_without_key])))
        if value_list[0][0].split('=')[-1].isnumeric():
            value_list.sort(key=lambda x: int(x[0].split('=')[-1]))
        else:
            value_list.sort(key=lambda x: x[0].split('=')[-1])
        for value in value_list:
            value_label = value[0].split('.')[-1]
            rewards_mean = rewards.unstack(level=key_levels).loc[value, :].stack()
            lengths_mean = lengths.unstack(level=key_levels).loc[value, :].stack()
            create_ax(rewards_mean, 0, 'Mean Rewards', label + str(value_label), levels=key_levels)
            create_ax(lengths_mean, 1, 'Mean Lengths', label + str(value_label), levels=key_levels)
    else:
        rewards_mean = rewards.unstack(level=key_levels).stack()
        lengths_mean = lengths.unstack(level=key_levels).stack()
        create_ax(rewards_mean, 0, 'Mean Rewards', label, levels=key_levels)
        create_ax(lengths_mean, 1, 'Mean Lengths', label, levels=key_levels)

    # save fig
    save_path = image_name
    fig.savefig(
        save_path,
        bbox_inches=kwargs.get('bbox_inches', 'tight'),
        dpi=kwargs.get('dpi', 600))
    print(f'figure saved to {save_path}')
    return fig, axs


def get_latest_dumps_prefix(
        dumps_dir: str,
) -> str:
    date_format = "%d-%m-%Y %H-%M-%S"
    prefixes = [string.replace('-timesteps.tsv','') for string in os.listdir(dumps_dir) if 'timesteps' in string]
    times = [datetime.datetime.strptime(prefix[-19:], date_format) for prefix in prefixes]
    max_index = np.argmax(times)
    return prefixes[max_index]


def compute_relative_timesteps(
        run_result_dirs: Dict[str, str],
        prefixes: Optional[List[str]] = None,
        n_steps: int = 201,
        reward_range: Tuple[float, float] = (.0, 1.),
        label: str = '',
        key: str = 'seed',
        metric: str = 't/tbest',
        delimiter: str = '\t',
        fig: matplotlib.figure.Figure = None,
        axs: Any = None,  #todo: define class
        x_range: Optional[Tuple[int, int]] = None,
        y_range: Optional[List[Tuple[int, int]]] = None,
        # smoothing: int = 1,
) -> pd.DataFrame:
    if prefixes is not None:
        assert len(run_result_dirs.items()) == len(prefixes), "length of runs_results_dirs and prefixes does not match"
    mean_rewards_df = pd.DataFrame()
    mean_lengths_df = pd.DataFrame()
    for id, item in enumerate(run_result_dirs.items()):
        label, dir = item
        prefix = prefixes[id] if prefixes is not None else get_latest_dumps_prefix(dumps_dir=dir)
        rewards, lengths, new_index, level_names = accumulate_results(
            results_dir=dir,
            prefix=prefix,
            delimiter=delimiter,
            x_range=x_range,
            y_range=y_range,
        )

        levels_without_key = [level_name for level_name in level_names if
                              key not in new_index.get_level_values(level_name)[
                                  0]]
        key_levels = [level_name for level_name in level_names if
                      level_name not in levels_without_key]

        if id == 0:
            mean_rewards_df[label] = rewards['mean'].unstack(level=key_levels).mean(axis=1)
            mean_lengths_df[label] = lengths['mean'].unstack(level=key_levels).mean(axis=1)
        else:
            mean_rewards_df = mean_rewards_df.join(pd.DataFrame(rewards['mean'].unstack(level=key_levels).mean(axis=1), columns=[label]), how='outer')
            mean_lengths_df = mean_lengths_df.join(pd.DataFrame(lengths['mean'].unstack(level=key_levels).mean(axis=1), columns=[label]), how='outer')  # [dir.split('\\')[-2]]

    # interpolate to fill missing values (from join)
    mean_rewards_df.interpolate(method='index', inplace=True)
    mean_lengths_df.interpolate(method='index', inplace=True)

    # compute the statistic
    rel_times = pd.DataFrame()
    for reward in np.linspace(0, 1, n_steps):
        timesteps_until_reward = mean_rewards_df.where(mean_rewards_df >= reward).where(mean_rewards_df <= reward, other=0).idxmin()
        relative_timesteps = timesteps_until_reward.min()/timesteps_until_reward
        rel_times = pd.concat([rel_times, pd.DataFrame({reward: relative_timesteps}).T], axis=0)
        if not np.isnan(timesteps_until_reward.min()):#and reward<0.995:
            rel_times = rel_times.fillna(0)

    rel_times.index.name = 'reward'
    return rel_times

def average_relative_timesteps(rels: List[pd.DataFrame]) -> pd.DataFrame:
    concat = pd.concat(rels, axis=1)
    average_rel = pd.DataFrame()
    for label in ['plain', 'knn', 'rw', 'rw-ner']:  # list(set(concat.columns)):
        average_rel[label] = concat[label].mean(axis=1)
    return average_rel


def plot_rel_times(rel_times: pd.DataFrame, save_path: str = 'bla.png', **kwargs):
    axes = rel_times.plot(
        figsize=kwargs.get('figsize', (4.5, 1.5)),
        xlabel='Reward',
        ylabel="Relative Timesteps",
        xlim=kwargs.get('xlim', (0, 1)),
        ylim=(0, 1.1),
    )
    axes.get_figure().savefig(
        save_path,
        bbox_inches=kwargs.get('bbox_inches', 'tight'),
        dpi=kwargs.get('dpi', 600))
    print(f'figure saved to {save_path}')

