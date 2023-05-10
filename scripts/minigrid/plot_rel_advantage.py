#%%
import math
import pandas as pd
import os
from kgrl.base.experiments.experiment_utils import accumulate_results
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#%%
data_dir = r"/deterministic"
experiments_folders = [
    os.path.join(data_dir, directory)
    for directory in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, directory)) and not ("dimensions" in directory)]
#%%
hardness_dict = dict(
    easy=['DoorKey-5x5', 'LavaGapS5'],
    medium=['DoorKey-8x8', 'LavaGapS7'],
    hard=['LavaCrossingS9N2', 'KeyCorridorS3R2', 'ObstructedMaze'],
)
data = dict(index=[], columns=['t_plain/t_alg'], data=[], index_names=['hardness', 'environment', 'method'], column_names=['rel_time'])
for hardness, env_list in hardness_dict.items():
    for environment in env_list:
        env_folders = [folder for folder in experiments_folders if environment in folder]
        # compute max reward of plain and corresponding timesteps
        rewards, lengths, new_index, level_names = accumulate_results(
            results_dir=os.path.join([folder for folder in env_folders if folder.endswith('v0')][0],
                                     '../../data/dumps'),
        )
        lengths['mean'].replace(1, lengths['mean'].max(), inplace=True)  # sometimes a lengths of 1 was reported at the beginning
        # experiments on the same environment have the same seeds!
        min_lengths = lengths.groupby(level='param_0').min()['mean']
        t_max_rewards = lengths.groupby(level='param_0').idxmin()['mean'].apply(lambda x: x[1])

        # compare with the timesteps the other approaches need to achieve this reward
        for folder in env_folders:
            if not folder.endswith('v0'):
                method = folder.split('v0_')[-1]
                rewards, lengths, new_index, level_names = accumulate_results(
                    results_dir=os.path.join(folder, '../../data/dumps'),
                )
                lengths['mean'].replace(1, lengths['mean'].max(), inplace=True)
                t_method_plain_max_reward = []
                for method_df_tuple, min_length_df_tuple in zip(lengths.groupby(level='param_0').__iter__(), min_lengths.groupby(level='param_0').__iter__()):
                    method_seed, method_df = method_df_tuple
                    max_reward_seed, min_length_df = min_length_df_tuple
                    assert method_seed == max_reward_seed, 'seed should be the same. Check ordering of the Dataframes.'
                    t_val = (method_df[method_df <= min_length_df.values[0]]).idxmin()['mean']
                    t_method_plain_max_reward.append(t_val if not isinstance(t_val, float) else (max_reward_seed, t_val))  # if the min length of the plain method is not achieved take the highest length

                idx, values = zip(*t_method_plain_max_reward)
                t_method_plain_max_reward = pd.Series(values, idx)

                data['index'].append((hardness, environment, method))
                data['data'].append(t_max_rewards.divide(t_method_plain_max_reward).fillna(0).mean())  # zero if the reward was not achieved
            else:
                data['index'].append((hardness, environment, 'plain'))
                data['data'].append(1)


df = pd.DataFrame.from_dict(data, orient='tight')
df = df.reset_index()
fig, axs = plt.subplots(ncols=1, figsize=(5, 4))
#for name, data in df.groupby(level='method').__iter__():
ax = sns.scatterplot(
    data=df,
    x='hardness',
    y='t_plain/t_alg',
    ax=axs,
    style='method',
    hue='method',
)
ax.set_yscale('log')
ax.set_yticks([0.5,1,2,3,4,5])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_ylabel(r'$t^{r_{plain}}_{plain}/t^{r_{plain}}_{\ast}$')
# ax.set(title='Method_advantage')
fig.savefig(
        fname='data/images/deterministic/plot_hard.png',
        bbox_inches='tight',  # kwargs.get('bbox_inches', 'tight'),
        dpi=600)  # kwargs.get('dpi', 600))
print('done')
