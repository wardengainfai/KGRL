from kgrl.base.experiments.experiment_utils import plot_results, compute_relative_timesteps, plot_rel_times, average_relative_timesteps
import os
eval_mode = "deterministic"
image_path = os.path.join("data", "images", eval_mode)
os.makedirs(image_path, exist_ok=True)
#%% lava - gap 5
results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-LavaGapS5-v0\dumps'
prefix = 'lava-gap5-11-01-2023 09-49-18'
key = 'seed'
label = 'plain'

fig, axs = plot_results(
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
    x_range=(0, 350000)
)

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-LavaGapS5-v0_knn\dumps'
prefix = 'lava-gap5-kg-knn-11-01-2023 09-49-21'
key = 'seed'
delimiter = '\t'
label = 'knn'

fig, axs = plot_results(
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
    fig=fig,
    axs=axs,
)

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-LavaGapS5-v0_rw\dumps'
prefix = 'lava-gap5-kg-rw-11-01-2023 09-49-11'
key = 'seed'
label = 'rw'

fig, axs = plot_results(
    #image_name='data/images/Lava-Gap5.png',
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
    fig=fig,
    axs=axs
)

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-LavaGapS5-v0_rw_ner\dumps'
key = 'seed'
label = 'rw-ner'

fig, axs = plot_results(
    image_name=os.path.join(image_path, "Lava-Gap5.png"),
    results_dir=results_dir,
    key=key,
    label=label,
    fig=fig,
    axs=axs
)
#%% LavaGap 7
results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-LavaGapS7-v0\dumps'
prefix = 'lava-gap7-20-12-2022 16-59-11'
label = 'plain'
key = 'seed'
delimiter = '\t'
fig, axs = plot_results(
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
)

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-LavaGapS7-v0_knn\dumps'
prefix = 'lava_gap7-kg-knn-20-12-2022 17-01-12'
label = 'knn'
fig, axs = plot_results(
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
    fig=fig,
    axs=axs,
)

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-LavaGapS7-v0_rw\dumps'
prefix = 'lava-gap7-kg-rw-20-12-2022 16-57-19'
label = 'rw'
fig, axs = plot_results(
    image_name=os.path.join(image_path,"Lava-Gap7.png"),
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
    fig=fig,
    axs=axs,
)

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-LavaGapS7-v0_rw_ner\dumps'
prefix = 'lava-gap7-kg-rw-20-12-2022 16-57-19'
label = 'rw-ner'
fig, axs = plot_results(
    image_name=os.path.join(image_path, "Lava-Gap7.png"),
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
    fig=fig,
    axs=axs,
)

#%% DoorKey 5x5
results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-DoorKey-5x5-v0\dumps'
prefix = 'door-key-5x5-04-01-2023 16-20-04'
label = 'plain'
key = 'seed'
delimiter = '\t'
x_range = (0, 200000)
fig, axs = plot_results(
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
    x_range=x_range,
)

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-DoorKey-5x5-v0_knn\dumps'
prefix = 'door-key-5x5-kg-knn-04-01-2023 21-52-30'
label = 'knn'
fig, axs = plot_results(
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
    fig=fig,
    axs=axs,
    x_range=x_range
)

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-DoorKey-5x5-v0_rw\dumps'
prefix = 'door-key-5x5-kg-rw-04-01-2023 21-52-45'
label = 'rw'
fig, axs = plot_results(
    image_name=os.path.join(image_path, "DoorKey-5x5.png"),
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
    fig=fig,
    axs=axs,
    x_range=x_range
)

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-DoorKey-5x5-v0_rw_ner\dumps'
prefix = 'door-key-5x5-kg-rw-ner-23-01-2023 15-49-08'
label = 'rw-ner'
fig, axs = plot_results(
    image_name=os.path.join(image_path,"DoorKey-5x5.png"),
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
    fig=fig,
    axs=axs,
    x_range=x_range
)

#%% MultiRoom N2 S4
results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-MultiRoom-N2-S4-v0\dumps'
prefix = 'multiroom-n2-24-01-2023 21-58-05'
label = 'plain'
key = 'seed'
x_range = (0, 300000)
delimiter = '\t'
fig, axs = plot_results(
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
    x_range=x_range,
)

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-MultiRoom-N2-S4-v0_knn\dumps'
prefix = 'multiroom-n2-kg-knn-24-01-2023 21-58-05'
label = 'knn'
fig, axs = plot_results(
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
    fig=fig,
    axs=axs,
    x_range=x_range
)

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-MultiRoom-N2-S4-v0_rw\dumps'
prefix = 'multiroom-n2-kg-rw-24-01-2023 21-58-05'
label = 'rw'
fig, axs = plot_results(
    image_name=os.path.join(image_path,"MultiRoom-s2.png"),
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
    fig=fig,
    axs=axs,
    x_range=x_range,
)

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-MultiRoom-N2-S4-v0_rw_ner\dumps'
prefix = 'multiroom-n2-kg-rw-ner-24-01-2023 21-58-05'
label = 'rw-ner'
fig, axs = plot_results(
    image_name=os.path.join(image_path,'MultiRoom-s2.png'),
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
    fig=fig,
    axs=axs,
    x_range=x_range,
)

#%% DoorKey 8x8
results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-DoorKey-8x8-v0\dumps'
prefix = 'door-key-8x8-08-01-2023 16-26-38'
label = 'plain'
key = 'seed'
x_range = (0, 300000)
delimiter = '\t'
fig, axs = plot_results(
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
    x_range=x_range
)

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-DoorKey-8x8-v0_knn\dumps'
prefix = 'door-key-8x8-kg-knn-08-01-2023 16-27-13'
label = 'knn'
fig, axs = plot_results(
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
    fig=fig,
    axs=axs,
    x_range=x_range
)

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-DoorKey-8x8-v0_rw\dumps'
prefix = 'door-key-8x8-kg-rw-08-01-2023 16-34-44'
label = 'rw'
fig, axs = plot_results(
    image_name=os.path.join(image_path,'DoorKey-8x8.png'),
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
    fig=fig,
    axs=axs,
    x_range=x_range
)

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-DoorKey-8x8-v0_rw_ner\dumps'
prefix = 'door-key-8x8-kg-rw-08-01-2023 16-34-44'
label = 'rw-ner'
fig, axs = plot_results(
    image_name=os.path.join(image_path,'DoorKey-8x8.png'),
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
    fig=fig,
    axs=axs,
    x_range=x_range
)
#%% Key Corridor S3R2

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-KeyCorridorS3R2-v0\dumps'
prefix = 'key-korridor-s3r2-15-01-2023 16-31-09'
label = 'plain'
key = 'seed'
delimiter = '\t'
fig, axs = plot_results(
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
)

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-KeyCorridorS3R2-v0_knn\dumps'
prefix = 'key-korridor-s3r2-kg-knn-15-01-2023 16-30-54'
label = 'knn'
fig, axs = plot_results(
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
    fig=fig,
    axs=axs,
)

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-KeyCorridorS3R2-v0_rw\dumps'
prefix = 'key-korridor-s3r2-kg-rw-15-01-2023 16-31-03'
label = 'rw'
fig, axs = plot_results(
    image_name=os.path.join(image_path, 'KeyKorridor-s3r2.png'),
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
    fig=fig,
    axs=axs,
)

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-KeyCorridorS3R2-v0_rw_ner\dumps'
label = 'rw-ner'
fig, axs = plot_results(
    image_name=os.path.join(image_path, 'KeyKorridor-s3r2.png'),
    results_dir=results_dir,
    key=key,
    label=label,
    fig=fig,
    axs=axs,
)
#%% KeyCorridor S4R3

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-KeyCorridorS4R3-v0\dumps'
label = 'plain'
key = 'seed'
delimiter = '\t'
fig, axs = plot_results(
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
)

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-KeyCorridorS4R3-v0_knn\dumps'
label = 'knn'
fig, axs = plot_results(
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
    fig=fig,
    axs=axs,
)

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-KeyCorridorS4R3-v0_rw\dumps'
label = 'rw'
fig, axs = plot_results(
    image_name=os.path.join(image_path,'KeyKorridor-s4r3.png'),
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
    fig=fig,
    axs=axs,
)

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-KeyCorridorS4R3-v0_rw_ner\dumps'
label = 'rw-ner'
fig, axs = plot_results(
    image_name=os.path.join(image_path,'KeyKorridor-s4r3.png'),
    results_dir=results_dir,
    key=key,
    label=label,
    fig=fig,
    axs=axs,
)
#%% Lava Crossing
results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-LavaCrossingS9N2-v0\dumps'
prefix = 'lava-crossing-s9n2-23-01-2023 14-11-31'
label = 'plain'
key = 'seed'
delimiter = '\t'
fig, axs = plot_results(
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
)

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-LavaCrossingS9N2-v0_knn\dumps'
prefix = 'lava-crossing-s9n2-kg-knn-23-01-2023 14-11-34'
label = 'knn'
fig, axs = plot_results(
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
    fig=fig,
    axs=axs,
)

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-LavaCrossingS9N2-v0_rw\dumps'
prefix = 'lava-crossing-s9n2-kg-rw-23-01-2023 14-11-13'
label = 'rw'
key = 'seed'
delimiter = '\t'
fig, axs = plot_results(
    image_name=os.path.join(image_path, 'LavaCrossing-s9n2.png'),
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
    fig=fig,
    axs=axs,
)

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-LavaCrossingS9N2-v0_rw_ner\dumps'
label = 'rw-ner'
key = 'seed'
delimiter = '\t'
fig, axs = plot_results(
    image_name=os.path.join(image_path, 'LavaCrossing-s9n2.png'),
    results_dir=results_dir,
    key=key,
    label=label,
    fig=fig,
    axs=axs,
)

#%% Obstructed Maze 1Dlhb
results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-ObstructedMaze-1Dlhb-v0\dumps'
prefix = 'obstructed-maze-1Dlhb-19-01-2023 15-03-25'
label = 'plain'
key = 'seed'
x_range = (0, 600000)
delimiter = '\t'
fig, axs = plot_results(
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
    x_range=x_range,
)

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-ObstructedMaze-1Dlhb-v0_knn\dumps'
prefix = 'obstructed-maze-1Dlhb-kg-knn-19-01-2023 14-18-19'
label = 'knn'
fig, axs = plot_results(
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
    fig=fig,
    axs=axs,
    x_range=x_range
)

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-ObstructedMaze-1Dlhb-v0_rw\dumps'
prefix = 'obstructed-maze-1Dlhb-kg-rw-19-01-2023 15-06-38'
label = 'rw'
fig, axs = plot_results(
    image_name=os.path.join(image_path, 'ObstructedMaze-1Dlhb.png'),
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
    fig=fig,
    axs=axs,
    x_range=x_range,
)

results_dir = f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-ObstructedMaze-1Dlhb-v0_rw_ner\dumps'
prefix = 'obstructed-maze-1Dlhb-kg-rw-ner-26-01-2023 14-02-11'
label = 'rw-ner'
fig, axs = plot_results(
    image_name=os.path.join(image_path, 'ObstructedMaze-1Dlhb.png'),
    results_dir=results_dir,
    prefix=None,
    key=key,
    label=label,
    fig=fig,
    axs=axs,
    x_range=x_range,
)

#%% relative timesteps:
run_result_dirs = {'plain': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-ObstructedMaze-1Dlhb-v0\dumps',
                   'knn': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-ObstructedMaze-1Dlhb-v0_knn\dumps',
                    'rw': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-ObstructedMaze-1Dlhb-v0_rw\dumps',
                   'rw-ner': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-ObstructedMaze-1Dlhb-v0_rw_ner\dumps',}
rel_obstructed = compute_relative_timesteps(run_result_dirs)
plot_rel_times(rel_obstructed, f'data/images\{eval_mode}/rel_obstructed-1Dlb.png')
#%%
run_result_dirs = {
    'plain': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-KeyCorridorS3R2-v0\dumps',
    'knn': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-KeyCorridorS3R2-v0_knn\dumps',
    'rw': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-KeyCorridorS3R2-v0_rw\dumps',
    'rw_ner': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-KeyCorridorS3R2-v0_rw_ner\dumps',
}
rel_keykorridor = compute_relative_timesteps(run_result_dirs)
plot_rel_times(rel_keykorridor, f'data/images\{eval_mode}/rel-keykorridors3r2.png')
#%%
run_result_dirs = {
    'plain': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-LavaCrossingS9N2-v0\dumps',
    'knn': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-LavaCrossingS9N2-v0_knn\dumps',
    'rw': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-LavaCrossingS9N2-v0_rw\dumps',
    'rw-ner': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-LavaCrossingS9N2-v0_rw_ner\dumps',
}
rel_lavacrossing = compute_relative_timesteps(run_result_dirs)
plot_rel_times(rel_lavacrossing, f'data/images\{eval_mode}/rel_lavacrossing.png')

#%%
plot_rel_times(average_relative_timesteps([rel_keykorridor, rel_lavacrossing, rel_obstructed]), save_path=f'data/images\{eval_mode}/rel_hard.png')
#%%
run_result_dirs = {'plain': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-LavaGapS5-v0\dumps',
                   'knn': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-LavaGapS5-v0_knn\dumps',
                   'rw': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-LavaGapS5-v0_rw\dumps',
                   'rw-ner': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-LavaGapS5-v0_rw_ner\dumps'}
rel_lavagap5 = compute_relative_timesteps(run_result_dirs)
plot_rel_times(rel_lavagap5, f'data/images\{eval_mode}/rel-lavagap5.png')

#%%
run_result_dirs = {'plain': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-DoorKey-5x5-v0\dumps',
                   'knn': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-DoorKey-5x5-v0_knn\dumps',
                   'rw': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-DoorKey-5x5-v0_rw\dumps',
                   'rw-ner': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-DoorKey-5x5-v0_rw_ner\dumps'}
rel_doorkey5x5 = compute_relative_timesteps(run_result_dirs)
plot_rel_times(rel_doorkey5x5,  f'data/images\{eval_mode}/rel-doorkey5x5.png')

#%%
run_result_dirs = {'plain': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-MultiRoom-N2-S4-v0\dumps',
                   'knn': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-MultiRoom-N2-S4-v0_knn\dumps',
                   'rw': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-MultiRoom-N2-S4-v0_rw\dumps',
                    'rw-ner': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-MultiRoom-N2-S4-v0_rw_ner\dumps'}
rel_multiroomn2 = compute_relative_timesteps(run_result_dirs)
plot_rel_times(rel_multiroomn2,  f'data/images\{eval_mode}/rel-multiroomn2.png')

#%%
plot_rel_times(average_relative_timesteps([rel_multiroomn2, rel_lavagap5]), save_path=f'data/images\{eval_mode}/rel_easy.png')

#%%
run_result_dirs = {'plain': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-LavaGapS7-v0\dumps',
                   'knn': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-LavaGapS7-v0_knn\dumps',
                   'rw': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-LavaGapS7-v0_rw\dumps',
                   'rw-ner': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-LavaGapS7-v0_rw_ner\dumps'}
rel_lavagap7 = compute_relative_timesteps(run_result_dirs)
plot_rel_times(rel_lavagap7,  f'data/images\{eval_mode}/rel-lavagap7.png')

#%%
run_result_dirs = {'plain': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-DoorKey-8x8-v0\dumps',
                   'knn': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-DoorKey-8x8-v0_knn\dumps',
                   'rw': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-DoorKey-8x8-v0_rw\dumps',
                   'rw-ner': f'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data\{eval_mode}\MiniGrid-DoorKey-8x8-v0_rw_ner\dumps',}
rel_doorkey8x8 = compute_relative_timesteps(run_result_dirs)
plot_rel_times(rel_doorkey8x8,  f'data/images\{eval_mode}/rel-doorkey8x8.png')

#%%
plot_rel_times(average_relative_timesteps([rel_doorkey8x8, rel_lavagap7]), save_path=f'data/images\{eval_mode}/rel_medium.png')

#%%
plot_rel_times(average_relative_timesteps([rel_doorkey8x8,rel_lavagap7,rel_doorkey5x5,rel_lavagap5,rel_lavacrossing,rel_multiroomn2,rel_obstructed,rel_keykorridor]), save_path=f'data/images\{eval_mode}/rel_all.png')

#%% Ablations dimensions
if eval_mode == "random_eval":
    d = "dims"
elif eval_mode =="deterministic":
    d = "dimensions"

#%% dims - keykorridors3r2
results_dir = r'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data'+"\\"+eval_mode+r'\MiniGrid-KeyCorridorS3R2-v0_'+d+r'\dumps'
prefix = 'key-korridor-s3r2-kge-dimensions-30-01-2023 13-40-22'

plot_results(
    results_dir=results_dir,
    prefix=None,
    key='seed',
    image_name=f'data/images\{eval_mode}/ablations-dimensions-keyKorridors3r2.png',
    smoothing=100,
    confidence_interval=False,
)

#%% dims - ObstructedMaze 1Dlhb
results_dir = r'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data'+"\\"+eval_mode+r'\MiniGrid-ObstructedMaze-1Dlhb-v0_'+d+r'\dumps'

plot_results(
    results_dir=results_dir,
    prefix=None,
    key='seed',
    image_name=f'data/images\{eval_mode}/ablations-dimensions-ObstructedMaze-1Dlhb.png',
    smoothing=100,
    confidence_interval=False,
)

#%% DoorKey 8x8
results_dir = r'G:\Meine Ablage\Computing Projekte\KGRL\kgrl\data'+"\\"+eval_mode+r'\MiniGrid-DoorKey-8x8-v0_'+d+r'\dumps'

plot_results(
    results_dir=results_dir,
    prefix=None,
    key='seed',
    image_name=f'data/images\{eval_mode}/ablations-dimensions-DoorKey-8x8.png',
    smoothing=100,
    confidence_interval=False
)