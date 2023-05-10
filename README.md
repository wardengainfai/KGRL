# KGRL - Injecting Knowledge graphs into the Reinforcement Learning Pipeline 

This Repository contain the supplemental material corresponding to the submission with the title
"Knowledge Graph Injection for Reinforcement Learning".

# Additional Material

# Reproducing Experiments
## Installing
### with conda
To create the environment in conda use the followinng command
````
conda env create -f requirements.yml
````

and enter the environment by 
````
conda activate kgrl
````


## Run Experiments
### Hyperparameter tuning:
The following command can be used for hyperparameter tuning:
````shell
python -m kgrl optimize <path-to-config> -hls -mg -v <level-of-verbosity [0-3]> -p <number-of-processes>
````
To optimize Hyperparameters for the LavaGapS7 environment:
````shell
python -m kgrl optimize data/trials/minigrid/tuning/lava-gap7.yml -hls -mg -v 1 -p 2
````
configurations for hyperparameter tuning can be found at [here](data/trials/minigrid/tuning).

### training and evaluation
Run training for the baseline: 
````shell
bash run_plain.sh 
````

Similarly for the proposed approaches:
````shell
bash run_knn.sh
bash run_ner.sh
bash run_rw.sh
````

for a more finegrained control - e.g. running a trial with parameters from the best trial in the database 'optuna.db'
````commandline
````shell
python -m kgrl trial data/trials/minigrid/final_training/door-key-8x8.yml -hls -ci -mg --best-from-study MiniGrid-DoorKey-8x8-v0
````
Using the ``--best-from-study`` flag overrides parameters in the config that have been optimized (if the study exists in the optuna database)
### Ablations
.. can be performed using the bash scripts with a flag specifying the type of ablation to run (e.g. dimensions - for dimensions of the KGE model):
````shell
bash run_abations.sh dimensions
````

### Plotting Figures
Assuming the data of the training runs has been stored to ``data/deterministic/`` the figures from the paper can be reproduced with:
````shell
python -m scripts.minigrid.plotting_results
python -m scripts.minigrid.plot_rel_advantage
````


in this env we can run training (only on round/configuration) with 
````
python -m scripts.minigrid.run_experiment
````
with the ``--config_files`` the location to a config file containing essential
parameters of the training run can be specified.

````
python -m scripts.minigrid.run_experiment 
 --config_files "data/config/minigrid/experiment_example.json"
````

## Run evaluation

Model training and evaluation may be run via the `evaluate` command, which allows to quickly test agent's ability to successfully navigate in a given environment. For example, the command call may look like this:

```sh
python -m kgrl evaluate -a dqn -e 20 -me 1000 -ef 0.2 -kg
```

## Run trial

The trial is not completely supported yet. Currently the following command:

```sh
python -m kgrl dev trial [-hls] [-ci] [-lf]
```

can be used for reading experiments configuration from an external file located at `data/trials/maze/default.yml` and running experiments in a maze environment. Knowledge graph wrapper is not yet
supported.

In future, support for `minigrid` environment will be added as well as saving a `Config` object to an external file in `yaml` format. The evaluation pipeline consists of 3 steps:

1. Edit [trials config](data/trials/maze/default.yml)
1. Run trials using the following command:

```sh
python -m kgrl trial -hls -dt -ci -lf
```

The command contains following flags:

`-hls` - short for `--headless`, orders the program to not to visualize the environment and agent actions  
`-dt` - short for `--disable-tqdm`, orders the program to not to show progress bars  
`-ci` - short for `--with-confidence-interval`, says that the generated images must contain confidence intervals  
`-lf` - short for `--long-format`, allows to save result in so-called `long format`, in which every pair of values (episode length and episode cumulative reward) is placed on a separate line in resulting dataframe dump  

3. Check generated results in folder `data/dumps` and visualization in `data/images`. The following picture contains an example of visualizing results of experiments, the configuration for which is passed in the [default trial configuration file](data/trials/default.yml). The full log is also provided [for analyzing the values of metrics](data/dumps/experiment-results-example.tsv) and [for analyzing the knowledge graph embeddings model training losses](data/dumps/model-training-losses-example.tsv).

![experiment results visualization example](data/images/experiment-results-visualization-example.png)

### Example Commands
Run a trial with parameters from the best trial in the database 'optuna.db'
````commandline
python -m kgrl trial data/trials/minigrid/final_training/door-key-8x8.yml -hls -ci -mg --best-from-study MiniGrid-DoorKey-8x8-v0
````

