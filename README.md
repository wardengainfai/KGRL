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

