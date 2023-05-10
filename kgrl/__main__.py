from datetime import datetime
from pathlib import Path
import os
from typing import Optional, Dict, Any, List, Tuple, Union

import click
import matplotlib
import atexit
import optuna
from joblib import Parallel, delayed

from click import group, option, Choice, argument
from cuco import make_configs

from kgrl.utils.logging import disable_redundant_logging, enable_verbose_logging
from kgrl.config import ExperimentConfig

disable_redundant_logging()  # Can't do it after all imports because some log messages are triggered by import statements

from scripts.minigrid.manual_control import ManualControlArgs, \
    main as run_minigrid_game  # key_handler, reset

from kgrl.use_cases.minigrid.experiments.experiment import MinigridExperimentConfig, MinigridExperimenter
from kgrl.use_cases.maze.experiments.experiment import MazeExperimentConfig, MazeExperimenter, KGMazeWrapperConfig
from kgrl.use_cases.maze.environments.gym_maze_wrapper import StateMazeWrapperConfig
from kgrl.base.experiments import EvaluationResults, FontPresets
from kgrl.config import RevealState, RevealGraph
from kgrl.utils.hyperparametertuing_utils import update_experiment_config
from kgrl.config.utils import force_new_parameters_for_config, group_configs_on_key

DATETIME_FORMAT = '%d-%m-%Y %H-%M-%S'
DEFAULT_TUNING_PATH = os.path.join('data', 'trials', 'tuning')


@group()
def main():
    pass


def post_process_config(config: ExperimentConfig, disable_tqdm: bool,
                        headless: bool):
    config.experiment_headless = headless

    use_tqdm = not disable_tqdm

    config.kg_wrapper_config.train.use_tqdm = use_tqdm
    config.kg_wrapper_config.eval.use_tqdm = use_tqdm


@main.command()
@argument('path', type=str, default='default')
@option('--headless', '-hls', is_flag=True)
@option('--minigrid', '-mg', is_flag=True)
@option('--verbose', '-v', is_flag=True)
@option('--disable-tqdm', '-dt', is_flag=True)
@option('--long-format', '-lf', is_flag=True)
@option('--with-confidence-interval', '-ci', is_flag=True)
@option('--plot-losses-in-separate-file', '-ls', is_flag=True)
@option('--suffix', '-s', type=str, default=None)
@option('--use-best', is_flag=True, help='look for the best parameters in the storage')
@option('--best-from-study', type=str, default=None)
@option('--opt-storage', type=str, default='sqlite:///optuna.db')
@option('--force-parameters', '-fp', type=click.UNPROCESSED, multiple=True, nargs=2, help='A list of key value pairs corresponding to parameters that should be enforced disregarding the corresponding value in the config. Take care of the types')
@option('--parallelize-key', multiple=False, type=str, default=None, help="parameter in the config for which the configs are split. All configs with the same value will get a process")
def trial(
        path: str,
        headless: bool,
        minigrid: bool,
        verbose: bool,
        disable_tqdm: bool,
        long_format: bool,
        with_confidence_interval: bool,
        plot_losses_in_separate_file: bool,
        suffix: Optional[str] = None,
        best_from_study: Optional[str] = None,
        use_best: bool = False,
        opt_storage: str = 'sqlite:///optuna.db',
        force_parameters: Optional[Tuple[str, Union[int, str]]] = None,
        parallelize_key: str = None,
):
    # process the forces parameters
    force_parameters = dict(force_parameters)
    for key, value in force_parameters.items():
        if value.isdigit():
            force_parameters[key] = int(value)
        elif value == 'True':
            force_parameters[key] = True
        elif value == "False":
            force_parameters[key] = False

    if path == 'default':
        path = f'data/trials/{"minigrid" if minigrid else "maze"}/default.yml'

    configs: List[ExperimentConfig] = make_configs(
        path=path, type_specification_root='data/trials/types',
        verbose=verbose, post_process_config=post_process_config,
        config_name_key='experiment_name', disable_tqdm=disable_tqdm,
        headless=headless
    )

    if verbose:
        print('Generated configs: ')
        print([config.experiment_name for config in configs])

    assert len(configs) > 1, 'Cannot run trial with just one config - use command "evaluate" instead'

    if parallelize_key is not None:
        # group configs and process the groups in parallel
        configs_dict = group_configs_on_key(configs=configs, key=parallelize_key)
        n_jobs = len(configs_dict.keys())
        r = Parallel(n_jobs=n_jobs)(delayed(
            process_configs)(
                configs=configs,
                path=path,
                minigrid=minigrid,
                verbose=verbose,
                long_format=long_format,
                with_confidence_interval=with_confidence_interval,
                plot_losses_in_separate_file=plot_losses_in_separate_file,
                suffix=suffix,
                best_from_study=best_from_study,
                use_best=use_best,
                opt_storage=opt_storage,
                force_parameters=force_parameters,
                additional_suffix=key,
            ) for key, configs in configs_dict.items())
    else:
        process_configs(
            configs,
            path=path,
            minigrid=minigrid,
            verbose=verbose,
            long_format=long_format,
            with_confidence_interval=with_confidence_interval,
            plot_losses_in_separate_file=plot_losses_in_separate_file,
            suffix=suffix,
            best_from_study=best_from_study,
            use_best=use_best,
            opt_storage=opt_storage,
            force_parameters=force_parameters,
        )



def process_configs(
        configs: List[ExperimentConfig],
        path: str,
        minigrid: bool,
        verbose: bool,
        long_format: bool,
        with_confidence_interval: bool,
        plot_losses_in_separate_file: bool,
        suffix: Optional[str] = None,
        best_from_study: Optional[str] = None,
        use_best: bool = False,
        opt_storage: str = 'sqlite:///optuna.db',
        force_parameters: Optional[Dict] = None,
        additional_suffix: Optional[str] = None,
):
    with FontPresets():

        results = EvaluationResults(
            label=f'{Path(path).stem}-{datetime.now().strftime(DATETIME_FORMAT)}',
            plot_losses_in_separate_file=plot_losses_in_separate_file
        )

        n_experiments = len(configs)
        i = 1

        def exit_cleanup(results, config):
            results.plot(label=config.experiment_name,
                         with_confidence_interval=with_confidence_interval)
            results.reset(config.experiment_name, long_format=long_format)
            if minigrid:
                results_path = f'data/{config.minigrid_env_name}'
            else:
                results_path = 'data/maze'
            results.save(
                folder=results_path + '/images',
                dumps_folder=results_path + '/dumps',
                as_single_dump=long_format,
                with_confidence_interval=with_confidence_interval
            )
            print(f'results saved to {results_path}')

        for config in configs:
            # update parameters based on the forced parameters dict
            config, unmapped_parameters = force_new_parameters_for_config(config=config, force_parameters=force_parameters.copy())
            if unmapped_parameters:
                print(f"the following parameters could not be mapped: f{unmapped_parameters}")
            if verbose:
                print(config)
            if best_from_study is not None or use_best:
                if use_best and best_from_study is None:
                    if minigrid:
                        best_from_study = config.minigrid_env_name
                    else:
                        pass  # todo: fix this in the case of other use cases
                try:
                    study = optuna.load_study(study_name=best_from_study, storage=opt_storage)
                    config = update_experiment_config(config, study.best_params, transform=True)
                    print(f'Using config updated with best params from {best_from_study} with:')
                    print(study.best_params)
                except KeyError as e:
                    print(f'Study {best_from_study} does not exist.')
                    print(f'Commencing Trial with default config from {path}.')

            atexit.register(exit_cleanup, results=results, config=config)
            print(f'Running {i} / {n_experiments} experiment ({config.experiment_name})')

            (MinigridExperimenter if minigrid else MazeExperimenter)(
                config).run_training(results=results, verbose=1 if verbose else 0)

            results.plot(label=(label := config.experiment_name),
                         with_confidence_interval=with_confidence_interval)

            results.reset(label, long_format=long_format)
            atexit.unregister(exit_cleanup)
            i += 1

    if minigrid:
        results_path = f'data/{config.minigrid_env_name}'
        if suffix is not None:
            results_path += f'_{suffix}'
        if additional_suffix is not None:
            results_path += f'/{additional_suffix}'
    else:
        results_path = 'data/maze'

    results.save(
        folder=results_path+'/images',
        dumps_folder=results_path+'/dumps',
        as_single_dump=long_format,
        with_confidence_interval=with_confidence_interval,
    )

    print(f'Results saved to {results_path}... Trial complete')


@main.command()
@option('--batch-size', '-b', type=int, default=128)
@option('--n-epochs', '-e', type=int, default=100)
@option('--learning-rate', '-lr', type=float, default=5e-5)
@option('--exploration-fraction', '-ef', type=float, default=0.1)
@option('--agent', '-a', type=Choice(('dqn',)), default='dqn')
@option('--n-max-episodes', '-me', type=int, default=2000)
@option('--enable-knowledge-graph', '-kg', is_flag=True)
@option('--minigrid', '-mg', is_flag=True)
@option('--trial', '-t', is_flag=True)
@option('--n-training-runs', '-nr', type=int, default=20)
@option('--enable-states', '-ss', is_flag=True)
@option('--reveal-neighbors', '-rn', is_flag=True)
@option('--reveal-n-hop-neighbors', '-rh', is_flag=True)
@option('--n-neighbors', '-nn', type=int, default=8)
@option('--n-hop', '-nh', type=int, default=2)
@option('--verbose', '-v', is_flag=True)
@option('--punishment-multiplier', '-pm', type=float, default=1.0)
@option('--library', '-l', type=Choice(('sb3', 'rll', 'tn')), default='sb3')
@option('--deterministic', '-d', is_flag=True)
@option('--evaluation-step-delay', '-esd', type=float, default=0.2)
@option('--k-nn-k', '-kn', type=int, default=8)
@option('--embedding-dim', '-ed', type=int, default=8)
@option('--reveal-knn-embedding', '-rk', is_flag=True)
@option('--reveal-subgraph', '-rs', is_flag=True)
@option('--include-walls', '-iw', is_flag=True)
@option('--disable-tqdm', '-dt', is_flag=True)
@option('--n-eval-episodes', '-nee', type=int, default=1)
@option('--max-eval-episode-length', '-meel', type=int, default=None)
@option('--eval-frequency', '-efr', type=int, default=8)
@option('--eval-start', '-es', type=int, default=16)
@option('--headless', '-hls', is_flag=True)
def evaluate(
        batch_size: int,
        n_epochs: int,
        learning_rate: float,
        exploration_fraction: float,
        agent: str,
        n_max_episodes: int,
        enable_knowledge_graph: bool,
        minigrid: bool,
        trial: bool,
        n_training_runs: int,
        enable_states: bool,
        reveal_neighbors: bool,
        reveal_n_hop_neighbors: bool,
        n_neighbors: int,
        n_hop: int,
        verbose: bool,
        punishment_multiplier: float,
        library: str,
        deterministic: bool,
        evaluation_step_delay: float,
        k_nn_k: int,
        embedding_dim: int,
        reveal_knn_embedding: bool,
        reveal_subgraph: bool,
        include_walls: bool,
        disable_tqdm: bool,
        n_eval_episodes: int,
        max_eval_episode_length: int,
        eval_frequency: int,
        eval_start: int,
        headless: bool
):
    if verbose:
        enable_verbose_logging()

    # 1. Configure experiment

    config = (MinigridExperimentConfig if minigrid else MazeExperimentConfig)(
        trainer_train_batch_size=batch_size,
        training_steps=n_epochs,
        training_lr=learning_rate,
        training_exploration_fraction=exploration_fraction,
        training_max_episodes=n_max_episodes,
        agent=agent,
        kg_wrapper_enabled=enable_knowledge_graph,
        states_wrapper_enabled=enable_states,
        states_wrapper_config=None if minigrid else StateMazeWrapperConfig(
            reveal=(
                RevealState.NEIGHBORS if reveal_neighbors and not reveal_n_hop_neighbors else
                RevealState.N_HOP if reveal_n_hop_neighbors and not reveal_neighbors else
                None
            ),
            verbose=verbose
        ),
        punishment_multiplier=punishment_multiplier,
        library='stable-baselines' if library == 'sb3' else 'rllib' if library == 'rll' else 'tune' if library == 'tn' else None,
        deterministic=deterministic,
        evaluation_step_delay=evaluation_step_delay,
        kg_wrapper_config=None if minigrid else KGMazeWrapperConfig(
            k_nn_k=k_nn_k,
            embedding_model_kwargs={'embedding_dim': embedding_dim},
            reveal=(
                RevealGraph.KNN_EMBEDDING if reveal_knn_embedding and not reveal_subgraph else
                RevealGraph.SUBGRAPH if reveal_subgraph and not reveal_knn_embedding else
                None
            ),
            include_walls=include_walls,
            use_tqdm=not disable_tqdm
        ),
        evaluation_n_episodes=n_eval_episodes,
        eval_freq=(eval_frequency, 'episode'),
        eval_start=eval_start,
        evaluation_max_episode_length=max_eval_episode_length,
        experiment_headless=headless
    )

    # 2. Make experimenter

    experimenter = (MinigridExperimenter if minigrid else MazeExperimenter)(config)

    # 3. Run training

    if trial:
        experimenter.run_trial(n_training_runs=n_training_runs)
    else:
        experimenter.run_training(verbose=1 if verbose else 0)


@main.command()
@option('--env', '-e', type=str, default='MiniGrid-MultiRoom-N6-v0')
@option('--seed', '-s', type=int, default=-1)
@option('--size', '-z', type=int, default=32)
@option('--agent-view', '-av', is_flag=True)
def play(env: str, seed: int, size: int, agent_view: bool):
    matplotlib.use('qtagg')
    run_minigrid_game(ManualControlArgs(env, seed, size, agent_view),
                      f'environment = {env}; seed = {seed}; size = {size}')


@main.command()
@argument('path', type=str, default='default')
@option('--search_space', '-ss', type=str, default='default')
@option('--log_path', type=str, default=None)
@option('--headless', '-hls', is_flag=True)
@option('--minigrid', '-mg', is_flag=True)
@option('--verbose', '-v', type=int, default=1)
@option('--n-processes', '-p', type=int, default=None)
@option('--study-name', '-n', type=str, default=None)
@option('--enqueue', type=bool, default=False)
@option('--force-parameters', '-fp', multiple=True, nargs=2, help='A list of key value pairs corresponding to parameters for a trial to enqueue (requires enqueue to be set to true)')
@option('--storage', '-s', type=str, default='sqlite:///optuna.db')
def optimize(
        path: str,
        search_space: str,
        headless: bool,
        verbose: int,
        minigrid: bool,
        log_path: Optional[str] = None,
        n_processes: Optional[int] = None,
        study_name: Optional[str] = None,
        enqueue: bool = False,
        force_parameters: Dict[str, Any] = None,
        storage: str = 'sqlite:///optuna.db',
):
    """
    Optimize the basis configuration in path over the parameters in the search
    space
    """
    if path == 'default':
        path = f'data/trials/{"minigrid" if minigrid else "maze"}/default.yml'
    if search_space == 'default':
        # todo: implement adjusting the search space according to values given in a json
        pass

    # make configuration
    configs = make_configs(
        path=path, type_specification_root='data/trials/types',
        verbose=verbose > 0, post_process_config=post_process_config,
        config_name_key='experiment_name', disable_tqdm=True,
        headless=headless
    )

    if verbose > 0:
        print('Generated configs: ')
        print([config.experiment_name for config in configs])
        if len(configs) > 1:
            print('More than one config generated. Augmenting the study name with ')

    # process force_parameters to dict
    f_params = {}
    for key, value in force_parameters:
        f_params[key] = float(value)

    for config in configs:
        experimenter = (MinigridExperimenter if minigrid else MazeExperimenter)(config)
        if len(configs) > 1:
            study_name_suffix = '_' + config.experiment_name
            if verbose > 0:
                print(study_name_suffix)
        else:
            study_name_suffix = ''
        if not enqueue:
            experimenter.hyperparameter_optimization(
                verbose=verbose,
                log_path=log_path,
                n_processes=n_processes,
                study_name=study_name,
                study_name_suffix=study_name_suffix,
                storage_url=storage,
            )
        else:
            assert force_parameters is not None, 'There are no specified parameters for a new trial. Add Parameters with the `--force-parameters` flag.'
            experimenter.enqueue_trial(
                force_parameters=f_params,
                verbose=verbose,
                study_name=study_name,
                study_name_suffix=study_name_suffix,
                storage=storage,
            )


if __name__ == '__main__':
    main()
