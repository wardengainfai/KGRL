import os
import logging

import warnings


def disable_redundant_logging():
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" 

    logging.getLogger('pykeen.utils').setLevel(logging.ERROR)
    logging.getLogger('pykeen.pipeline.api').setLevel(logging.ERROR)
    logging.getLogger('pykeen.training.training_loop').setLevel(logging.ERROR)
    logging.getLogger('pykeen.stoppers.stopper').setLevel(logging.ERROR)
    logging.getLogger('pykeen.evaluation.evaluator').setLevel(logging.ERROR)
    logging.getLogger('pykeen.triples.triples_factory').setLevel(logging.ERROR)
    logging.getLogger('pykeen.triples.splitting').setLevel(logging.ERROR)
    logging.getLogger('stable_baselines3.common.logger').setLevel(logging.ERROR)

    warnings.filterwarnings("ignore")

    def warn(*args, **kwargs):
        pass

    warnings.warn = warn  # Disable deprecation warnings from gym


def enable_verbose_logging():
    logging.getLogger('pykeen.utils').setLevel(logging.INFO)
    logging.getLogger('pykeen.pipeline.api').setLevel(logging.INFO)
    logging.getLogger('pykeen.training.training_loop').setLevel(logging.INFO)
    logging.getLogger('pykeen.stoppers.stopper').setLevel(logging.INFO)
    logging.getLogger('pykeen.evaluation.evaluator').setLevel(logging.INFO)
    logging.getLogger('pykeen.triples.triples_factory').setLevel(logging.INFO)
    logging.getLogger('pykeen.triples.splitting').setLevel(logging.INFO)
    logging.getLogger('stable_baselines3.common.logger').setLevel(logging.INFO)

    warnings.resetwarnings()
