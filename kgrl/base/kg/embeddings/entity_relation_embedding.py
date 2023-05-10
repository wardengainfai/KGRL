"""
embeddinGs with the pykeen "ERmodel" and "EntityRelationEmbeddingModel" which
represent the graph in terms of thier entity representations and relation
representations.
"""
import os
import pickle
from typing import Tuple, Optional, Any, Mapping

from rdflib import Graph

from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline, PipelineResult

from ....dto.embedding import KGEmbeddingModel
from .triples import triples_factory_from_graph


def embed_graph(
        graph: Graph,
        tf: TriplesFactory = None,
        model: str = "TransE",
        epochs: int = 5,
        split: Tuple[float, float, float] = (.8, .1, .1),
        save_dir: str = None,
        model_kwargs: [Optional[Mapping[str, Any]]] = None,
        training_kwargs: Optional[Mapping[str, Any]] = None,
        evaluation_kwargs: Optional[Mapping[str, Any]] = None,
        compute_embedding: bool = True,
) -> KGEmbeddingModel:
    """
    Train the KGE model `model` on the given Graph `g`.
    :param graph:
    :param tf: `TriplesFactory` to use for the embedding
    :param model:
    :param epochs:
    :param split:
    :param save_dir:
    :param model_kwargs: Optional Model Arguments
    :param training_kwargs: Optional Arguments for training
    :return: `PipelineResult`
    # todo: What happens with src with multiple representations?
    """
    if (model_kwargs is None) and model == "TransE":
        model_kwargs = {'embedding_dim': 10}
    if tf is None:
        tf = triples_factory_from_graph(graph)
    train, test, val = tf.split(ratios=split)
    if compute_embedding:
        result = pipeline(
            training=train,
            testing=test,
            validation=val,
            model=model,
            epochs=epochs,
            model_kwargs=model_kwargs,
            training_kwargs=training_kwargs,
            evaluation_kwargs=evaluation_kwargs
        )

        if save_dir is not None:
            try:
                result.save_to_directory(save_dir)
            except Exception as ex:
                print(ex)
            with open(os.path.join(save_dir, "tf.pkl"), "wb") as file:
                pickle.dump(tf, file, pickle.HIGHEST_PROTOCOL)
        return KGEmbeddingModel(
            triples_factory=tf,
            pipeline_results=result,
        )
    else:
        return KGEmbeddingModel(
            triples_factory=tf,
            pipeline_results=None
        )


def load_model(save_dir: str, checkpoint_name: str) -> KGEmbeddingModel:
    """
    see also https://pykeen.readthedocs.io/en/stable/tutorial/checkpoints.html#loading-models-manually
    for a method to load and align the triples factory to a trained model.
    # TODO: how can results be loaded from a saved point? This happens automatically in pykeen but might be usefull more specifically here.
    """
    print(" loading {} from directory {}".format(checkpoint_name, save_dir))
    with open(os.path.join(save_dir, "tf.pkl"), "rb") as file:
        tf = pickle.load(file)
    checkpoint = torch.load(os.path.join(save_dir, checkpoint_name))

    pass
