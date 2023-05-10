"""
todo: remove
"""
import os
import pickle
from typing import Tuple, Optional, Any, Mapping

import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

import csv
from rdflib import Graph, URIRef

from pykeen.pipeline import pipeline, PipelineResult
from pykeen.triples import TriplesFactory
from pykeen.models import Model


def triples_factory_from_graph(graph: Graph) -> TriplesFactory:
    triples = np.array(list(graph.triples((None, None, None))))
    return TriplesFactory.from_labeled_triples(triples)


def load_model_and_tf(save_dir: str, checkpoint_name: str) -> Tuple[Model, TriplesFactory]:
    """
    see also https://pykeen.readthedocs.io/en/stable/tutorial/checkpoints.html#loading-models-manually
    for a method to load and align the triples factory to a trained model.
    # TODO: how can results be loaded from a saved point? This happens automatically in pykeen
    """
    print(" loading {} from directory {}".format(checkpoint_name, save_dir))
    with open(os.path.join(save_dir, "tf.pkl"), "rb") as file:
        tf = pickle.load(file)
    checkpoint = torch.load(os.path.join(save_dir, checkpoint_name))

    pass


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
        force_overwrite_checkpoint = False
) -> PipelineResult:
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
    # print(training_kwargs)
    # if force_overwrite_checkpoint and 'checkpoint_directory' in training_kwargs and 'checkpoint_name' in training_kwargs:
    #     checkpoint_file_path = os.path.join(training_kwargs['checkpoint_directory'], training_kwargs['checkpoint_name'])
    #     print(checkpoint_file_path)
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
    return result, tf

def get_subgraph_representations(
        model: Model,
        tf: TriplesFactory,
        subgraph: Graph,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get embedding of a subgraph of the graph used to train `model` with `tf`.
    :param model:
    :param tf:
    :param subgraph:
    :return: A tuple `(entity_embeddings, relation_embeddings)` containing the
     embedding vectors of the entities and the embeddings vectros of the
     relations
    """
    entity_representation_modules = model.entity_representations  # List['pykeen.nn.RepresentationModule']
    relation_representation_modules = model.relation_representations  # List['pykeen.nn.RepresentationModule']
    if len(entity_representation_modules) > 1 or len(relation_representation_modules) > 1:
        raise NotImplementedError("Not implemented for Embedding Models with multiple representation")

    entity_embeddings = entity_representation_modules[0]  # pykeen.nn.Embedding
    relation_embeddings = relation_representation_modules[0]  # pykeen.nn.Embedding

    entities_in_subgraph = [str(node) for node in subgraph.all_nodes()]
    relations_in_subgraph = {str(predicate) for predicate in subgraph.predicates(None, None)}
    entity_indices = tf.entities_to_ids(entities_in_subgraph)
    relation_indices = tf.relations_to_ids(relations_in_subgraph)
    return (
        entity_embeddings(torch.tensor(entity_indices)).detach().cpu().numpy(),
        relation_embeddings(torch.tensor(relation_indices)).detach().cpu().numpy(),
    )


def get_k_nn(k: int, model: Model, return_representations: bool = True) -> np.ndarray:
    """
    Get the representation of the `k` nearest neighbors of the nodes in
    embedding `model`
    :param k: the number of nearest neighbors including the root node
    :param model: Optional Model Arguments
    :param return_representations: If `False` return indices instead of the representations
    :return: An array of representations of shape: (n,k,d) with n being the number of
      entities in the model d the emdedding dimension.

    todo: What about the relations? in principle the model should be able to
      learn from the entity embeddings alone (by exploration)
    """
    entity_representations = model.entity_representations[0]().detach().cpu().numpy()
    d = entity_representations.shape[1]
    nbrs = NearestNeighbors(n_neighbors=k).fit(entity_representations)
    _, nbrs_indices = nbrs.kneighbors(entity_representations)
    if return_representations:
        return np.vectorize(lambda ind: entity_representations[ind], signature="()->({})".format(d))(nbrs_indices)
    else:
        return nbrs_indices


def get_k_hop_subgraph(
        graph: Graph,
        node: URIRef,
        k: int,
        directional: bool = True
) -> Graph:
    """
    Return the k-hop subgraph of graph starting from node.

    :param graph: Graph from which the subgraph is constructed.
    :param node: starting node.
    :param k: number of hops from the starting node.
    :param directional: If set to `True` only outbound connections from a node
      are considered
    :return: Graph.
    """
    visited = []
    current = {node}
    triples = set()
    for i in range(k):
        while current:
            s = current.pop()
            visited.append(s)
            triples.update(graph.triples((s, None, None)))
            if not directional:
                triples.update(graph.triples((None, None, s)))

        current = {o for s, p, o in triples}.union({s for s, p, o in triples})  # don't have to distinguish directional here since if directional than only visited entities get added in the union
        current.difference_update(visited)  # subtract the entities that have been visited
    sub_graph = Graph()
    for triple in triples:
        sub_graph.add(triple)
    return sub_graph


class KG_Utils:
    def __init__(self):
        self.path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))


    def nt2tsv(self, path_to_graph, path_train, path_test):
        """
        :param: path_to_graph: path to the rdf graph in .nt format
        :param path_train: path to the output .tsv file with train set
        :param path_test: path to the output .tsv file with test set
        """
        with open(os.path.join(self.path, path_to_graph), encoding='utf-8') as datafile, \
                open(os.path.join(self.path, path_train), 'w', encoding='utf-8') as train_file, \
                open(os.path.join(self.path, path_test), 'w', encoding='utf-8') as test_file:

            train_writer = csv.writer(train_file, delimiter='\t')
            test_writer = csv.writer(test_file, delimiter='\t')

            dataset = []
            for line in datafile:
                items = line.split(' ')
                row = [x[1:-1] for x in items[0:-1]]  # clean triples
                dataset.append(row)

            # write train set
            for row in dataset[0: int(0.7 * len(dataset))]:
                train_writer.writerow(row)
            # write test set
            for row in dataset[int(0.7 * len(dataset)):]:
                test_writer.writerow(row)

    def train_embedding(
            self,
            dataset_name,
            path_train=None,
            path_test=None,
            model_name="TransE",
            epochs=5,
    ):
        """
        PyKEEN input data format: https://pykeen.readthedocs.io/en/stable/byo/data.html
        :param path_train: path to the input .tsv file with train set
        :param path_test: path to the input .tsv file with test set
        :param model_name: i.e. 'TransE'
        :param epochs: number of epochs
        :param dataset_name: used for saving the model
        """
        result = pipeline(
            training=os.path.join(self.path, path_train),
            testing=os.path.join(self.path, path_test),
            model=model_name,
            epochs=epochs,  # short epochs for testing - you should go higher
        )

        name = ''.join(['src/embeddings/', dataset_name.lower(), '-', model_name.lower(), '-num_ep', str(epochs)])
        result.save_to_directory(os.path.join(self.path, name))


if __name__ == "__main__":
    try:
        kg = KG_Utils()
        kg.nt2tsv(path_to_graph='data/graphs/domain_kg_nt.txt',
                  path_train='data/graphs/maze_train.txt',
                  path_test='data/graphs/maze_test.txt')

        kg.train_embedding(path_train='data/graphs/maze_train.txt',
                           path_test='data/graphs/maze_test.txt',
                           model_name='TransE',
                           dataset_name='maze',
                           epochs=5)

    except Exception as ex:
        print(ex)
