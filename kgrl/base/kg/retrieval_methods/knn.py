import numpy as np
from sklearn.neighbors import NearestNeighbors

from ....dto.embedding import SubgraphRepresentation, KGEmbeddingModel


def get_knn(
        k: int,
        model: KGEmbeddingModel,
) -> SubgraphRepresentation:
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
    entity_representations = model.pipeline_results.model.entity_representations[0]().detach().cpu().numpy()
    d = entity_representations.shape[1]
    nbrs = NearestNeighbors(n_neighbors=k).fit(entity_representations)
    _, nbrs_indices = nbrs.kneighbors(entity_representations)
    return SubgraphRepresentation(
        entity_embeddings=np.vectorize(lambda ind: entity_representations[ind], signature="()->({})".format(d))(nbrs_indices),
        entity_embedding_indices=nbrs_indices,
    )
