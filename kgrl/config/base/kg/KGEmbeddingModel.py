from enum import Enum


class KGEmbeddingModel(Enum):
    TRANSE = 'TransE'
    TRANSH = 'TransH'
    ROTATE = 'RotatE'
    RGCN = 'RGCN'
    COMPLEX = 'ComplEx'
    CONVE = 'ConvE'

    #add model
