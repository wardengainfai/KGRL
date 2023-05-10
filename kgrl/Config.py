from dataclasses import dataclass

from cuco import config_parser, Config as BaseConfig


@config_parser()
@dataclass
class Config(BaseConfig):
    root: str

    data: str
    test_data: str

    kg_model_checkpoint: str
    checkpoint: str

    ontology: str
    maze_ontology: str
    minigrid_ontology: str

    name: str
