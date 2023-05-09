from typing import Tuple
from dataclasses import dataclass
from typing import Optional

from cuco import config_parser

from ..base.kg.KGUseCaseWrapperConfig import KGUseCaseWrapperConfig


@config_parser(object_type='kg-minigrid-wrapper-config', module_name='minigrid.kg')
@dataclass
class KGMinigridWrapperConfig(KGUseCaseWrapperConfig):

    label = 'Minigrid'

    original_observation_keys: Tuple[str] = ("state", "image", "direction")
    dynamic_kg: bool = False

    transform_mission: bool = False
    n_triples: int = 5
    er_padding: Optional['str'] = None

    @classmethod
    def load(
        cls,
        original_observation_keys: Tuple[str] = None,
        dynamic_kg: bool = False,
        transform_mission: bool = False,
        n_triples: int = 5,
        er_padding: Optional['str'] = None,
        **kwargs,
    ):
        if original_observation_keys is None:
            original_observation_keys = ["state", "image", "direction"]

        return cls(
            original_observation_keys=tuple(original_observation_keys),
            dynamic_kg=dynamic_kg,
            transform_mission=transform_mission,
            n_triples=n_triples,
            er_padding=er_padding,
            **super(cls, cls).load(**kwargs).__dict__
        )

    @property
    def compute_embeddings(self):
        return super().compute_embeddings or self.transform_mission
