from typing import ClassVar

from cuco import config_parser

from ..base.kg.KGUseCaseWrapperConfig import KGUseCaseWrapperConfig


@config_parser(object_type = 'kg-maze-wrapper-config', module_name = 'maze.kg')
class KGMazeWrapperConfig(KGUseCaseWrapperConfig):
    label: ClassVar[str] = 'Maze'

    @classmethod
    def load(cls, **kwargs):
        return cls(
            **super(cls, cls).load(**kwargs).__dict__
        )
