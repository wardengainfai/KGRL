from typing import Optional

from cuco import config_parser

from .RevealState import RevealState


@config_parser(module_name = 'maze.state')
class StateMazeWrapperConfig:
    def __init__(
        self,
        reveal: RevealState = RevealState.NEIGHBORS,
        n_neighbors: int = 8,
        n_hop: int = 2,
        truncate: Optional[int] = None,
        render_visible: bool = True,
        verbose: bool = False
    ):
        self.reveal = reveal
        self.n_neighbors = n_neighbors
        self.n_hop = n_hop
        self.truncate = truncate if truncate is not None else n_neighbors
        self.render_visible = render_visible
        self.verbose = verbose

    @property
    def reveal_neighbors(self):
        return self.reveal == RevealState.NEIGHBORS

    @property
    def reveal_n_hop_neighbors(self):
        return self.reveal == RevealState.N_HOP

    @staticmethod
    def load(
        reveal: str,
        n_neighbors: int = 8,
        n_hop: int = 2,
        truncate: Optional[int] = None,
        render_visible: bool = True,
        verbose: bool = False
    ):
        return StateMazeWrapperConfig(
            reveal = RevealState(reveal),
            n_neighbors = n_neighbors,
            n_hop = n_hop,
            truncate = truncate,
            render_visible = render_visible,
            verbose = verbose
        )

    def dump(self) -> dict:
        raise NotImplementedError('Coming soon')
