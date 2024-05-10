from joblib import Parallel, delayed

from . import OrthoSpace, array
from .sgd import SGD
from .midas import Midas
from typing import Callable

## @note Multiple inheritance shenanigans here: 
## https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way
class Tantalus(SGD, Midas):
    """Performs Midas optimization to seed the starting point of SGD."""
    def __init__(self, space: OrthoSpace, loss: Callable) -> None:
        super().__init__(space, loss)
        for _ in range(len(space._dimensions)):
            Midas.step(self)
        self._x = self.optimal
    
    def step(self) -> int:
        """Perform SGD steps on the current point."""
        return SGD.step(self)
