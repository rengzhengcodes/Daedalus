from . import Optimizer, OrthoSpace, array
from typing import Callable, Generator

import numpy as np

class Midas(Optimizer):
    """Performs a modified MFA algorithm to optimize the architecture."""
    def __init__(self, space: OrthoSpace, loss: Callable) -> None:
        super().__init__(space, loss)
        self.dim = 0
        self._optimal: array = np.zeros(space.center().shape, dtype=np.float128)
        self._optimal[:] = np.nan
    
    def step(self) -> None:
        """
        Take a step in the search space from x
        
        Returns:
            A tuple representing the next point in the search space.
        """
        # Makes a copy of x to do the axial comparisons on.
        x: array = self.space.center()
        # Initializes the loss array.
        loss: list[int] = {}
        for point in range(*self.space._bounds[self.dim]):
            # Calculates the new point on that dimension.
            x[self.dim] = point
            # Calculates the loss of the new point.
            loss[point] = self.loss(tuple(x))
        # Calculates the dim with the minimal loss
        min_loss: int = min(loss, key=loss.get)
        self._optimal[self.dim] = min_loss
        self.dim += 1
    
    @property
    def optimal(self) -> array:
        ret: array = self._optimal.copy()
        ret = ret.astype(np.int64)
        return ret