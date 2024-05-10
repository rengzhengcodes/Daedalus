from . import Optimizer, OrthoSpace, array, n_jobs
from typing import Callable

import numpy as np


class Grid(Optimizer):
    """Performs a grid search to optimize the architecture."""

    def __init__(self, space: OrthoSpace, loss: Callable) -> None:
        super().__init__(space, loss)
        self._space_idx_last = None
        self._space_idx = np.array(space._bounds)[:, 0]  # slice out lower bounds
        self._optimal_map = np.zeros(len(space._dimensions))
        self._optimal = np.nan

    def step(self) -> None:
        """
        Take a step in the search space from x

        Returns:
            A tuple representing the next point in the search space.
        """
        # Initializes the loss array.
        loss: list[int] = {}

        # Calculates the loss of the new point.
        new_loss = self.loss(tuple(self._space_idx))
        # Calculates the dim with the minimal loss

        if np.isnan(self._optimal) or new_loss < self._optimal:
            self._optimal = new_loss
            self._optimal_map = self._space_idx.copy()

        # increment the space index and wrap around if necessary
        self._space_idx_last = self._space_idx.copy()
        for i in range(len(self._space_idx)):
            self._space_idx[i] += 1
            if self._space_idx[i] >= self.space._bounds[i][1]:
                self._space_idx[i] = self.space._bounds[i][0]
            else:
                break

    @property
    def optimal(self) -> array:
        ret: array = self._optimal_map.copy()
        ret = ret.astype(np.int64)
        return ret
