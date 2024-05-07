from joblib import Parallel, delayed
import numpy as np

from . import Optimizer, OrthoSpace, array
from typing import Callable, Tuple

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

        # Creates the parallelizable process.
        def eval_point(point: int) -> None:
            # Makes a copy of x to do the axial comparisons on.
            x: array = self.space.center()
            # Calculates the new point on that dimension.
            x[self.dim] = point
            # Calculates the loss of the new point.
            return point, self.loss(tuple(x))

        # Evaluates the loss of each point in the current dimension.
        results: list[Tuple[int, int]] = Parallel(n_jobs=8)(delayed(eval_point)(point) for point in range(*self.space._bounds[self.dim]))

        # Calculates the dim with the minimal loss
        min_loss: int = min(results, key=lambda result: result[1])
        self._optimal[self.dim] = min_loss[0]
        self.dim += 1
    
    @property
    def optimal(self) -> array:
        ret: array = self._optimal.copy()
        ret = ret.astype(np.int64)
        return ret