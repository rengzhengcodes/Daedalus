from joblib import Parallel, delayed

from . import Optimizer, OrthoSpace, array, n_jobs
from typing import Callable

import numpy as np

class SGD(Optimizer):
    """Performs stochastic gradient descent."""
    def __init__(self, space: OrthoSpace, loss: Callable) -> None:
        super().__init__(space, loss)
        self._x: array = self.space.center()
    
    def step(self) -> int:
        """
        Take a step in the search space from x
        
        Returns:
            The number of timeloop evaluations run.
        """
        step: array = np.zeros(self._x.shape, dtype=np.int64)
        x_loss: float = self.loss(tuple(self._x))

        # Evaluates the partial gradient of the loss function at the current point.
        def eval_dim_gradient(dim: int, lower: int, upper: int):
            # Checks the gradient of the loss function at the current point.
            gradient: float = 0
            l_loss, u_loss = None, None
            if self.space.contains(lower) and (l_loss := self.loss(tuple(lower))) < x_loss:
                gradient -= l_loss
            if self.space.contains(upper) and (u_loss := self.loss(tuple(upper))) < x_loss:
                gradient += u_loss

            eval_count = (l_loss is not None) + (u_loss is not None)
            return (
                dim, gradient, l_loss if l_loss is not None else np.inf, 
                               u_loss if u_loss is not None else np.inf, 
                eval_count
            )

        results = Parallel(n_jobs=n_jobs)(delayed(eval_dim_gradient)(dim, *bounds)
                                     for dim, bounds in enumerate(self.space.adj(self._x)))
        
        # Moves in the opposite direction of the gradient. If we are at a local minimum, we will not move.
        eval_total = 0
        for dim, gradient, l_loss, u_loss, eval_count in results:
            eval_total += eval_count
            # print(l_loss, u_loss, x_loss)
            if x_loss >= l_loss or x_loss >= u_loss:
                step[dim] = np.sign(gradient)
                # Don't follow gradient if it leads out of the search space.
                if not self.space.in_dim(dim, self._x[dim] + step[dim]):
                    step[dim] = 0
                # If the gradient is in this case, we know either side of the
                # cliff is lower, but both values are equal, so the gradient
                # is interpolated to be 0. To compensate, we choose to go in the
                # lower exponent direction as it needs less hardware support.
                elif step[dim] == 0 and l_loss == u_loss and l_loss != np.inf:
                    step[dim] = -1
                else:
                    step[dim] = 0
            else:
                step[dim] = 0
        self._x += step
        return eval_total
    
    @property
    def x(self) -> array:
        return self._x.copy()