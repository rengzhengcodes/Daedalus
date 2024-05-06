from . import Optimizer, OrthoSpace, array
from typing import Callable, Generator

import numpy as np

class SGD(Optimizer):
    """Performs stochastic gradient descent."""
    def __init__(self, space: OrthoSpace, loss: Callable) -> None:
        super().__init__(space, loss)
        self._x: array = self.space.center()
    
    def step(self) -> None:
        """
        Take a step in the search space from x
        
        Returns:
            A tuple representing the next point in the search space.
        """
        gradient: array = np.zeros(self._x.shape, dtype=np.float64)
        step: array = np.zeros(self._x.shape, dtype=np.int64)
        x_loss: float = self.loss(tuple(self._x))

        for i, (lower, upper) in enumerate(self.space.adj(self._x)):
            # Checks the gradient of the loss function at the current point.
            if self.space.contains(lower) and (l_loss := self.loss(tuple(lower))) < x_loss:
                gradient[i] -= l_loss
            if self.space.contains(upper) and (u_loss := self.loss(tuple(upper))) < x_loss:
                gradient[i] += u_loss
            
            # Moves in the opposite direction of the gradient. If we are at a
            # local minimum, we will not move.
            print(l_loss, u_loss, x_loss)
            if x_loss >= l_loss or x_loss >= u_loss:
                step[i] = np.sign(gradient[i])
                if not self.space.in_dim(i, self._x[i] + step[i]):
                    step[i] = 0
            else:
                step[i] = 0

        print(gradient)
        print(step)
        self._x += step
    
    @property
    def x(self) -> array:
        return self._x