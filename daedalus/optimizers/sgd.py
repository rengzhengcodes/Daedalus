from . import Optimizer, OrthoSpace, array
from typing import Callable, Generator

import numpy as np

class SGD(Optimizer):
    """Performs stochastic gradient descent."""
    def __init__(self, space: OrthoSpace, loss: Callable) -> None:
        super().__init__(space, loss)
        self.x: array = self.space.center()
    
    def step(self, x) -> None:
        """
        Take a step in the search space from x
        
        Returns:
            A tuple representing the next point in the search space.
        """
        gradient: array = np.zeros(self.x.shape)
        x_loss: float = self.loss(self.x)

        for i, (lower, upper) in enumerate(self.space.adj(x)):
            # Checks the gradient of the loss function at the current point.
            if (l_loss := self.loss(lower)) < x_loss:
                gradient[i] -= l_loss
            if (u_loss := self.loss(upper)) < x_loss:
                gradient[i] += u_loss
            
            # Moves in the opposite direction of the gradient. If we are at a
            # local minimum, we will not move.
            if x_loss >= l_loss or x_loss >= u_loss:
                gradient[i] = np.sign(gradient[i])
                if self.space.in_dim(i, self.x[i] - gradient[i]):
                    gradient[i] = 0
            else:
                gradient[i] = 0
        
        self.x += gradient