from abc import ABC, abstractmethod
from typing import Callable, Generator

import numpy as np
array = np.ndarray

class Space(ABC):
    """Abstract base class for discrete-space search spaces."""
    def __init__(self, dimensions: array[str], bounds: array[int, int]) -> None:
        """
        Initialize the search space with the dimensions of the space. Assumes
        all dimensions can be orthogonalized (i.e., no two dimensions are collinear).
        
        Args:
            @param dimensions: A iterable of strings representing the names of
            the dimensions in the search space.
            @param bounds: A iterable of tuples representing the lower and upper
            bounds of each dimension in the search space. Corresponds to the
            order of the dimensions iterable.
        """
        assert len(dimensions) == len(bounds), "Dimensions and bounds must have the same length."
        self._dimensions: tuple = tuple(dimensions)
        self._bounds: tuple = tuple(bounds)
    
    @abstractmethod
    def contains(self, x: array[int]) -> bool:
        """Check if the point x is in the search space."""
        for i, (lower, upper) in enumerate(self._bounds):
            if not lower <= x[i] < upper:
                return False
        return True

    @abstractmethod
    def center(self) -> tuple:
        """Return the center of the search space."""
        return np.array((lower + upper) // 2 for lower, upper in self._bounds)
    
    @abstractmethod
    def in_dim(self, i: float, x: float) -> bool:
        """Check if x is in the bounds of the ith dimension of the search space."""
        return self._bounds[i][0] <= x < self._bounds[i][1]

class OrthoSpace(Space):
    """A Space in which all input dimensions are assumed to be orthogonal."""
    def __init__(self, dimensions: array[str], bounds: array[int, int]) -> None:
        super().__init__(dimensions, bounds)

    def adj(self, x: array) -> Generator[array[tuple]]:
        """
        Return the adjacent points to x in the search space.
        
        Args:
            @param x: A tuple representing a point in the search space.
        """
        for i, (lower, upper) in enumerate(self._bounds):
            adj_points = []
            if x[i] > lower:
                adj = list(x)
                adj[i] -= 1
                adj_points.append(np.array(adj))
            else:
                adj_points.append(x)

            if x[i] < upper - 1:
                adj = list(x)
                adj[i] += 1
                adj_points.append(np.array(adj))
            else:
                adj_points.append(x)

            yield tuple(adj_points)

class Optimizer(ABC):
    """Abstract base class for discrete-space optimizers."""
    def __init__(self, space: DiscreteOrthoSpace, loss: Callable) -> None:
        """
        Initialize the optimizer with the space it will search over and the
        loss function it will use to evaluate points in that space.
        
        Args:
            @param space: The search space, represented as a type which is a
            subclass of the Space class, representing the class of all points
            in the search space.
            @param loss: A function of the form f(x) -> float, where x is in
            the search space.
        """
        self.space: Space = space
        self.loss: Callable = loss

    @abstractmethod
    def step(self) -> tuple:
        """Take a step in the optimization process and returns the next point
        to move to."""
        pass
