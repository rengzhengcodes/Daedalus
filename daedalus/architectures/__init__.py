from abc import ABC, abstractmethod
import functools
import numpy as np

from ..optimizers import OrthoSpace, array


class Architecture(ABC):
    def __init__(self, dimensions: tuple, bounds: tuple, spec: str) -> None:
        assert len(dimensions) == len(
            bounds
        ), "Dimensions and bounds must have the same length."
        self._dimensions: tuple = tuple(dimensions)
        self._bounds: tuple = tuple(bounds)
        self._orthospace: OrthoSpace = OrthoSpace(self._dimensions, self._bounds)
        self._spec: str = spec

    @functools.lru_cache
    def evaluate(self, x: tuple, brief_print: bool = False) -> float:
        """Evaluate the architecture at point x."""
        x = np.array(x)
        assert self._orthospace.contains(x), "Point x is not in the search space."
        # Converts the array into a dictionary.
        x_dict = {self._dimensions[i]: x[i] for i in range(len(x))}

        return self._evaluate(x_dict, brief_print=brief_print)

    @abstractmethod
    def _evaluate(self, x_dict: dict) -> float:
        pass
