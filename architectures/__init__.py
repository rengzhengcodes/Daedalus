from ..optimizers import OrthoSpace, array
import os
import timeloopfe.v4 as tl
brief_point = False

class Architecture:
    def __init__(self, dimensions: tuple, bounds: tuple, spec: str) -> None:
        assert len(dimensions) == len(bounds), "Dimensions and bounds must have the same length."
        self._dimensions: tuple = tuple(dimensions)
        self._bounds: tuple = tuple(bounds)
        self._orthospace: OrthoSpace = OrthoSpace(self._dimensions, self._bounds)
        self._spec: str = spec

    def evaluate(self, x: array, brief_print: bool=False) -> float:
        """Evaluate the architecture at point x."""
        assert self._orthospace.contains(x), "Point x is not in the search space."
        # Converts the array into a dictionary.
        x_dict = {self._dimensions[i]: x[i] for i in range(len(x))}

       