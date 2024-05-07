import os

import math
import numpy as np

from ..architectures.eyeriss import Eyeriss
from ..optimizers import midas, sgd, grid

file_path = os.path.abspath(__file__)
ex_path = os.path.join(os.path.dirname(file_path))
ex_path = os.path.abspath(ex_path)


def test_sgd():
    """Perform optimization on the Eyeriss architecture."""
    # Set up the search space
    dimensions = ("global_buffer_size_scale", "pe_scale")
    bounds = ((-1, 8), (-1, 8))
    spec = os.path.join(ex_path, "top.yaml.jinja")
    arch = Eyeriss(dimensions, bounds, spec)

    # Set up the optimizer
    optim = sgd.SGD(arch._orthospace, lambda x: arch.evaluate(x, brief_print=True)[-1])

    # Perform the optimization
    print(f"Initial point: {(prev_step := optim.x)}")
    for i in range(10):
        optim.step()
        print(f"Step {i}: {optim.x}")
        print(f"Loss: {optim.loss(tuple(optim.x))}")
        print()
        if np.array_equal(optim._x, prev_step):
            break
        prev_step = optim.x
    print(f"Done. Final point: {optim.x}, Previous: {prev_step}")

def test_midas():
    """Perform optimization on the Eyeriss architecture."""
    # Set up the search space
    dimensions = ("global_buffer_size_scale", "pe_scale")
    bounds = ((-1, 8), (-1, 8))
    spec = os.path.join(ex_path, "top.yaml.jinja")
    arch = Eyeriss(dimensions, bounds, spec)

    # Set up the optimizer
    optim = midas.Midas(
        arch._orthospace, lambda x: arch.evaluate(x, brief_print=True)[-1]
    )

    # Perform the optimization
    for i in range(len(dimensions)):
        print(f"Starting step {i}")
        optim.step()
        print(f"Step {i}: {optim.optimal}")

    print(f"Done. Final point: {optim.optimal}")
    print(f"Loss: {optim.loss(tuple(optim.optimal))}")

def test_grid():
    """Perform optimization on the Eyeriss architecture."""
    # Set up the search space
    dimensions = ("global_buffer_size_scale", "pe_scale")
    bounds = ((0, 5), (0, 5))
    spec = os.path.join(ex_path, "top.yaml.jinja")
    arch = Eyeriss(dimensions, bounds, spec)

    # Set up the optimizer
    optim = grid.Grid(
        arch._orthospace, lambda x: arch.evaluate(x, brief_print=True)[-1]
    )

    # Perform the optimization
    total_iters = math.prod(
        len(range(*bounds[dim_idx])) for dim_idx, dim in enumerate(dimensions)
    )

    print(f"Running {total_iters} steps!")
    for i in range(total_iters):
        print(f"Starting step {i}: {optim._space_idx}")
        optim.step()
        print(f"Loss: {optim.loss(tuple(optim._space_idx_last))}")
        print(f"Step {i} optimal: {optim.optimal}")

    print(f"Done. Final point: {optim.optimal} with loss {optim._optimal}")


if __name__ == "__main__":
    test_sgd()
    test_midas()
    test_grid()