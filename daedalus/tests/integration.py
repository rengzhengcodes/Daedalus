import os
import time
import math
import numpy as np

from ..architectures.eyeriss import Eyeriss
from ..optimizers import midas, sgd, grid

file_path = os.path.abspath(__file__)
ex_path = os.path.join(os.path.dirname(file_path))
ex_path = os.path.abspath(ex_path)


# Set up the search space
dimensions = ("global_buffer_size_scale", "pe_scale")
bounds = ((-1, 8), (-1, 8))
spec = os.path.join(ex_path, "top.yaml.jinja")


def test_sgd(problem, arch):
    """Perform optimization on the Eyeriss architecture."""

    # Set up the optimizer
    optim = sgd.SGD(
        arch._orthospace, lambda x: arch.evaluate(x, problem, brief_print=True)[-1]
    )

    # Perform the optimization
    print(f"Initial point: {(prev_step := optim.x)}")
    for i in range(10):
        optim.step()
        print(f"Step {i}: {optim.x} | Loss: {optim.loss(tuple(optim.x))}")
        print()
        if np.array_equal(optim._x, prev_step):
            break
        prev_step = optim.x
    print(f"DONE. Final point: {optim.x}, Previous: {prev_step}")


def test_midas(problem, arch):
    """Perform optimization on the Eyeriss architecture."""
    # Set up the optimizer
    optim = midas.Midas(
        arch._orthospace, lambda x: arch.evaluate(x, problem, brief_print=True)[-1]
    )

    # Perform the optimization
    for i in range(len(dimensions)):
        # print(f"Starting step {i}")
        optim.step()
        print(f"Step {i}: {optim.optimal}")

    print(
        f"DONE. Final point: {optim.optimal} | Loss: {optim.loss(tuple(optim.optimal))}"
    )


def test_grid(problem, arch):
    """Perform optimization on the Eyeriss architecture."""
    # Set up the optimizer
    optim = grid.Grid(
        arch._orthospace, lambda x: arch.evaluate(x, problem, brief_print=True)[-1]
    )

    # Perform the optimization
    total_iters = math.prod(
        len(range(*bounds[dim_idx])) for dim_idx, dim in enumerate(dimensions)
    )

    print(f"Running {total_iters} steps!")
    for i in range(total_iters):
        print(f"Starting step {i}: {optim._space_idx}")
        optim.step()
        print(
            f"Step {i}: {optim._space_idx} with loss {optim.loss(tuple(optim._space_idx_last))}"
            f" | optimal: {optim.optimal} with loss: {optim._optimal}"
        )

    print(f"DONE. Final point: {optim.optimal} with loss {optim._optimal}")


if __name__ == "__main__":
    for problem in ["VGG02_layer1.yaml", "VGG02_layer2.yaml"]:
        arch = Eyeriss(dimensions, bounds, spec)
        print(f"====Running problem {problem}====")
        for test, tfunc in [
            ("SGD", test_sgd),
            ("Midas", test_midas),
            ("Grid", test_grid),
        ]:
            print(f"Running {test}")

            t_start = time.time()
            tfunc(problem, arch)
            t_end = time.time()

            print(f"Time taken: {t_end - t_start:.1f}s", end="\n\n")
