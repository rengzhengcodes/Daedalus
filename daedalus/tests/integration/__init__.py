import math
import numpy as np

from ...optimizers import midas, sgd, grid

def test_sgd(problem, arch):
    """Perform optimization on the Eyeriss architecture."""

    # Set up the optimizer
    optim = sgd.SGD(
        arch._orthospace, lambda x: arch.evaluate(x, problem, brief_print=True)[-1]
    )

    # Perform the optimization
    print(f"Initial point: {(prev_step := optim.x)}")
    eval_total = 0
    for i in range(10):
        eval_total += optim.step()
        print(f"Step {i}: {optim.x} | Loss: {optim.loss(tuple(optim.x))}")
        print()
        if np.array_equal(optim.x, prev_step):
            break
        prev_step = optim.x
    print(f"DONE. Final point: {optim.x}, Previous: {prev_step}")
    return eval_total


def test_midas(problem, arch):
    """Perform optimization on the Eyeriss architecture."""
    # Set up the optimizer
    optim = midas.Midas(
        arch._orthospace, lambda x: arch.evaluate(x, problem, brief_print=True)[-1]
    )

    # Perform the optimization
    for i in range(len(arch._dimensions)):
        # print(f"Starting step {i}")
        optim.step()
        print(f"Step {i}: {optim.optimal}")

    print(
        f"DONE. Final point: {optim.optimal} | Loss: {optim.loss(tuple(optim.optimal))}"
    )
    return sum(len(range(*arch._orthospace._bounds[dim_idx])) for dim_idx, dim in enumerate(arch._dimensions))


def test_grid(problem, arch):
    """Perform optimization on the Eyeriss architecture."""
    # Set up the optimizer
    optim = grid.Grid(
        arch._orthospace, lambda x: arch.evaluate(x, problem, brief_print=True)[-1]
    )

    # Perform the optimization
    total_iters = math.prod(
        len(range(*arch._orthospace._bounds[dim_idx])) for dim_idx, dim in enumerate(arch._dimensions)
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
    return total_iters
