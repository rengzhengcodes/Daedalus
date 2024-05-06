import os

import numpy as np

from ..architectures import Architecture
from ..optimizers.sgd import SGD

file_path = os.path.abspath(__file__)
ex_path = os.path.join(os.path.dirname(file_path))
ex_path = os.path.abspath(ex_path)

def example():
    """Perform optimization on the Eyeriss architecture."""
    # Set up the search space
    dimensions = ("global_buffer_size_scale", "pe_scale")
    bounds = ((-1, 8), (-1, 8))
    spec = os.path.join(ex_path, "top.yaml.jinja")
    arch = Architecture(dimensions, bounds, spec)
    
    # Set up the optimizer
    sgd = SGD(arch._orthospace, lambda x: arch.evaluate(x, brief_print=True)[-1])
    
    # Perform the optimization
    print(f"Initial point: {(prev_step := sgd._x.copy())}")
    for i in range(10):
        sgd.step()
        print(f"Step {i}: {sgd._x}")
        print(f"Loss: {sgd.loss(tuple(sgd._x))}")
        print()
        if np.array_equal(sgd._x, prev_step):
            break
        prev_step = sgd._x.copy()
    print(f"Done. Final point: {sgd._x}, Previous: {prev_step}")

example()