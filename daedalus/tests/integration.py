import os

import numpy as np

from ..architectures import Architecture
from ..optimizers import sgd, mfa

file_path = os.path.abspath(__file__)
ex_path = os.path.join(os.path.dirname(file_path))
ex_path = os.path.abspath(ex_path)

def test_sgd():
    """Perform optimization on the Eyeriss architecture."""
    # Set up the search space
    dimensions = ("global_buffer_size_scale", "pe_scale")
    bounds = ((-1, 8), (-1, 8))
    spec = os.path.join(ex_path, "top.yaml.jinja")
    arch = Architecture(dimensions, bounds, spec)
    
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

# test_sgd()

def test_mfa():
    """Perform optimization on the Eyeriss architecture."""
    # Set up the search space
    dimensions = ("global_buffer_size_scale", "pe_scale")
    bounds = ((0, 5), (0, 5))
    spec = os.path.join(ex_path, "top.yaml.jinja")
    arch = Architecture(dimensions, bounds, spec)
    
    # Set up the optimizer
    optim = mfa.MFA(arch._orthospace, lambda x: arch.evaluate(x, brief_print=True)[-1])
    
    # Perform the optimization
    for i in range(len(dimensions)):
        print(f"Starting step {i}")
        optim.step()
        print(f"Step {i}: {optim.optimal}")

    print(f"Done. Final point: {optim.optimal}")

test_mfa()