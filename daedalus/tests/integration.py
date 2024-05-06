from ..architectures import Architecture
from ..optimizers.sgd import SGD

import os
file_path = os.path.abspath(__file__)
ex_path = os.path.join(os.path.dirname(file_path))
ex_path = os.path.abspath(ex_path)

def example():
    """Perform optimization on the Eyeriss architecture."""
    # Set up the search space
    dimensions = ("global_buffer_size_scale", "pe_scale")
    bounds = ((-1, 1), (-1, 1))
    spec = os.path.join(ex_path, "top.yaml.jinja")
    arch = Architecture(dimensions, bounds, spec)
    
    # Set up the optimizer
    sgd = SGD(arch._orthospace, lambda x: arch.evaluate(x, brief_print=True)[-1])
    
    # Perform the optimization
    for i in range(100):
        sgd.step()
        print(f"Step {i}: {sgd.x}")
        print(f"Loss: {sgd.loss(tuple(sgd._x))}")
        print()

example()