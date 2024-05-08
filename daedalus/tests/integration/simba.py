import os
import time
import math
import numpy as np

from . import test_sgd, test_midas, test_grid
from ...architectures.simba import Simba

file_path = os.path.abspath(__file__)
ex_path = os.path.join(os.path.dirname(file_path))
ex_path = os.path.abspath(ex_path)


# Set up the search space
dimensions = ("global_buffer_size_scale", "pe_scale")
bounds = ((-1, 8), (-1, 8))
spec = os.path.join(ex_path, "cases", "simba_top.yaml.jinja")


if __name__ == "__main__":
    for problem in [f"VGG02_layer{i}.yaml" for i in range(1, 14)]:
        arch = Simba(dimensions, bounds, spec)
        print(f"====Running problem {problem}====")
        for test, tfunc in [
            ("SGD", test_sgd),
            ("Midas", test_midas),
            ("Grid", test_grid),
        ]:
            print(f"Running {test}")

            t_start = time.time()
            total_evals = tfunc(problem, arch)
            t_end = time.time()

            print(f"Ran {total_evals} evals in {t_end - t_start:.1f}s", end="\n\n")
