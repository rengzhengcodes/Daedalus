import os
import time

from . import test_sgd, test_midas, test_grid
from ...architectures.eyeriss import Eyeriss

# Sets up the file location for the tests.
file_path = os.path.abspath(__file__)
ex_path = os.path.join(os.path.dirname(file_path))
ex_path = os.path.abspath(ex_path)
spec = os.path.join(ex_path, "cases", "eyeriss_top.yaml.jinja")

# Set up the search space
dimensions = ("global_buffer_size_scale", "pe_scale")
bounds = ((-1, 8), (-1, 8))


if __name__ == "__main__":
    for problem in [f"VGG02_layer{i}.yaml" for i in range(1, 14)]:
        arch = Eyeriss(dimensions, bounds, spec)
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