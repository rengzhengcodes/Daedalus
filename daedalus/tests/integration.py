from ..architectures import Architecture
from ..optimizers.sgd import SGD

import os
file_path = os.path.abspath(__file__)
lib_path = os.path.join(os.path.dirname(file_path), "..", "lib")
print(lib_path)

def run():
    pass