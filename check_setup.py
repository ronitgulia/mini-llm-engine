import sys

import numpy
import pytest
import safetensors
import torch
import transformers

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("Numpy:", numpy.__version__)
print("Transformers:", transformers.__version__)
print("Safetensors:", safetensors.__version__)
print("Pytest:", pytest.__version__)
print("GPU available:", torch.cuda.is_available())

x=torch.tensor ([[1.0,2.0],[3.0,4.0]])
print("x @ x =")
print(x@x)