import torch
import numpy as np

data = torch.load(
    "src/adaptive_gnn_vqe/data/test_spinchain_dataset.pt"
)

all_signs = []

for d in data:
    g = d.g_signed.numpy()
    all_signs.extend(np.sign(g))

all_signs = np.array(all_signs)

npos = np.sum(all_signs > 0)
nneg = np.sum(all_signs < 0)
nzero = np.sum(all_signs == 0)

print("Sign statistics")
print("----------------")
print("positive:", npos)
print("negative:", nneg)
print("zero:", nzero)
print()
print("positive fraction =", npos / (npos + nneg))
print("negative fraction =", nneg / (npos + nneg))