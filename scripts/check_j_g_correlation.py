import os
import torch
import numpy as np


def corr(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def main():
    data_path = os.path.abspath("src/adaptive_gnn_vqe/data/test_spinchain_dataset.pt")
    data_list = torch.load(data_path, map_location="cpu")

    by_step = {}

    for data in data_list:
        step = int(data.step)
        logJ = data.edge_attr[:, 0].numpy()
        g = data.g.numpy()

        by_step.setdefault(step, []).append(corr(logJ, g))

    print("Correlation between log(J) and ADAPT gradient g")
    print("------------------------------------------------")
    for step in sorted(by_step):
        vals = np.array(by_step[step])
        vals = vals[~np.isnan(vals)]
        print(
            f"step {step}: "
            f"mean corr={vals.mean():.4f}, "
            f"median corr={np.median(vals):.4f}"
        )


if __name__ == "__main__":
    main()