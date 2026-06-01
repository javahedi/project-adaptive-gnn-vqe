import os
import torch
import numpy as np


def main():
    data_path = os.path.abspath(
        "src/adaptive_gnn_vqe/data/test_spinchain_dataset.pt"
    )

    data_list = torch.load(data_path, map_location="cpu")

    print("Dataset integrity check")
    print("-----------------------")
    print("Total graph samples:", len(data_list))

    bad_label = 0
    missing_attrs = 0

    ranks_by_step = {}
    top1_by_step = {}

    for data in data_list:
        # Check required fields
        for attr in ["x", "edge_index", "edge_attr", "y", "g", "step", "alpha"]:
            if not hasattr(data, attr):
                missing_attrs += 1
                print("Missing attribute:", attr)

        gvals = data.g
        y = int(data.y.item())

        # Check label equals argmax(g)
        if y != int(torch.argmax(gvals).item()):
            bad_label += 1

        # Strongest-J rank of true label
        logJ = data.edge_attr[:, 0]
        order = torch.argsort(logJ, descending=True)

        rank = (order == y).nonzero(as_tuple=True)[0].item() + 1

        step = int(data.step)
        ranks_by_step.setdefault(step, []).append(rank)
        top1_by_step.setdefault(step, []).append(int(rank == 1))

    print("Missing attribute count:", missing_attrs)
    print("Bad labels y != argmax(g):", bad_label)

    print("\nStrongest-J rank by circuit step:")
    for step in sorted(ranks_by_step):
        ranks = np.array(ranks_by_step[step])
        top1 = np.mean(top1_by_step[step])
        print(
            f"step {step}: "
            f"top1={top1:.4f}, "
            f"mean_rank={ranks.mean():.2f}, "
            f"median_rank={np.median(ranks):.2f}, "
            f"min={ranks.min()}, "
            f"max={ranks.max()}"
        )

    # Simple feature shape check
    first = data_list[0]
    print("\nFirst graph shapes:")
    print("x:", tuple(first.x.shape))
    print("edge_index:", tuple(first.edge_index.shape))
    print("edge_attr:", tuple(first.edge_attr.shape))
    print("g:", tuple(first.g.shape))
    print("y:", first.y.item())


if __name__ == "__main__":
    main()