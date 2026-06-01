import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader


def edge_batch_index(batch):
    return batch.batch[batch.edge_index[0]]


def edge_ptr_from_edge_batch(e_batch, num_graphs):
    counts = torch.bincount(e_batch, minlength=num_graphs)
    ptr = torch.zeros(num_graphs + 1, device=e_batch.device, dtype=torch.long)
    ptr[1:] = torch.cumsum(counts, dim=0)
    return counts, ptr


def main():
    data_path = os.path.abspath(
        "src/adaptive_gnn_vqe/data/test_spinchain_dataset.pt"
    )

    data_list = torch.load(data_path, map_location="cpu")
    loader = DataLoader(data_list, batch_size=128, shuffle=False)

    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    ranks = []
    total = 0

    for batch in loader:
        e_batch = edge_batch_index(batch)
        _, e_ptr = edge_ptr_from_edge_batch(e_batch, batch.num_graphs)

        logJ = batch.edge_attr[:, 0]
        y_local = batch.y.view(-1)

        for g in range(batch.num_graphs):
            start = e_ptr[g].item()
            end = e_ptr[g + 1].item()

            scores = logJ[start:end]
            true_local = int(y_local[g].item())

            order = torch.argsort(scores, descending=True)

            rank = (order == true_local).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)

            top1_hits += int(rank <= 1)
            top3_hits += int(rank <= 3)
            top5_hits += int(rank <= 5)
            total += 1

    print("Strongest-J diagnostic")
    print("----------------------")
    print(f"Total samples: {total}")
    print(f"Top-1 accuracy: {top1_hits / total:.4f}")
    print(f"Top-3 accuracy: {top3_hits / total:.4f}")
    print(f"Top-5 accuracy: {top5_hits / total:.4f}")
    print(f"Mean rank of true ADAPT edge by J: {np.mean(ranks):.2f}")
    print(f"Median rank: {np.median(ranks):.2f}")
    print(f"Best rank: {np.min(ranks)}")
    print(f"Worst rank: {np.max(ranks)}")


if __name__ == "__main__":
    main()