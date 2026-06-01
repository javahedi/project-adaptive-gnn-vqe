import os
import random
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from adaptive_gnn_vqe.models import PointerGNN


def main():
    device = torch.device("cpu")

    data_path = os.path.abspath(
        "src/adaptive_gnn_vqe/data/test_spinchain_dataset.pt"
    )
    data_list = torch.load(data_path, map_location=device)

    # Same realization-level split as training
    random.seed(0)
    real_ids = sorted(set(int(d.realization_id) for d in data_list))
    random.shuffle(real_ids)

    split = int(0.7 * len(real_ids))
    val_ids = set(real_ids[split:])
    val_ds = [d for d in data_list if int(d.realization_id) in val_ids]

    loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    node_in = data_list[0].x.size(1)
    edge_in = data_list[0].edge_attr.size(1)

    model = PointerGNN(node_in=node_in, edge_in=edge_in, hidden=96, mp_layers=5)
    model.load_state_dict(
        torch.load("outputs/models/pointer_gnn_best_acc.pt", map_location=device)
    )
    model.to(device)
    model.eval()

    ratios = []
    ranks = []
    exact_hits = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            scores = model(batch)

            e_batch = batch.batch[batch.edge_index[0]]
            counts = torch.bincount(e_batch, minlength=batch.num_graphs)

            ptr = torch.zeros(batch.num_graphs + 1, dtype=torch.long, device=device)
            ptr[1:] = torch.cumsum(counts, dim=0)

            for g in range(batch.num_graphs):
                start = ptr[g].item()
                end = ptr[g + 1].item()

                graph_scores = scores[start:end]
                graph_g = torch.abs(batch.g_signed[start:end])

                pred_local = int(torch.argmax(graph_scores).item())
                oracle_local = int(torch.argmax(graph_g).item())

                g_pred = float(graph_g[pred_local].item())
                g_oracle = float(graph_g[oracle_local].item())

                ratio = g_pred / (g_oracle + 1e-12)
                ratios.append(ratio)

                order = torch.argsort(graph_g, descending=True)
                rank = (order == pred_local).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(rank)

                exact_hits += int(pred_local == oracle_local)
                total += 1

    ratios = np.array(ratios)
    ranks = np.array(ranks)

    print("Gradient-quality diagnostic")
    print("---------------------------")
    print(f"Validation samples: {total}")
    print(f"Exact oracle-edge accuracy: {exact_hits / total:.4f}")
    print()
    print(f"Mean |g_pred| / |g_oracle|:   {ratios.mean():.4f}")
    print(f"Median |g_pred| / |g_oracle|: {np.median(ratios):.4f}")
    print(f"25th percentile ratio:        {np.percentile(ratios, 25):.4f}")
    print(f"10th percentile ratio:        {np.percentile(ratios, 10):.4f}")
    print()
    print(f"Mean rank of GNN edge by true |g|:   {ranks.mean():.2f}")
    print(f"Median rank of GNN edge by true |g|: {np.median(ranks):.2f}")
    print(f"Top-3 by true |g|: {(ranks <= 3).mean():.4f}")
    print(f"Top-5 by true |g|: {(ranks <= 5).mean():.4f}")


if __name__ == "__main__":
    main()