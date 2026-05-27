import json
import torch

from adaptive_gnn_vqe.models import PointerGNN
from adaptive_gnn_vqe.vqe.compare import run_one_realization


if __name__ == "__main__":

    device = torch.device("cpu")

    model = PointerGNN(
        node_in=8,
        edge_in=7,
        hidden=96,
        mp_layers=5,
    )

    model.load_state_dict(
        torch.load(
            "outputs/models/pointer_gnn_best_acc.pt",
            map_location=device,
        )
    )

    model.to(device)
    model.eval()

    result = run_one_realization(
        model,
        N=8,
        L=80,
        alpha=1.5,
        K=6,
        T=6,
        seed=0,
    )

    print(json.dumps(result, indent=2))