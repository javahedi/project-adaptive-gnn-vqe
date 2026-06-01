import os
import json
import copy
import random
from collections import Counter

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

from adaptive_gnn_vqe.models import PointerGNN
from adaptive_gnn_vqe.training.losses import (
    pointer_soft_loss,
    pointer_accuracy,
    pointer_topk_accuracy,
    pointer_mean_rank,
)


# ============================================================
# Config
# ============================================================
DATA_PATH = "src/adaptive_gnn_vqe/data/test_spinchain_dataset.pt"
OUT_JSON = "outputs/results/ablation_results.json"
OUT_FIG = "figures/figure_ablation_summary.pdf"

EPOCHS = 30
BATCH_SIZE = 64
VAL_BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 0


ABLATIONS = {
    "full": {
        "node_cols": [0, 1, 2, 3, 4, 5, 6, 7],
        "edge_cols": [0, 1, 2, 3, 4, 5, 6],
        "mp_layers": 5,
    },
    "static_only": {
        "node_cols": [0, 1, 2, 3, 6],
        "edge_cols": [0, 1, 2],
        "mp_layers": 5,
    },
    "no_p_var": {
        "node_cols": [0, 1, 2, 3, 4, 5, 6, 7],
        "edge_cols": [0, 1, 2, 3, 4, 5],
        "mp_layers": 5,
    },
    "no_message_passing": {
        "node_cols": [0, 1, 2, 3, 4, 5, 6, 7],
        "edge_cols": [0, 1, 2, 3, 4, 5, 6],
        "mp_layers": 0,
    },
    "no_alpha": {
        "node_cols": [0, 1, 2, 3, 4, 5, 7],
        "edge_cols": [0, 1, 2, 3, 4, 5, 6],
        "mp_layers": 5,
    },
    "no_rank_feature": {
        "node_cols": [0, 1, 2, 3, 4, 5, 6, 7],
        "edge_cols": [0, 1, 3, 4, 5, 6],
        "mp_layers": 5,
    },
    "no_correlators": {
        "node_cols": [0, 1, 2, 3, 4, 5, 6, 7],
        "edge_cols": [0, 1, 2, 6],
        "mp_layers": 5,
    },
}


# ============================================================
# Helpers
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mask_dataset(data_list, node_cols, edge_cols):
    out = []

    for d in data_list:
        d2 = copy.copy(d)
        d2.x = d.x[:, node_cols].clone()
        d2.edge_attr = d.edge_attr[:, edge_cols].clone()
        out.append(d2)

    return out


def split_by_realization(data_list):
    real_ids = sorted(set(int(d.realization_id) for d in data_list))
    random.shuffle(real_ids)

    split = int(0.7 * len(real_ids))
    train_ids = set(real_ids[:split])
    val_ids = set(real_ids[split:])

    train_ds = [d for d in data_list if int(d.realization_id) in train_ids]
    val_ds = [d for d in data_list if int(d.realization_id) in val_ids]

    return train_ds, val_ds


def make_loaders(train_ds, val_ds):
    beta = 0.5
    keys = [(float(d.alpha), int(d.N)) for d in train_ds]
    cnt = Counter(keys)

    weights = torch.tensor(
        [(1.0 / cnt[(float(d.alpha), int(d.N))]) ** beta for d in train_ds],
        dtype=torch.double,
    )

    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


@torch.no_grad()
def gradient_ratio(model, loader, device):
    ratios = []
    ranks = []

    model.eval()

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

            pred = int(torch.argmax(graph_scores).item())
            oracle = int(torch.argmax(graph_g).item())

            g_pred = float(graph_g[pred].item())
            g_oracle = float(graph_g[oracle].item())

            ratios.append(g_pred / (g_oracle + 1e-12))

            order = torch.argsort(graph_g, descending=True)
            rank = (order == pred).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)

    return {
        "mean_gradient_ratio": float(np.mean(ratios)),
        "median_gradient_ratio": float(np.median(ratios)),
        "top3_by_gradient": float(np.mean(np.array(ranks) <= 3)),
        "top5_by_gradient": float(np.mean(np.array(ranks) <= 5)),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    acc = 0.0
    top5 = 0.0
    rank = 0.0
    nb = 0

    for batch in loader:
        batch = batch.to(device)
        scores = model(batch)

        acc += pointer_accuracy(scores, batch)
        top5 += pointer_topk_accuracy(scores, batch, 5)
        rank += pointer_mean_rank(scores, batch)
        nb += 1

    return {
        "acc": acc / nb,
        "top5": top5 / nb,
        "rank": rank / nb,
    }


def train_one_variant(name, cfg, base_data, device):
    print(f"\n==============================")
    print(f"Ablation: {name}")
    print(f"==============================")

    data = mask_dataset(base_data, cfg["node_cols"], cfg["edge_cols"])

    train_ds, val_ds = split_by_realization(data)
    train_loader, val_loader = make_loaders(train_ds, val_ds)

    node_in = data[0].x.size(1)
    edge_in = data[0].edge_attr.size(1)

    model = PointerGNN(
        node_in=node_in,
        edge_in=edge_in,
        hidden=96,
        mp_layers=cfg["mp_layers"],
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best = {
        "epoch": 0,
        "val_acc": 0.0,
        "val_top5": 0.0,
        "val_rank": float("inf"),
        "train_acc": 0.0,
        "train_rank": float("inf"),
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()

        tr_loss = 0.0
        nb = 0

        for batch in train_loader:
            batch = batch.to(device)

            opt.zero_grad()
            scores = model(batch)
            loss = pointer_soft_loss(scores, batch, tau=0.3)
            loss.backward()
            opt.step()

            tr_loss += loss.item()
            nb += 1

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)

        if val_metrics["acc"] > best["val_acc"]:
            best.update(
                {
                    "epoch": epoch,
                    "val_acc": val_metrics["acc"],
                    "val_top5": val_metrics["top5"],
                    "val_rank": val_metrics["rank"],
                    "train_acc": train_metrics["acc"],
                    "train_rank": train_metrics["rank"],
                }
            )

        print(
            f"Epoch {epoch:03d} | "
            f"loss {tr_loss / nb:.4f} | "
            f"val acc {val_metrics['acc']:.3f} | "
            f"val top5 {val_metrics['top5']:.3f} | "
            f"val rank {val_metrics['rank']:.2f} | "
            f"best {best['val_acc']:.3f} @ {best['epoch']}"
        )

    grad_metrics = gradient_ratio(model, val_loader, device)

    best.update(grad_metrics)
    best["node_dim"] = node_in
    best["edge_dim"] = edge_in
    best["mp_layers"] = cfg["mp_layers"]

    return best


def plot_results(results):
    names = list(results.keys())

    acc = [results[n]["val_acc"] for n in names]
    rank = [results[n]["val_rank"] for n in names]
    ratio = [results[n]["mean_gradient_ratio"] for n in names]

    x = np.arange(len(names))

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))

    ax[0].bar(x, acc)
    ax[0].set_ylabel("Validation accuracy")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(names, rotation=45, ha="right")

    ax[1].bar(x, rank)
    ax[1].set_ylabel("Mean rank")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(names, rotation=45, ha="right")

    ax[2].bar(x, ratio)
    ax[2].set_ylabel(r"Mean $|g_{\rm pred}|/|g_{\rm oracle}|$")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(names, rotation=45, ha="right")
    ax[2].set_ylim(0, 1)

    fig.tight_layout()

    os.makedirs("figures", exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    fig.savefig(OUT_FIG.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")

    print(f"\nSaved {OUT_FIG}")


def main():
    set_seed(SEED)

    os.makedirs("outputs/results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    base_data = torch.load(DATA_PATH, map_location="cpu")
    print("Loaded graphs:", len(base_data))
    print("Node dim:", base_data[0].x.size(1))
    print("Edge dim:", base_data[0].edge_attr.size(1))

    results = {}

    for name, cfg in ABLATIONS.items():
        set_seed(SEED)
        results[name] = train_one_variant(name, cfg, base_data, device)

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {OUT_JSON}")

    plot_results(results)

    print("\nSummary:")
    for name, r in results.items():
        print(
            f"{name:18s} | "
            f"acc={r['val_acc']:.3f} | "
            f"rank={r['val_rank']:.2f} | "
            f"ratio={r['mean_gradient_ratio']:.3f} | "
            f"top5g={r['top5_by_gradient']:.3f}"
        )


if __name__ == "__main__":
    main()