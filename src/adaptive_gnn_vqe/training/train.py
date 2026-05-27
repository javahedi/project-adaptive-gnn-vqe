import os, json
import random
from collections import  Counter, defaultdict

import torch
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

from adaptive_gnn_vqe.models import PointerGNN
from adaptive_gnn_vqe.training.losses import (
    pointer_soft_loss,
    pointer_accuracy,
    pointer_topk_accuracy,
    pointer_mean_rank,
)



BASE_DIR = os.path.dirname(__file__)

MODEL_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "../../../outputs/models")
)

LOG_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "../../../outputs/logs")
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


log = {
    "epoch": [],
    "train_loss": [],
    "train_acc": [],
    "train_rank": [],
    "val_acc": [],
    "val_rank": [],
    "val_top5": [],
    "per_alpha": [],
    "per_N": []
}

os.makedirs("logs", exist_ok=True)


# ============================================================
# Utilities
# ============================================================
def edge_batch_index(batch):
    return batch.batch[batch.edge_index[0]]


def edge_ptr_from_edge_batch(e_batch, num_graphs):
    counts = torch.bincount(e_batch, minlength=num_graphs)
    ptr = torch.zeros(num_graphs + 1, device=e_batch.device, dtype=torch.long)
    ptr[1:] = torch.cumsum(counts, dim=0)
    return counts, ptr


def local_to_global_targets(batch, e_batch, e_ptr):
    y_local = batch.y.view(-1)
    g = torch.arange(batch.num_graphs, device=y_local.device)
    return e_ptr[g] + y_local


# ============================================================
# Per-alpha validation
# ============================================================
@torch.no_grad()
def evaluate_per_alpha(model, loader, device):
    acc_by_alpha = defaultdict(list)
    model.eval()

    for batch in loader:
        batch = batch.to(device)
        scores = model(batch)

        e_batch = edge_batch_index(batch)
        _, e_ptr = edge_ptr_from_edge_batch(e_batch, batch.num_graphs)
        y_global = local_to_global_targets(batch, e_batch, e_ptr)

        for g in range(batch.num_graphs):
            start, end = e_ptr[g].item(), e_ptr[g + 1].item()
            pred = start + torch.argmax(scores[start:end]).item()
            correct = int(pred == y_global[g])

            alpha_val = float(batch.alpha[g].item())
            acc_by_alpha[alpha_val].append(correct)

    out = {a: sum(v) / len(v) for a, v in acc_by_alpha.items()}
    return dict(sorted(out.items(), key=lambda kv: kv[0]))



@torch.no_grad()
def evaluate_per_N(model, loader, device):
    acc_by_N = defaultdict(list)
    model.eval()

    for batch in loader:
        batch = batch.to(device)
        scores = model(batch)

        e_batch = edge_batch_index(batch)
        _, e_ptr = edge_ptr_from_edge_batch(e_batch, batch.num_graphs)
        y_global = local_to_global_targets(batch, e_batch, e_ptr)

        for g in range(batch.num_graphs):
            start, end = e_ptr[g].item(), e_ptr[g + 1].item()
            pred = start + torch.argmax(scores[start:end]).item()
            correct = int(pred == y_global[g])

            N_val = int(batch.N[g].item())
            acc_by_N[N_val].append(correct)

    return {k: sum(v)/len(v) for k,v in sorted(acc_by_N.items())}




@torch.no_grad()
def uniform_baseline_accuracy(batch):
    e_batch = edge_batch_index(batch)
    counts = torch.bincount(e_batch, minlength=batch.num_graphs)
    return float((1.0 / counts.float()).mean().item())


@torch.no_grad()
def strongest_J_baseline_accuracy(batch):
    e_batch = edge_batch_index(batch)
    _, e_ptr = edge_ptr_from_edge_batch(e_batch, batch.num_graphs)

    logJ = batch.edge_attr[:, 0]
    y_local = batch.y.view(-1)

    correct = 0
    for g in range(batch.num_graphs):
        start, end = e_ptr[g].item(), e_ptr[g + 1].item()
        pred = start + torch.argmax(logJ[start:end]).item()
        true = start + y_local[g].item()
        correct += int(pred == true)
    return correct / batch.num_graphs





# ============================================================
# Training
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    random.seed(0)
    torch.manual_seed(0)

    #data_list = torch.load("../data/spinchain_dataset_multisize_realizations_per_alpha200.pt")

    data_path = os.path.join(
    os.path.dirname(__file__),
    "../data/test_spinchain_dataset.pt"
    )

    data_path = os.path.abspath(data_path)
    data_list = torch.load(data_path)

    # split by realization_id (as you had)
    real_ids = sorted(set(int(d.realization_id) for d in data_list))
    random.shuffle(real_ids)

    split = int(0.8 * len(real_ids))
    train_ids = set(real_ids[:split])
    val_ids = set(real_ids[split:])

    train_ds = [d for d in data_list if int(d.realization_id) in train_ids]
    val_ds = [d for d in data_list if int(d.realization_id) in val_ids]

    # ---- β-balanced sampling across alpha (training only) ----
    beta = 0.5  # 0 -> no balance, 1 -> fully balanced

    keys = [(float(d.alpha), int(d.N)) for d in train_ds]
    cnt = Counter(keys)

    weights = torch.tensor(
        [(1.0 / cnt[(float(d.alpha), int(d.N))]) ** beta for d in train_ds],
        dtype=torch.double
    )
   
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler)  # keep sampler, NO shuffle
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    node_in = data_list[0].x.size(1)
    base_edge_in = data_list[0].edge_attr.size(1)

    # add alpha into edge features inside forward => +1
    model = PointerGNN(node_in, base_edge_in , hidden=96, mp_layers=5).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_val = 0.0
    best_epoch = 0

    best_rank = float("inf")
    best_rank_epoch = 0

    for epoch in range(1, 101):
        model.train()
        tr_loss = tr_acc = tr_top5 = tr_unif = tr_J = 0.0
        tr_rank = 0.0
        nb = 0

        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            scores = model(batch)
            #loss = pointer_loss(scores, batch)
            loss = pointer_soft_loss(scores, batch, tau=0.3)
            loss.backward()
            opt.step()

            tr_loss += loss.item()
            tr_acc += pointer_accuracy(scores, batch)
            tr_top5 += pointer_topk_accuracy(scores, batch, 5)
            tr_unif += uniform_baseline_accuracy(batch)
            tr_J += strongest_J_baseline_accuracy(batch)
            tr_rank += pointer_mean_rank(scores, batch)
            nb += 1

        tr_loss /= nb
        tr_acc /= nb
        tr_top5 /= nb
        tr_unif /= nb
        tr_J /= nb
        tr_rank /= nb

        model.eval()
        va_acc = va_top5 = va_unif = va_J = 0.0
        va_rank = 0.0
        nb = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                scores = model(batch)
                va_acc += pointer_accuracy(scores, batch)
                va_top5 += pointer_topk_accuracy(scores, batch, 5)
                va_unif += uniform_baseline_accuracy(batch)
                va_J += strongest_J_baseline_accuracy(batch)
                va_rank += pointer_mean_rank(scores, batch)
                nb += 1

        va_acc /= nb
        va_top5 /= nb
        va_unif /= nb
        va_rank /= nb
        va_J /= nb

        per_alpha = evaluate_per_alpha(model, val_loader, device)
        per_N = evaluate_per_N(model, val_loader, device)

        if va_acc > best_val:
            best_val = va_acc
            best_epoch = epoch
            torch.save( model.state_dict(),
            os.path.join(MODEL_DIR, "pointer_gnn_best_acc.pt")
            )

        if va_rank < best_rank:
            best_rank = va_rank
            best_rank_epoch = epoch
            torch.save(
            model.state_dict(),
            os.path.join(MODEL_DIR, "pointer_gnn_best_rank.pt")
            )

        print(
            f"Epoch {epoch:03d} | loss {tr_loss:.4f} | "
            f"train acc {tr_acc:.3f} (top5 {tr_top5:.3f}) | "
            f"train rank {tr_rank:.1f} | "
            f"val acc {va_acc:.3f} (top5 {va_top5:.3f}) | "
            f"val rank {va_rank:.1f} | "
            f"best acc {best_val:.3f} @ {best_epoch} | best rank {best_rank:.2f} @ {best_rank_epoch}"
        )
        print("  Val per alpha:", {round(k, 2): round(v, 3) for k, v in per_alpha.items()})
        print("  Val per N:", {k: round(v, 3) for k, v in per_N.items()})

        # ------------------------------------------------------------
        # Save metrics to log
        # ------------------------------------------------------------
        log["epoch"].append(epoch)
        log["train_loss"].append(tr_loss)
        log["train_acc"].append(tr_acc)
        log["train_rank"].append(tr_rank)

        log["val_acc"].append(va_acc)
        log["val_rank"].append(va_rank)
        log["val_top5"].append(va_top5)

        log["per_alpha"].append(per_alpha)
        log["per_N"].append(per_N)

       

        with open(os.path.join(LOG_DIR, "train_log.json"), "w") as f:       
            json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()