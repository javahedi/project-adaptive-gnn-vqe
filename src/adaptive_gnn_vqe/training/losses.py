
import torch
from torch_geometric.utils import softmax, scatter
import torch.nn.functional as F


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


def pointer_loss(scores, batch):
    e_batch = edge_batch_index(batch)
    _, e_ptr = edge_ptr_from_edge_batch(e_batch, batch.num_graphs)
    y_global = local_to_global_targets(batch, e_batch, e_ptr)

    p = softmax(scores, e_batch)
    return -torch.log(p[y_global] + 1e-12).mean()


def pointer_soft_loss(scores, batch, tau=0.5, eps=1e-12):
    """
    scores: [E_total] model logits over edges
    batch.g: [E_total] teacher edge scores (gvals) concatenated across graphs
    """
    e_batch = batch.batch[batch.edge_index[0]]  # [E_total]

    # student distribution
    p = softmax(scores, e_batch)  # [E_total]
    logp = torch.log(p + eps)

    # teacher distribution (temperature tau)
    g = batch.g
    q = softmax(g / tau, e_batch)  # [E_total]

    # cross-entropy: sum_e q * (-log p) per graph
    per_edge_ce = -(q * logp)                      # [E_total]
    per_graph_ce = scatter(per_edge_ce, e_batch, dim=0, reduce="sum")  # [B]
    return per_graph_ce.mean()




@torch.no_grad()
def pointer_accuracy(scores, batch):
    e_batch = edge_batch_index(batch)
    _, e_ptr = edge_ptr_from_edge_batch(e_batch, batch.num_graphs)
    y_global = local_to_global_targets(batch, e_batch, e_ptr)

    correct = 0
    for g in range(batch.num_graphs):
        start, end = e_ptr[g].item(), e_ptr[g + 1].item()
        pred = start + torch.argmax(scores[start:end]).item()
        correct += int(pred == y_global[g])
    return correct / batch.num_graphs


@torch.no_grad()
def pointer_topk_accuracy(scores, batch, k=5):
    e_batch = edge_batch_index(batch)
    _, e_ptr = edge_ptr_from_edge_batch(e_batch, batch.num_graphs)
    y_global = local_to_global_targets(batch, e_batch, e_ptr)

    hit = 0
    for g in range(batch.num_graphs):
        start, end = e_ptr[g].item(), e_ptr[g + 1].item()
        k_eff = min(k, end - start)
        topk = torch.topk(scores[start:end], k_eff).indices + start
        hit += int((topk == y_global[g]).any().item())
    return hit / batch.num_graphs



@torch.no_grad()
def pointer_mean_rank(scores, batch):
    """
    Mean rank of the correct edge within each graph.
    Rank = 1 means best possible.
    """
    e_batch = edge_batch_index(batch)
    _, e_ptr = edge_ptr_from_edge_batch(e_batch, batch.num_graphs)
    y_global = local_to_global_targets(batch, e_batch, e_ptr)

    ranks = []

    for g in range(batch.num_graphs):
        start, end = e_ptr[g].item(), e_ptr[g + 1].item()

        graph_scores = scores[start:end]
        true_edge = y_global[g].item() - start

        # sort descending
        order = torch.argsort(graph_scores, descending=True)

        rank = (order == true_edge).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    return sum(ranks) / len(ranks)



def sign_loss(sign_logits, batch):
    """
    Binary sign prediction loss.

    Target:
        g_signed > 0  -> 1
        g_signed < 0  -> 0
    """
    target = (batch.g_signed > 0).float()
    return F.binary_cross_entropy_with_logits(sign_logits, target)


@torch.no_grad()
def sign_accuracy(sign_logits, batch):
    pred = sign_logits > 0
    target = batch.g_signed > 0
    return (pred == target).float().mean().item()



def selected_sign_loss(sign_logits, batch):
    """
    Sign loss only on the oracle-selected edge y for each graph.
    """
    e_batch = edge_batch_index(batch)
    _, e_ptr = edge_ptr_from_edge_batch(e_batch, batch.num_graphs)
    y_global = local_to_global_targets(batch, e_batch, e_ptr)

    selected_logits = sign_logits[y_global]
    selected_targets = (batch.g_signed[y_global] > 0).float()

    return F.binary_cross_entropy_with_logits(
        selected_logits,
        selected_targets,
    )


@torch.no_grad()
def selected_sign_accuracy(sign_logits, batch):
    """
    Sign accuracy only on the oracle-selected edge y for each graph.
    """
    e_batch = edge_batch_index(batch)
    _, e_ptr = edge_ptr_from_edge_batch(e_batch, batch.num_graphs)
    y_global = local_to_global_targets(batch, e_batch, e_ptr)

    pred = sign_logits[y_global] > 0
    target = batch.g_signed[y_global] > 0

    return (pred == target).float().mean().item()