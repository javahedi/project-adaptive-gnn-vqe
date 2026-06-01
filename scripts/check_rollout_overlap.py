import numpy as np
import torch

from adaptive_gnn_vqe.models import PointerGNN
from adaptive_gnn_vqe.data.xxz_generator import (
    sample_positions_discrete,
    couplings_from_positions,
    topK_edge_list,
    build_H_xxz_longrange,
    random_product_state,
    P_ij,
    apply_entangler,
    commutator_expectation_signed,
)
from adaptive_gnn_vqe.data.builders import build_xxz_graph_from_state


def select_gnn(model, psi, r, J, edges, P_cache, alpha, Delta):
    data = build_xxz_graph_from_state(
        psi=psi,
        r=r,
        J=J,
        edges=edges,
        P_cache=P_cache,
        alpha=alpha,
        Delta=Delta,
    )
    data = data.to(next(model.parameters()).device)

    with torch.no_grad():
        scores = model(data)

    return int(torch.argmax(scores).item())


def run_one(model, seed, N=10, L=100, alpha=1.5, K=7, T=6):
    J0 = 1.0
    Delta = 0.0
    theta0 = 0.05

    rng = np.random.default_rng(seed)

    r = sample_positions_discrete(N, L, rng)
    J = couplings_from_positions(r, J0, alpha)
    edges = topK_edge_list(J, K)

    H = build_H_xxz_longrange(J, Delta)
    P_cache = [P_ij(N, i, j, Delta) for (i, j) in edges]

    psi_gnn = random_product_state(N, np.random.default_rng(seed + 10000))
    psi_oracle = psi_gnn.copy()

    matches = []
    oracle_edges = []
    gnn_edges = []

    for t in range(T):
        signed_grads = np.array([
            commutator_expectation_signed(H, P_cache[e], psi_oracle)
            for e in range(len(edges))
        ])

        oracle_idx = int(np.argmax(np.abs(signed_grads)))

        gnn_idx = select_gnn(
            model=model,
            psi=psi_gnn,
            r=r,
            J=J,
            edges=edges,
            P_cache=P_cache,
            alpha=alpha,
            Delta=Delta,
        )

        matches.append(int(oracle_idx == gnn_idx))
        oracle_edges.append(edges[oracle_idx])
        gnn_edges.append(edges[gnn_idx])

        theta_oracle = theta0 * np.sign(signed_grads[oracle_idx])
        if theta_oracle == 0:
            theta_oracle = theta0

        psi_oracle = apply_entangler(psi_oracle, P_cache[oracle_idx], theta_oracle)
        psi_oracle /= np.linalg.norm(psi_oracle)

        # GNN currently has no signed angle prediction.
        # Use fixed positive theta, same as training rollout.
        psi_gnn = apply_entangler(psi_gnn, P_cache[gnn_idx], theta0)
        psi_gnn /= np.linalg.norm(psi_gnn)

    return matches, oracle_edges, gnn_edges


def main():
    device = torch.device("cpu")

    model = PointerGNN(node_in=8, edge_in=7, hidden=96, mp_layers=5)
    model.load_state_dict(
        torch.load("outputs/models/pointer_gnn_best_acc.pt", map_location=device)
    )
    model.to(device)
    model.eval()

    n_realizations = 200
    T = 6

    all_matches = []

    for seed in range(n_realizations):
        matches, _, _ = run_one(model, seed=seed, T=T)
        all_matches.append(matches)

    all_matches = np.array(all_matches)

    print("GNN vs gradient-oracle rollout overlap")
    print("--------------------------------------")
    print(f"Realizations: {n_realizations}")

    for t in range(T):
        print(f"depth {t+1}: overlap = {all_matches[:, t].mean():.4f}")

    print("--------------------------------------")
    print(f"mean overlap over all depths = {all_matches.mean():.4f}")


if __name__ == "__main__":
    main()