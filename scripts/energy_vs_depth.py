# scripts/energy_vs_depth.py
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

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


def energy(H, psi):
    return float(np.real(np.vdot(psi, H.dot(psi))))


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


def run_one_realization(model, N=10, L=100, alpha=1.5, K=7, T=6, seed=0):
    J0 = 1.0
    Delta = 0.0
    theta0 = 0.05

    rng = np.random.default_rng(seed)

    r = sample_positions_discrete(N, L, rng)
    J = couplings_from_positions(r, J0, alpha)
    edges = topK_edge_list(J, K=K)

    H = build_H_xxz_longrange(J, Delta)
    P_cache = [P_ij(N, i, j, Delta) for (i, j) in edges]

    psi0 = random_product_state(N, np.random.default_rng(seed + 10000))

    psi_random = psi0.copy()
    psi_J = psi0.copy()
    psi_gnn = psi0.copy()
    psi_grad = psi0.copy()

    e_random = []
    e_J = []
    e_gnn = []
    e_grad = []

    J_edges = np.array([J[i, j] for (i, j) in edges])

    rng_random = np.random.default_rng(seed + 999)

    for _ in range(T):
        # random
        idx_random = rng_random.integers(len(edges))

        # strongest coupling
        idx_J = int(np.argmax(J_edges))

        # gradient oracle, evaluated on its own current state
        signed_grads = np.array([
        commutator_expectation_signed(H, P_cache[e], psi_grad)
            for e in range(len(edges))
        ])

        idx_grad = int(np.argmax(np.abs(signed_grads)))

        theta_grad = theta0 * np.sign(signed_grads[idx_grad])
        if theta_grad == 0:
            theta_grad = theta0

        # GNN, evaluated on its own current state
        idx_gnn = select_gnn(
            model=model,
            psi=psi_gnn,
            r=r,
            J=J,
            edges=edges,
            P_cache=P_cache,
            alpha=alpha,
            Delta=Delta,
        )

        psi_random = apply_entangler(psi_random, P_cache[idx_random], theta0)
        psi_J = apply_entangler(psi_J, P_cache[idx_J], theta0)
        psi_grad = apply_entangler(psi_grad, P_cache[idx_grad], theta_grad)

        signed_grads_gnn = np.array([
            commutator_expectation_signed(H, P_cache[e], psi_gnn)
            for e in range(len(edges))
        ])

        theta_gnn = theta0 * np.sign(signed_grads_gnn[idx_gnn])
        if theta_gnn == 0:
            theta_gnn = theta0

        psi_gnn = apply_entangler(psi_gnn, P_cache[idx_gnn], theta_gnn)
        

        psi_random /= np.linalg.norm(psi_random)
        psi_J /= np.linalg.norm(psi_J)
        psi_grad /= np.linalg.norm(psi_grad)
        psi_gnn /= np.linalg.norm(psi_gnn)

        e_random.append(energy(H, psi_random))
        e_J.append(energy(H, psi_J))
        e_gnn.append(energy(H, psi_gnn))
        e_grad.append(energy(H, psi_grad))

    return e_random, e_J, e_gnn, e_grad


def mean_err(arr):
    arr = np.array(arr)
    return arr.mean(axis=0), arr.std(axis=0) / np.sqrt(arr.shape[0])


def main():
    device = torch.device("cpu")

    model = PointerGNN(node_in=8, edge_in=7, hidden=96, mp_layers=5)
    model.load_state_dict(
        torch.load("outputs/models/pointer_gnn_best_acc.pt", map_location=device)
    )
    model.to(device)
    model.eval()

    n_realizations = 200

    all_random = []
    all_J = []
    all_gnn = []
    all_grad = []

    for seed in range(n_realizations):
        print(f"Realization {seed+1}/{n_realizations}")

        r, j, gnn, grad = run_one_realization(
            model=model,
            N=10,
            L=100,
            alpha=1.5,
            K=7,
            T=6,
            seed=seed,
        )

        all_random.append(r)
        all_J.append(j)
        all_gnn.append(gnn)
        all_grad.append(grad)

    rand_m, rand_e = mean_err(all_random)
    J_m, J_e = mean_err(all_J)
    gnn_m, gnn_e = mean_err(all_gnn)
    grad_m, grad_e = mean_err(all_grad)

    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    results = {
        "random_mean": rand_m.tolist(),
        "random_err": rand_e.tolist(),
        "strongestJ_mean": J_m.tolist(),
        "strongestJ_err": J_e.tolist(),
        "gnn_mean": gnn_m.tolist(),
        "gnn_err": gnn_e.tolist(),
        "gradient_mean": grad_m.tolist(),
        "gradient_err": grad_e.tolist(),
    }

    with open("outputs/results/energy_vs_depth.json", "w") as f:
        json.dump(results, f, indent=2)

    depth = np.arange(1, len(rand_m) + 1)

    plt.figure(figsize=(6, 4))
    plt.errorbar(depth, rand_m, yerr=rand_e, fmt="o-", capsize=3, label="Random")
    plt.errorbar(depth, J_m, yerr=J_e, fmt="s-", capsize=3, label="Strongest $J$")
    plt.errorbar(depth, gnn_m, yerr=gnn_e, fmt="^-", capsize=3, label="GNN policy")
    plt.errorbar(depth, grad_m, yerr=grad_e, fmt="D-", capsize=3, label="Gradient oracle")

    plt.xlabel("Circuit depth $t$")
    plt.ylabel("Energy $E$")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig("figures/figure_energy_vs_depth.pdf", bbox_inches="tight")
    plt.savefig("figures/figure_energy_vs_depth.png", dpi=300, bbox_inches="tight")

    print("Saved figures/figure_energy_vs_depth.pdf")


if __name__ == "__main__":
    main()