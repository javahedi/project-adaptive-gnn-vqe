import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh

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


def exact_ground_energy(H):
    return float(np.real(eigsh(H, k=1, which="SA", return_eigenvectors=False)[0]))


def vqe_refine(H, psi0, P_cache, chosen_edges, maxiter=250):
    theta0 = np.zeros(len(chosen_edges), dtype=float)

    def objective(thetas):
        psi = psi0.copy()
        for theta, eidx in zip(thetas, chosen_edges):
            psi = apply_entangler(psi, P_cache[eidx], float(theta))
            psi /= np.linalg.norm(psi)
        return energy(H, psi)

    res = minimize(
        objective,
        theta0,
        method="Nelder-Mead",
        options={"maxiter": maxiter, "disp": False},
    )

    return float(res.fun)


def select_gnn_edge(model, psi, r, J, edges, P_cache, alpha, Delta):
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


def build_circuit_oracle(H, psi0, P_cache, T, theta0=0.05):
    psi = psi0.copy()
    chosen = []

    for _ in range(T):
        signed_grads = np.array([
            commutator_expectation_signed(H, P_cache[e], psi)
            for e in range(len(P_cache))
        ])

        e_star = int(np.argmax(np.abs(signed_grads)))
        chosen.append(e_star)

        theta = theta0 * np.sign(signed_grads[e_star])
        if theta == 0:
            theta = theta0

        psi = apply_entangler(psi, P_cache[e_star], theta)
        psi /= np.linalg.norm(psi)

    return chosen


def build_circuit_gnn(model, H, psi0, r, J, edges, P_cache, alpha, Delta, T, theta0=0.05):
    psi = psi0.copy()
    chosen = []

    for _ in range(T):
        e_star = select_gnn_edge(
            model=model,
            psi=psi,
            r=r,
            J=J,
            edges=edges,
            P_cache=P_cache,
            alpha=alpha,
            Delta=Delta,
        )
        chosen.append(e_star)

        # Diagnostic mode: use oracle sign for selected GNN edge.
        signed_grad = commutator_expectation_signed(H, P_cache[e_star], psi)
        theta = theta0 * np.sign(signed_grad)
        if theta == 0:
            theta = theta0

        psi = apply_entangler(psi, P_cache[e_star], theta)
        psi /= np.linalg.norm(psi)

    return chosen


def build_circuit_strongest_j(J, edges, T):
    J_edges = np.array([J[i, j] for i, j in edges])
    e_star = int(np.argmax(J_edges))
    return [e_star for _ in range(T)]


def build_circuit_random(num_edges, T, seed):
    rng = np.random.default_rng(seed + 999)
    return [int(rng.integers(num_edges)) for _ in range(T)]


def run_one_realization(model, N, L, alpha, K, Delta, T, seed):
    J0 = 1.0

    rng = np.random.default_rng(seed)
    r = sample_positions_discrete(N, L, rng)
    J = couplings_from_positions(r, J0, alpha)
    edges = topK_edge_list(J, K=K)

    H = build_H_xxz_longrange(J, Delta)
    P_cache = [P_ij(N, i, j, Delta) for i, j in edges]

    psi0 = random_product_state(N, np.random.default_rng(seed + 10000))
    E0 = exact_ground_energy(H)

    oracle_edges = build_circuit_oracle(H, psi0, P_cache, T)
    gnn_edges = build_circuit_gnn(model, H, psi0, r, J, edges, P_cache, alpha, Delta, T)
    j_edges = build_circuit_strongest_j(J, edges, T)
    rand_edges = build_circuit_random(len(edges), T, seed)

    E_oracle = vqe_refine(H, psi0, P_cache, oracle_edges)
    E_gnn = vqe_refine(H, psi0, P_cache, gnn_edges)
    E_j = vqe_refine(H, psi0, P_cache, j_edges)
    E_rand = vqe_refine(H, psi0, P_cache, rand_edges)

    return {
        "oracle": E_oracle - E0,
        "gnn": E_gnn - E0,
        "strongest_j": E_j - E0,
        "random": E_rand - E0,
    }


def mean_err(values):
    values = np.array(values, dtype=float)
    return float(values.mean()), float(values.std() / np.sqrt(len(values)))


def main():
    device = torch.device("cpu")

    model = PointerGNN(node_in=8, edge_in=7, hidden=96, mp_layers=5)
    model.load_state_dict(
        torch.load("outputs/models/pointer_gnn_best_acc.pt", map_location=device)
    )
    model.to(device)
    model.eval()

    N = 8
    L = 80
    alpha = 1.5
    K = 6
    Delta = 0.0

    T_list = [2, 4, 6, 8, 10, 12]
    n_realizations = 10

    results = {
        "T": T_list,
        "oracle": [],
        "gnn": [],
        "strongest_j": [],
        "random": [],
    }

    for T in T_list:
        print(f"\nDepth T={T}")

        vals = {
            "oracle": [],
            "gnn": [],
            "strongest_j": [],
            "random": [],
        }

        for seed in range(n_realizations):
            print(f"  realization {seed+1}/{n_realizations}")

            out = run_one_realization(
                model=model,
                N=N,
                L=L,
                alpha=alpha,
                K=K,
                Delta=Delta,
                T=T,
                seed=seed,
            )

            for key in vals:
                vals[key].append(out[key])

        for key in vals:
            m, e = mean_err(vals[key])
            results[key].append({"mean": m, "err": e})

        print(
            f"  dE oracle={results['oracle'][-1]['mean']:.4f}, "
            f"gnn={results['gnn'][-1]['mean']:.4f}, "
            f"J={results['strongest_j'][-1]['mean']:.4f}, "
            f"rand={results['random'][-1]['mean']:.4f}"
        )

    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    with open("outputs/results/vqe_refined_depth_scaling.json", "w") as f:
        json.dump(results, f, indent=2)

    T = np.array(T_list)

    def unpack(key):
        mean = np.array([x["mean"] for x in results[key]])
        err = np.array([x["err"] for x in results[key]])
        return mean, err

    oracle_m, oracle_e = unpack("oracle")
    gnn_m, gnn_e = unpack("gnn")
    j_m, j_e = unpack("strongest_j")
    rand_m, rand_e = unpack("random")

    plt.figure(figsize=(6, 4))
    plt.errorbar(T, oracle_m, yerr=oracle_e, fmt="o-", capsize=3, label="ADAPT-VQE")
    plt.errorbar(T, gnn_m, yerr=gnn_e, fmt="s-", capsize=3, label="GNN-VQE")
    plt.errorbar(T, j_m, yerr=j_e, fmt="^-", capsize=3, label="Strongest $J$")
    plt.errorbar(T, rand_m, yerr=rand_e, fmt="D-", capsize=3, label="Random")

    plt.xlabel("Circuit depth $T$")
    plt.ylabel(r"Energy error $\Delta E = E - E_0$")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig("figures/figure_depth_scaling.pdf", bbox_inches="tight")
    plt.savefig("figures/figure_depth_scaling.png", dpi=300, bbox_inches="tight")

    print("\nSaved figures/figure_depth_scaling.pdf")


if __name__ == "__main__":
    main()