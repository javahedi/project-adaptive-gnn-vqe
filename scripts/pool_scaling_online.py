import os
import json
import time
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


def build_circuit_adapt_timed(H, psi0, P_cache, T, theta0=0.05):
    psi = psi0.copy()
    chosen = []
    selection_time = 0.0

    for _ in range(T):
        t0 = time.perf_counter()

        signed_grads = np.array([
            commutator_expectation_signed(H, P_cache[e], psi)
            for e in range(len(P_cache))
        ])

        e_star = int(np.argmax(np.abs(signed_grads)))
        selection_time += time.perf_counter() - t0

        chosen.append(e_star)

        theta = theta0 * np.sign(signed_grads[e_star])
        if theta == 0:
            theta = theta0

        psi = apply_entangler(psi, P_cache[e_star], theta)
        psi /= np.linalg.norm(psi)

    return chosen, selection_time


def select_gnn_edge(model, psi, r, J, edges, P_cache, alpha, Delta):
    device = next(model.parameters()).device

    t0 = time.perf_counter()
    data = build_xxz_graph_from_state(
        psi=psi,
        r=r,
        J=J,
        edges=edges,
        P_cache=P_cache,
        alpha=alpha,
        Delta=Delta,
    )
    t_graph = time.perf_counter() - t0

    data = data.to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        scores = model(data)
    t_forward = time.perf_counter() - t0

    e_star = int(torch.argmax(scores).item())

    return e_star, t_graph, t_forward


def build_circuit_gnn_timed(
    model, H, psi0, r, J, edges, P_cache, alpha, Delta, T, theta0=0.05
):
    psi = psi0.copy()
    chosen = []

    selection_time = 0.0
    graph_time = 0.0
    forward_time = 0.0

    for _ in range(T):
        t0 = time.perf_counter()

        e_star, t_graph, t_forward = select_gnn_edge(
            model=model,
            psi=psi,
            r=r,
            J=J,
            edges=edges,
            P_cache=P_cache,
            alpha=alpha,
            Delta=Delta,
        )

        selection_time += time.perf_counter() - t0
        graph_time += t_graph
        forward_time += t_forward

        chosen.append(e_star)

        # Diagnostic/upper-bound mode: oracle sign for selected GNN edge.
        signed_grad = commutator_expectation_signed(H, P_cache[e_star], psi)
        theta = theta0 * np.sign(signed_grad)
        if theta == 0:
            theta = theta0

        psi = apply_entangler(psi, P_cache[e_star], theta)
        psi /= np.linalg.norm(psi)

    return chosen, selection_time, graph_time, forward_time


def run_one(model, N, L, alpha, K, Delta, T, seed):
    J0 = 1.0

    rng = np.random.default_rng(seed)
    r = sample_positions_discrete(N, L, rng)
    J = couplings_from_positions(r, J0, alpha)
    edges = topK_edge_list(J, K=K)

    H = build_H_xxz_longrange(J, Delta)
    P_cache = [P_ij(N, i, j, Delta) for i, j in edges]

    psi0 = random_product_state(N, np.random.default_rng(seed + 10000))
    E0 = exact_ground_energy(H)

    adapt_edges, t_adapt = build_circuit_adapt_timed(H, psi0, P_cache, T)

    gnn_edges, t_gnn, t_graph, t_forward = build_circuit_gnn_timed(
        model=model,
        H=H,
        psi0=psi0,
        r=r,
        J=J,
        edges=edges,
        P_cache=P_cache,
        alpha=alpha,
        Delta=Delta,
        T=T,
    )

    E_adapt = vqe_refine(H, psi0, P_cache, adapt_edges)
    E_gnn = vqe_refine(H, psi0, P_cache, gnn_edges)

    return {
        "pool_size": len(edges),
        "t_adapt": t_adapt,
        "t_gnn": t_gnn,
        "t_graph": t_graph,
        "t_forward": t_forward,
        "dE_adapt": E_adapt - E0,
        "dE_gnn": E_gnn - E0,
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
    Delta = 0.0
    T = 6

    K_list = [3, 4, 5, 6]
    n_realizations = 5

    results = {
        "config": {
            "N": N,
            "L": L,
            "alpha": alpha,
            "Delta": Delta,
            "T": T,
            "K_list": K_list,
            "n_realizations": n_realizations,
        },
        "pool_size": [],
        "adapt_time": [],
        "gnn_time": [],
        "gnn_graph_time": [],
        "gnn_forward_time": [],
        "adapt_dE": [],
        "gnn_dE": [],
    }

    for K in K_list:
        print(f"\nK={K}")

        pool_sizes = []
        adapt_times = []
        gnn_times = []
        graph_times = []
        forward_times = []
        adapt_dEs = []
        gnn_dEs = []

        for seed in range(n_realizations):
            print(f"  realization {seed + 1}/{n_realizations}")

            out = run_one(
                model=model,
                N=N,
                L=L,
                alpha=alpha,
                K=K,
                Delta=Delta,
                T=T,
                seed=seed,
            )

            pool_sizes.append(out["pool_size"])
            adapt_times.append(out["t_adapt"])
            gnn_times.append(out["t_gnn"])
            graph_times.append(out["t_graph"])
            forward_times.append(out["t_forward"])
            adapt_dEs.append(out["dE_adapt"])
            gnn_dEs.append(out["dE_gnn"])

        pool_m = float(np.mean(pool_sizes))

        adapt_time_m, adapt_time_e = mean_err(adapt_times)
        gnn_time_m, gnn_time_e = mean_err(gnn_times)
        graph_time_m, graph_time_e = mean_err(graph_times)
        forward_time_m, forward_time_e = mean_err(forward_times)

        adapt_dE_m, adapt_dE_e = mean_err(adapt_dEs)
        gnn_dE_m, gnn_dE_e = mean_err(gnn_dEs)

        results["pool_size"].append(pool_m)
        results["adapt_time"].append({"mean": adapt_time_m, "err": adapt_time_e})
        results["gnn_time"].append({"mean": gnn_time_m, "err": gnn_time_e})
        results["gnn_graph_time"].append({"mean": graph_time_m, "err": graph_time_e})
        results["gnn_forward_time"].append({"mean": forward_time_m, "err": forward_time_e})
        results["adapt_dE"].append({"mean": adapt_dE_m, "err": adapt_dE_e})
        results["gnn_dE"].append({"mean": gnn_dE_m, "err": gnn_dE_e})

        print(
            f"  |E|={pool_m:.1f} | "
            f"t_adapt={adapt_time_m:.4f}s, "
            f"t_gnn={gnn_time_m:.4f}s "
            f"(graph={graph_time_m:.4f}s, forward={forward_time_m:.4f}s) | "
            f"dE_adapt={adapt_dE_m:.4f}, dE_gnn={gnn_dE_m:.4f}"
        )

    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    with open("outputs/results/pool_scaling_online.json", "w") as f:
        json.dump(results, f, indent=2)

    pool = np.array(results["pool_size"])

    adapt_t = np.array([x["mean"] for x in results["adapt_time"]])
    adapt_t_err = np.array([x["err"] for x in results["adapt_time"]])
    gnn_t = np.array([x["mean"] for x in results["gnn_time"]])
    gnn_t_err = np.array([x["err"] for x in results["gnn_time"]])

    adapt_e = np.array([x["mean"] for x in results["adapt_dE"]])
    adapt_e_err = np.array([x["err"] for x in results["adapt_dE"]])
    gnn_e = np.array([x["mean"] for x in results["gnn_dE"]])
    gnn_e_err = np.array([x["err"] for x in results["gnn_dE"]])

    fig, ax = plt.subplots(1, 2, figsize=(10, 3))

    ax[0].errorbar(pool, adapt_t, yerr=adapt_t_err, fmt="o-", capsize=3, label="ADAPT selection")
    ax[0].errorbar(pool, gnn_t, yerr=gnn_t_err, fmt="s-", capsize=3, label="GNN selection")
    ax[0].set_xlabel(r"Pool size $|E|$")
    ax[0].set_ylabel(r"Classical selection time (s)")
    ax[0].set_yscale("log")
    ax[0].legend(frameon=False)
    ax[0].grid(alpha=0.25)

    ax[1].errorbar(pool, adapt_e, yerr=adapt_e_err, fmt="o-", capsize=3, label="ADAPT-VQE")
    ax[1].errorbar(pool, gnn_e, yerr=gnn_e_err, fmt="s-", capsize=3, label="GNN-VQE")
    ax[1].set_xlabel(r"Pool size $|E|$")
    ax[1].set_ylabel(r"Energy error $\Delta E$")
    ax[1].legend(frameon=False)
    ax[1].grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig("figures/figure_pool_scaling.pdf", bbox_inches="tight")
    fig.savefig("figures/figure_pool_scaling.png", dpi=300, bbox_inches="tight")

    print("\nSaved figures/figure_pool_scaling.pdf")


if __name__ == "__main__":
    main()