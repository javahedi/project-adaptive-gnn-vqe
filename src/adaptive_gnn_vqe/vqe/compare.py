# adapt_vs_gnn_vqe.py
import time
import json
import numpy as np
import torch

from scipy.optimize import minimize

from adaptive_gnn_vqe.data.xxz_generator import (
    sample_positions_discrete,
    couplings_from_positions,
    topK_edge_list,
    build_H_xxz_longrange,
    random_product_state,
    P_ij,
    apply_entangler,
    commutator_expectation_abs,
)

from adaptive_gnn_vqe.models import PointerGNN


import os

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "../../../outputs/models/pointer_gnn_best_acc.pt"
    )
)


# -----------------------------
# Helpers
# -----------------------------
def energy(H, psi):
    return float(np.real(np.vdot(psi, H.dot(psi))))

def make_pool(N, L, alpha, K, J0=1.0, seed=0):
    rng = np.random.default_rng(seed)
    r = sample_positions_discrete(N, L, rng)
    J = couplings_from_positions(r, J0, alpha)
    edges = topK_edge_list(J, K=K)  # pool = sparsified interaction graph
    return r, J, edges

def build_cached_P(N, edges, Delta):
    return [P_ij(N, i, j, Delta) for (i, j) in edges]


# -----------------------------
# Circuit builders
# -----------------------------
def build_circuit_adapt(H, psi0, edges, P_cache, T):
    """ADAPT-like: at each step scan all pool operators via commutator magnitude."""
    psi = psi0.copy()
    chosen = []

    for t in range(T):
        gvals = np.array([commutator_expectation_abs(H, P_cache[e], psi) for e in range(len(edges))])
        e_star = int(np.argmax(gvals))
        chosen.append(e_star)

        # apply a tiny update just to move state for next selection (like your dataset)
        psi = apply_entangler(psi, P_cache[e_star], theta=0.05)
        psi = psi / np.linalg.norm(psi)

    return chosen

def build_circuit_gnn(model, data_builder_fn, psi0, edges, P_cache, T):
    """GNN policy: at each step forward pass, pick argmax edge."""
    psi = psi0.copy()
    chosen = []

    for t in range(T):
        data = data_builder_fn(psi, t)  # user supplies how to build features for this state
        with torch.no_grad():
            scores = model(data)
        e_star = int(torch.argmax(scores).item())
        chosen.append(e_star)

        psi = apply_entangler(psi, P_cache[e_star], theta=0.05)
        psi = psi / np.linalg.norm(psi)

    return chosen


# -----------------------------
# VQE refinement for a fixed circuit topology
# -----------------------------
def vqe_refine(H, psi0, edges, P_cache, chosen_edges, theta_init=None, method="Nelder-Mead", maxiter=200):
    T = len(chosen_edges)
    if theta_init is None:
        theta_init = np.zeros(T, dtype=float)

    def objective(thetas):
        psi = psi0
        for k, eidx in enumerate(chosen_edges):
            psi = apply_entangler(psi, P_cache[eidx], float(thetas[k]))
            psi = psi / np.linalg.norm(psi)
        return energy(H, psi)

    res = minimize(
        objective,
        theta_init,
        method=method,
        options=dict(maxiter=maxiter, disp=False)
    )
    return float(res.fun), res.x, res.nit, res.success


# -----------------------------
# Main experiment for one realization
# -----------------------------
def run_one_realization(model, N=8, L=80, alpha=1.5, Delta=0.0, K=6, T=6, seed=0):
    # Build pool + Hamiltonian
    r, J, edges = make_pool(N, L, alpha, K, seed=seed)
    H = build_H_xxz_longrange(J, Delta)

    # Initial state
    psi0 = random_product_state(N, np.random.default_rng(seed))

    # Cache operators
    P_cache = build_cached_P(N, edges, Delta)

    # --- ADAPT selection ---
    t0 = time.time()
    adapt_edges = build_circuit_adapt(H, psi0, edges, P_cache, T)
    t_adapt_select = time.time() - t0

    # --- GNN selection ---
    # You MUST provide a feature builder for the current state.
    # If you already have generate_realization_samples that outputs PyG Data,
    # then easiest: reuse that pipeline instead of rebuilding features here.
    #
    # For now this is a placeholder stub.
    def data_builder_stub(psi, t):
        raise RuntimeError(
            "Implement data_builder_fn(psi,t) to build PyG Data features for the GNN at inference.\n"
            "Tip: reuse your generate_realization_samples(...) step t features."
        )

    t0 = time.time()
    # gnn_edges = build_circuit_gnn(model, data_builder_stub, psi0, edges, P_cache, T)
    # comment out until you plug in your builder
    t_gnn_select = time.time() - t0
    gnn_edges = None

    # --- VQE refinement (same optimizer for both) ---
    E_adapt, thetas_adapt, nit_a, ok_a = vqe_refine(H, psi0, edges, P_cache, adapt_edges)

    out = dict(
        N=N, L=L, alpha=alpha, K=K, T=T, seed=seed,
        adapt_edges=[tuple(map(int, edges[e])) for e in adapt_edges],
        E_adapt=E_adapt,
        adapt_select_time=t_adapt_select,
        vqe_iters_adapt=int(nit_a),
        vqe_ok_adapt=bool(ok_a),
    )

    if gnn_edges is not None:
        E_gnn, thetas_gnn, nit_g, ok_g = vqe_refine(H, psi0, edges, P_cache, gnn_edges)
        overlap = np.mean([int(a == b) for a, b in zip(adapt_edges, gnn_edges)])
        out.update(dict(
            gnn_edges=[tuple(map(int, edges[e])) for e in gnn_edges],
            E_gnn=E_gnn,
            gnn_select_time=t_gnn_select,
            vqe_iters_gnn=int(nit_g),
            vqe_ok_gnn=bool(ok_g),
            overlap=float(overlap),
        ))

    return out


if __name__ == "__main__":
    device = torch.device("cpu")

    # Load trained model (make sure node_in/edge_in match your dataset)
    model = PointerGNN(node_in=8, edge_in=7, hidden=96, mp_layers=5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    model.to(device)
    model.eval()

    result = run_one_realization(model, N=8, L=80, alpha=1.5, K=6, T=6, seed=0)

    with open("adapt_vs_gnn_one_realization.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
