# src/adaptive_gnn_vqe/data/xxz_generator.py

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict

from scipy.sparse import csr_matrix, kron
from scipy.sparse.linalg import expm_multiply

import torch
from torch_geometric.data import Data

import os

# ============================================================
# Sparse Pauli matrices
# ============================================================
I2 = csr_matrix(np.eye(2, dtype=complex))
X2 = csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
Y2 = csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
Z2 = csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))


# ============================================================
# N-qubit operator builders (sparse)
# ============================================================
def op_on_site(N: int, op2: csr_matrix, site: int) -> csr_matrix:
    out = None
    for k in range(N):
        factor = op2 if k == site else I2
        out = factor if out is None else kron(out, factor, format="csr")
    return out


def two_site_op(N: int, opA: csr_matrix, i: int, opB: csr_matrix, j: int) -> csr_matrix:
    if i == j:
        raise ValueError("i and j must differ")
    out = None
    for k in range(N):
        if k == i:
            factor = opA
        elif k == j:
            factor = opB
        else:
            factor = I2
        out = factor if out is None else kron(out, factor, format="csr")
    return out


def P_ij(N: int, i: int, j: int, Delta: float) -> csr_matrix:
    # Generator for the entangler you use:
    # P_ij = XiXj + YiYj + Delta ZiZj
    return (
        two_site_op(N, X2, i, X2, j)
        + two_site_op(N, Y2, i, Y2, j)
        + Delta * two_site_op(N, Z2, i, Z2, j)
    )


# ============================================================
# Geometry / couplings
# ============================================================
def sample_positions_discrete(N: int, L: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample N distinct integer positions from {0,...,L}, sorted.
    Min spacing >= 1 automatically because positions are unique integers.
    Returns float array for downstream math.
    """
    if N > L + 1:
        raise ValueError(f"N={N} cannot fit into L={L} with min distance 1.")
    r = rng.choice(np.arange(L + 1), size=N, replace=False)
    r.sort()
    return r.astype(float)


def couplings_from_positions(r: np.ndarray, J0: float, alpha: float) -> np.ndarray:
    N = len(r)
    J = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            dist = abs(r[i] - r[j])
            J[i, j] = J0 / (dist ** alpha)
            J[j, i] = J[i, j]
    return J


def topK_edge_list(J: np.ndarray, K: int) -> List[Tuple[int, int]]:
    """
    Sparsify: for each node keep K strongest couplings; symmetrize.
    Returns list of undirected edges (i<j).
    """
    N = J.shape[0]
    edges = set()
    for i in range(N):
        idx = np.argsort(-J[i])  # descending
        picked = []
        for j in idx:
            if j == i:
                continue
            picked.append(j)
            if len(picked) >= K:
                break
        for j in picked:
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))
    return sorted(list(edges))


# ============================================================
# Hamiltonian build
# ============================================================
def build_H_xxz_longrange(J: np.ndarray, Delta: float) -> csr_matrix:
    N = J.shape[0]
    dim = 2 ** N
    H = csr_matrix((dim, dim), dtype=complex)
    for i in range(N):
        for j in range(i + 1, N):
            Jij = J[i, j]
            if Jij == 0:
                continue
            H += Jij * (
                two_site_op(N, X2, i, X2, j)
                + two_site_op(N, Y2, i, Y2, j)
                + Delta * two_site_op(N, Z2, i, Z2, j)
            )
    return H


# ============================================================
# Static features
# ============================================================
def node_features_static(r: np.ndarray, J: np.ndarray) -> np.ndarray:
    N = len(r)
    x = r / (r.max() - r.min() + 1e-12)
    d = np.zeros(N)
    for i in range(N):
        left = r[i] - r[i - 1] if i - 1 >= 0 else np.inf
        right = r[i + 1] - r[i] if i + 1 < N else np.inf
        d[i] = min(left, right)
    d = d / (r.max() - r.min() + 1e-12)
    s = J.sum(axis=1)
    m = J.max(axis=1)
    return np.stack([x, d, s, m], axis=1).astype(np.float32)


def edge_rank_feature(J: np.ndarray, edges: List[Tuple[int, int]]) -> Dict[Tuple[int, int], float]:
    """
    Proxy: inverse of sum of local ranks at endpoints (bigger=more locally dominant).
    """
    N = J.shape[0]
    ranks = {}
    for (i, j) in edges:
        vals_i = np.sort(J[i][np.arange(N) != i])[::-1]
        vals_j = np.sort(J[j][np.arange(N) != j])[::-1]
        Jij = J[i, j]
        ri = (np.where(vals_i == Jij)[0][0] + 1) if np.any(vals_i == Jij) else (np.searchsorted(-vals_i, -Jij) + 1)
        rj = (np.where(vals_j == Jij)[0][0] + 1) if np.any(vals_j == Jij) else (np.searchsorted(-vals_j, -Jij) + 1)
        ranks[(i, j)] = float(1.0 / (ri + rj))
    return ranks


def edge_features_static(i: int, j: int, r: np.ndarray, J: np.ndarray, ranks: Dict[Tuple[int, int], float]) -> np.ndarray:
    dij = abs(r[i] - r[j])
    Jij = J[i, j]
    return np.array(
        [np.log(Jij + 1e-30), np.log(dij + 1e-30), ranks[(min(i, j), max(i, j))]],
        dtype=np.float32,
    )


# ============================================================
# State: random product state
# ============================================================
def random_product_state(N: int, rng: np.random.Generator) -> np.ndarray:
    psi = np.array([1.0 + 0.0j])
    for _ in range(N):
        theta = rng.uniform(0, np.pi)
        phi = rng.uniform(0, 2 * np.pi)
        v = np.array(
            [np.cos(theta / 2.0), np.exp(1j * phi) * np.sin(theta / 2.0)],
            dtype=complex,
        )
        psi = np.kron(psi, v)
    psi = psi / np.linalg.norm(psi)
    return psi



def weakly_tilted_product_state(N: int, rng: np.random.Generator, eps=0.1) -> np.ndarray:
    psi = np.array([1.0 + 0.0j])
    for _ in range(N):
        theta = eps
        phi = rng.uniform(0, 2*np.pi)
        v = np.array([
            np.cos(theta/2),
            np.exp(1j*phi)*np.sin(theta/2)
        ], dtype=complex)
        psi = np.kron(psi, v)
    return psi / np.linalg.norm(psi)

# ============================================================
# Expectation values using cached operators
# ============================================================
def expval_cached(psi: np.ndarray, O: csr_matrix) -> float:
    return float(np.real(np.vdot(psi, O.dot(psi))))


# ============================================================
# Greedy label: | i <psi| [H, P_ij] |psi> |
# ============================================================
def commutator_expectation_abs(H: csr_matrix, P: csr_matrix, psi: np.ndarray) -> float:
    HP_psi = H.dot(P.dot(psi))
    PH_psi = P.dot(H.dot(psi))
    val = np.vdot(psi, HP_psi - PH_psi)
    return float(np.abs(1j * val))


def commutator_expectation_signed(H: csr_matrix, P: csr_matrix, psi: np.ndarray) -> float:
    HP_psi = H.dot(P.dot(psi))
    PH_psi = P.dot(H.dot(psi))
    val = np.vdot(psi, HP_psi - PH_psi)
    return float(np.real(1j * val))

def apply_entangler(psi: np.ndarray, P: csr_matrix, theta: float) -> np.ndarray:
    A = (-1j * theta) * P
    return expm_multiply(A, psi)


# ============================================================
# Sample container
# ============================================================

@dataclass
class Sample:
    node_feat: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    label_edge: int
    step: int
    gvals: np.ndarray     # <-- add this (shape [E])
    g_signed: np.ndarray   # <-- add this (shape [E])

# ============================================================
# Main generator (cache friendly)
# ============================================================
def generate_realization_samples(
    N: int,
    L: int,
    J0: float,
    alpha: float,
    Delta: float,
    K: int,
    T: int,
    theta0: float,
    seed: int,
) -> List[Sample]:

    rng = np.random.default_rng(seed)

    # Geometry + couplings
    r = sample_positions_discrete(N, L, rng)
    J = couplings_from_positions(r, J0, alpha)

    edges = topK_edge_list(J, K=K)
    rng.shuffle(edges)
    J_edge = np.array([J[i, j] for (i, j) in edges], dtype=np.float32)

    # Build Hamiltonian once
    H = build_H_xxz_longrange(J, Delta=Delta)

    # init |psi0> = |0...0>
    #dim = 2**N
    #psi = np.zeros(dim, dtype=complex)
    #psi[0] = 1.0

    # Random product state
    #psi = random_product_state(N, rng)
    # Weakly tilted product state
    psi = weakly_tilted_product_state(N, rng, eps=0.5)
                                      
    # Static features (cache)
    v_static = node_features_static(r, J)  # [N,4] , x,d,s,m
    ranks = edge_rank_feature(J, edges)

    E = len(edges)
    edge_index = np.array(
        [[i for (i, j) in edges], [j for (i, j) in edges]],
        dtype=np.int64,
    )

    edge_attr_static = np.stack(
        [edge_features_static(i, j, r, J, ranks) for (i, j) in edges],
        axis=0,
    )  # [E,3]

    # ------------------------------------------------------------
    # CACHE OPERATORS (big speedup)
    # ------------------------------------------------------------
    Z_ops = [op_on_site(N, Z2, i) for i in range(N)]
    X_ops = [op_on_site(N, X2, i) for i in range(N)]
    ZZ_ops = [two_site_op(N, Z2, i, Z2, j) for (i, j) in edges]
    XX_ops = [two_site_op(N, X2, i, X2, j) for (i, j) in edges]
    YY_ops = [two_site_op(N, Y2, i, Y2, j) for (i, j) in edges]
    P_cache = [P_ij(N, i, j, Delta=Delta) for (i, j) in edges]

    samples: List[Sample] = []

    last_label = None

    for t in range(T):
        # state-dependent features...
        z = np.array([expval_cached(psi, Z_ops[i]) for i in range(N)], dtype=np.float32)
        x = np.array([expval_cached(psi, X_ops[i]) for i in range(N)], dtype=np.float32)
        v_state = np.stack([z, x], axis=1)
        v_t = np.concatenate([v_static, v_state], axis=1)

        alpha_feat = np.full((N, 1), alpha, dtype=np.float32)
        v_t = np.concatenate([v_t, alpha_feat], axis=1) 

        zz = np.array([expval_cached(psi, ZZ_ops[e]) for e in range(E)], dtype=np.float32)
        xx = np.array([expval_cached(psi, XX_ops[e]) for e in range(E)], dtype=np.float32)
        yy = np.array([expval_cached(psi, YY_ops[e]) for e in range(E)], dtype=np.float32)
        
        
        # local bond energy on each candidate edge
        bond_energy = J_edge * (xx + yy + Delta * zz)  # [E]

        # accumulate to nodes: h_i = sum_{edges incident to i} bond_energy(edge)
        h_node = np.zeros(N, dtype=np.float32)
        for e, (i, j) in enumerate(edges):
            h_node[i] += bond_energy[e]
            h_node[j] += bond_energy[e]

        # OPTIONAL: normalize by total incident coupling (precompute once if you want)
        # h_node = h_node / (J.sum(axis=1).astype(np.float32) + 1e-12)

        h_feat = h_node[:, None]  # [N,1]
        v_t = np.concatenate([v_t, h_feat], axis=1)
        
     

        # greedy label
        #gvals = np.array([commutator_expectation_abs(H, P_cache[e], psi) for e in range(E)])
        g_signed = np.array(
            [commutator_expectation_signed(H, P_cache[e], psi) for e in range(E)],
            dtype=np.float32,
        )

        gvals = np.abs(g_signed).astype(np.float32)

        p_mean = np.zeros(E, dtype=np.float32)
        p2_mean = np.zeros(E, dtype=np.float32)

        for e in range(E):
            Ppsi = P_cache[e].dot(psi)                          # |v> = P|psi>
            p_mean[e] = float(np.real(np.vdot(psi, Ppsi)))      # <psi|P|psi>
            p2_mean[e] = float(np.real(np.vdot(Ppsi, Ppsi)))    # <psi|P^2|psi> = ||P|psi>||^2

        p_var = p2_mean - p_mean**2
        p_var = np.maximum(p_var, 0.0).astype(np.float32)       # numerical safety


        edge_attr_t = np.concatenate(
                                [edge_attr_static, zz[:, None], xx[:, None], yy[:, None], p_var[:, None]],
                                 axis=1
                                )

        # tree-like restriction removed

        # no-repeat constraint
        #if last_label is not None:
        #    gvals[last_label] = -np.inf

        

        label = int(np.argmax(gvals))
        last_label = label


        samples.append(Sample(
            node_feat=v_t,
            edge_index=edge_index,
            edge_attr=edge_attr_t,
            label_edge=label,
            step=t,
            gvals=gvals.astype(np.float32),
            g_signed=g_signed.astype(np.float32),
        ))

        psi = apply_entangler(psi, P_cache[label], theta=theta0)
        psi = psi / np.linalg.norm(psi)

    return samples


# ============================================================
# Convert to PyG Data
# ============================================================
def sample_to_pyg(sample: Sample, realization_id: int, alpha: float) -> Data:
    data = Data(
        x=torch.tensor(sample.node_feat, dtype=torch.float32),
        edge_index=torch.tensor(sample.edge_index, dtype=torch.long),
        edge_attr=torch.tensor(sample.edge_attr, dtype=torch.float32),
        y=torch.tensor([sample.label_edge], dtype=torch.long),
        g=torch.tensor(sample.gvals, dtype=torch.float32),
        g_signed=torch.tensor(sample.g_signed, dtype=torch.float32),
    )
    data.realization_id = int(realization_id)
    data.step = int(sample.step)
    data.alpha = float(alpha)
    return data



# ============================================================
# Script entry point
# ============================================================
if __name__ == "__main__":
    from adaptive_gnn_vqe.data.builders import build_xxz_dataset


    Delta = 0.0
    J0 = 1.0
    theta0 = 0.05

    system_configs = [
    dict(N=8, L=80, T=4, K=6),
]

    alpha_list = [0.5, 1.0, 1.5, 2.0, 2.5]
    realizations_per_alpha = 500

    all_pyg_data = build_xxz_dataset(
        system_configs=system_configs,
        alpha_list=alpha_list,
        realizations_per_alpha=realizations_per_alpha,
        Delta=Delta,
        J0=J0,
        theta0=theta0,
    )

    save_path = os.path.join(
        os.path.dirname(__file__),
        "test_spinchain_dataset.pt"
    )

    torch.save(all_pyg_data, save_path)

    print("Total graph samples:", len(all_pyg_data))