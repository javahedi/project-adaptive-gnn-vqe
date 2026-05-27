import numpy as np
import torch
from torch_geometric.data import Data


from adaptive_gnn_vqe.data.xxz_generator import (
    generate_realization_samples,
    sample_to_pyg,
    node_features_static,
    edge_rank_feature,
    edge_features_static,
    op_on_site,
    two_site_op,
    expval_cached,
    X2,
    Y2,
    Z2,
)

def build_xxz_dataset(
    system_configs,
    alpha_list,
    realizations_per_alpha,
    Delta=0.0,
    J0=1.0,
    theta0=0.05,
):
    all_pyg_data = []
    rid = 0

    for cfg in system_configs:
        N = cfg["N"]
        L = cfg["L"]
        T = cfg["T"]
        K = cfg["K"]

        for alpha in alpha_list:
            for _ in range(realizations_per_alpha):
                samples = generate_realization_samples(
                    N=N,
                    L=L,
                    J0=J0,
                    alpha=alpha,
                    Delta=Delta,
                    K=K,
                    T=T,
                    theta0=theta0,
                    seed=rid,
                )

                for s in samples:
                    data = sample_to_pyg(s, realization_id=rid, alpha=alpha)
                    data.N = N
                    data.L = L
                    all_pyg_data.append(data)

                rid += 1

    return all_pyg_data


def build_xxz_graph_from_state(
    psi,
    r,
    J,
    edges,
    P_cache,
    alpha,
    Delta,
):
    """
    Build a PyG graph for online GNN inference during ADAPT-VQE.

    Inputs
    ------
    psi:
        Current quantum state.

    r:
        Qubit positions.

    J:
        Coupling matrix.

    edges:
        Candidate operator pool.

    P_cache:
        Cached entangler operators.

    alpha:
        Power-law interaction exponent.

    Delta:
        XXZ anisotropy parameter.
    """

    N = len(r)
    E = len(edges)

    v_static = node_features_static(r, J)
    ranks = edge_rank_feature(J, edges)

    edge_index = np.array(
        [[i for (i, j) in edges], [j for (i, j) in edges]],
        dtype=np.int64,
    )

    edge_attr_static = np.stack(
        [edge_features_static(i, j, r, J, ranks) for (i, j) in edges],
        axis=0,
    )

    Z_ops = [op_on_site(N, Z2, i) for i in range(N)]
    X_ops = [op_on_site(N, X2, i) for i in range(N)]

    z = np.array([expval_cached(psi, Z_ops[i]) for i in range(N)], dtype=np.float32)
    x = np.array([expval_cached(psi, X_ops[i]) for i in range(N)], dtype=np.float32)

    v_state = np.stack([z, x], axis=1)
    v_t = np.concatenate([v_static, v_state], axis=1)

    alpha_feat = np.full((N, 1), alpha, dtype=np.float32)
    v_t = np.concatenate([v_t, alpha_feat], axis=1)

    J_edge = np.array([J[i, j] for (i, j) in edges], dtype=np.float32)

    ZZ_ops = [two_site_op(N, Z2, i, Z2, j) for (i, j) in edges]
    XX_ops = [two_site_op(N, X2, i, X2, j) for (i, j) in edges]
    YY_ops = [two_site_op(N, Y2, i, Y2, j) for (i, j) in edges]

    zz = np.array([expval_cached(psi, ZZ_ops[e]) for e in range(E)], dtype=np.float32)
    xx = np.array([expval_cached(psi, XX_ops[e]) for e in range(E)], dtype=np.float32)
    yy = np.array([expval_cached(psi, YY_ops[e]) for e in range(E)], dtype=np.float32)

    bond_energy = J_edge * (xx + yy + Delta * zz)

    h_node = np.zeros(N, dtype=np.float32)
    for e, (i, j) in enumerate(edges):
        h_node[i] += bond_energy[e]
        h_node[j] += bond_energy[e]

    h_feat = h_node[:, None]
    v_t = np.concatenate([v_t, h_feat], axis=1)


    p_mean = np.zeros(E, dtype=np.float32)
    p2_mean = np.zeros(E, dtype=np.float32)

    for e in range(E):
        Ppsi = P_cache[e].dot(psi)
        p_mean[e] = float(np.real(np.vdot(psi, Ppsi)))
        p2_mean[e] = float(np.real(np.vdot(Ppsi, Ppsi)))

    p_var = p2_mean - p_mean**2
    p_var = np.maximum(p_var, 0.0).astype(np.float32)

    edge_attr_t = np.concatenate(
        [
            edge_attr_static,
            zz[:, None],
            xx[:, None],
            yy[:, None],
            p_var[:, None],
        ],
        axis=1,
    )

    data = Data(
        x=torch.tensor(v_t, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr_t, dtype=torch.float32),
    )

    return data

    #raise NotImplementedError("Static graph construction works; state features next.")