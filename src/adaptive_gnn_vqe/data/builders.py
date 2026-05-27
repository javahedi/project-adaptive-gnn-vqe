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

    raise NotImplementedError("Static graph construction works; state features next.")