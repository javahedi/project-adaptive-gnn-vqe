from adaptive_gnn_vqe.data.xxz_generator import generate_realization_samples, sample_to_pyg


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


def build_xxz_graph_from_state(*args, **kwargs):
    raise NotImplementedError(
        "Build a PyG graph from the current quantum state psi. "
        "This will be used for online GNN-guided ADAPT selection."
    )