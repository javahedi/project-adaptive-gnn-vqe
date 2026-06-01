import numpy as np

from adaptive_gnn_vqe.data.xxz_generator import (
    sample_positions_discrete,
    couplings_from_positions,
    topK_edge_list,
    build_H_xxz_longrange,
    random_product_state,
    P_ij,
    apply_entangler,
)


def energy(H, psi):
    return float(np.real(np.vdot(psi, H.dot(psi))))


def signed_gradient(H, P, psi):
    HP_psi = H.dot(P.dot(psi))
    PH_psi = P.dot(H.dot(psi))
    val = np.vdot(psi, HP_psi - PH_psi)
    return float(np.real(1j * val))


def main():
    N = 8
    L = 80
    alpha = 1.5
    J0 = 1.0
    Delta = 0.0
    K = 6
    theta = 1e-3
    seed = 0

    rng = np.random.default_rng(seed)

    r = sample_positions_discrete(N, L, rng)
    J = couplings_from_positions(r, J0, alpha)
    edges = topK_edge_list(J, K)

    H = build_H_xxz_longrange(J, Delta)
    psi = random_product_state(N, np.random.default_rng(seed + 10000))

    E0 = energy(H, psi)

    print("Gradient sign diagnostic")
    print("------------------------")
    print(f"E0 = {E0:.12f}")
    print()

    for e, (i, j) in enumerate(edges[:10]):
        P = P_ij(N, i, j, Delta)

        g = signed_gradient(H, P, psi)

        psi_plus = apply_entangler(psi, P, +theta)
        psi_plus /= np.linalg.norm(psi_plus)

        psi_minus = apply_entangler(psi, P, -theta)
        psi_minus /= np.linalg.norm(psi_minus)

        Eplus = energy(H, psi_plus)
        Eminus = energy(H, psi_minus)

        print(f"edge {e:02d} ({i},{j})")
        print(f"  signed gradient g     = {g:+.8e}")
        print(f"  E(+theta) - E0        = {Eplus - E0:+.8e}")
        print(f"  E(-theta) - E0        = {Eminus - E0:+.8e}")

        if Eplus < E0 and Eminus > E0:
            print("  downhill direction    = +theta")
        elif Eminus < E0 and Eplus > E0:
            print("  downhill direction    = -theta")
        elif Eplus < E0 and Eminus < E0:
            print("  downhill direction    = both decrease")
        else:
            print("  downhill direction    = neither decreases")

        print()


if __name__ == "__main__":
    main()