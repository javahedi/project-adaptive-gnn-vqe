import os
import json
import numpy as np
import matplotlib.pyplot as plt


def main():
    log_path = "outputs/logs/train_log.json"

    with open(log_path, "r") as f:
        log = json.load(f)

    epochs = np.array(log["epoch"])

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    ax = axes.ravel()

    # (a) Accuracy
    ax[0].plot(epochs, log["train_acc"], label="Train")
    ax[0].plot(epochs, log["val_acc"], label="Validation")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].set_title("(a) Operator-selection accuracy")
    ax[0].legend()

    # (b) Mean rank
    ax[1].plot(epochs, log["train_rank"], label="Train")
    ax[1].plot(epochs, log["val_rank"], label="Validation")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Mean rank")
    ax[1].set_title("(b) Mean rank of correct operator")
    ax[1].legend()

    # (c) Baselines
    final_gnn = log["val_acc"][-1]
    # final_random = log.get("val_unif", [None])[-1]
    # final_strongest_J = log.get("val_J", [None])[-1]

    val_unif = log.get("val_unif", [])
    val_J = log.get("val_J", [])

    final_random = val_unif[-1] if len(val_unif) > 0 else 0.0
    final_strongest_J = val_J[-1] if len(val_J) > 0 else 0.0

    labels = ["Random", "Strongest J", "GNN"]
    values = [final_random, final_strongest_J, final_gnn]

    print("Final accuracies:")
    print(f"  Random baseline: {final_random:.3f}")
    print(f"  Strongest J baseline: {final_strongest_J:.3f}")
    print(f"  GNN: {final_gnn:.3f}")

    ax[2].bar(labels, values)
    ax[2].set_ylabel("Validation accuracy")
    ax[2].set_title("(c) Comparison with baselines")
    ax[2].set_ylim(0, 0.5)

    # (d) Per-alpha accuracy
    per_alpha_last = log["per_alpha"][-1]
    alphas = sorted(float(a) for a in per_alpha_last.keys())
    accs = [per_alpha_last[str(a)] if str(a) in per_alpha_last else per_alpha_last[a] for a in alphas]

    ax[3].plot(alphas, accs, marker="o")
    ax[3].set_xlabel(r"Interaction exponent $\alpha$")
    ax[3].set_ylabel("Validation accuracy")
    ax[3].set_title(r"(d) Accuracy vs $\alpha$")
    ax[3].set_ylim(0, 1)

    fig.tight_layout()

    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/figure_training_summary.pdf", bbox_inches="tight")
    fig.savefig("figures/figure_training_summary.png", dpi=300, bbox_inches="tight")

    print("Saved:")
    print("  figures/figure_training_summary.pdf")
    print("  figures/figure_training_summary.png")


if __name__ == "__main__":
    main()