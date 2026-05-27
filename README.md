# Adaptive GNN-VQE

Graph neural network policies for accelerating ADAPT-VQE operator selection.

This repository explores learning-based operator selection strategies for variational quantum eigensolvers (VQE), with an initial focus on long-range XXZ spin-chain systems and future extensions to molecular Hamiltonians.

---

# Features

- Long-range XXZ spin-chain dataset generation
- Pointer-style graph neural network for operator ranking
- ADAPT-VQE inspired training targets
- Modular training and evaluation pipeline
- Clean package structure for future extensions

---

# Repository Structure

```text
project-adaptive-gnn-vqe/
├── configs/
├── docs/
├── notebooks/
├── outputs/
├── scripts/
├── src/
│   └── adaptive_gnn_vqe/
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── utils/
│       └── vqe/
└── tests/
```

---

# Installation

Clone the repository:

```bash
git clone <repo-url>
cd project-adaptive-gnn-vqe
```

Create environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install package:

```bash
pip install -e .
```

---

# Generate Dataset

Run:

```bash
python src/adaptive_gnn_vqe/data/xxz_generator.py
```

Generated datasets are saved inside:

```text
src/adaptive_gnn_vqe/data/
```

---

# Train the GNN

Run:

```bash
python scripts/train_xxz.py
```

Outputs:
- trained models → `outputs/models/`
- logs → `outputs/logs/`

---

# Compare ADAPT vs GNN

Run:

```bash
python scripts/compare_adapt_gnn.py
```

---

# Current Status

The repository currently supports:
- XXZ spin-chain systems
- Pointer GNN training
- ADAPT-style operator selection

Planned future extensions:
- molecular Hamiltonians
- generalized operator pools
- online GNN-guided ADAPT-VQE

---

# Contributors

- Javad Vahedi
- Collaborators : Hadi Hassanian Arefi

---

# License

MIT License