from adaptive_gnn_vqe.models import PointerGNN
from adaptive_gnn_vqe.vqe import run_one_realization
from adaptive_gnn_vqe.training.losses import pointer_soft_loss


def test_imports():
    assert PointerGNN is not None
    assert run_one_realization is not None
    assert pointer_soft_loss is not None