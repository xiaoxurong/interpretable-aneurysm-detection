import pytest
import torch

from pointcbm.backbones import available_backbones, build_backbone
from pointcbm.models import BlackBoxClassifier, ConceptBottleneck3D, build_model, pool_concepts


@pytest.mark.parametrize(
    "name,N,kw",
    [("PointNetLite", 128, {}), ("PointNet", 128, {}), ("DGCNN", 128, {"k": 8}), ("PointNetPlusPlus", 512, {})],
)
def test_backbone_contract(name, N, kw):
    """Every backbone maps (B, C_in, N) -> (B, N, d_f) per-point features."""
    bb = build_backbone(name, d_f=32, in_channel=6, **kw)
    out = bb(torch.randn(2, 6, N))
    assert out.shape == (2, N, 32)
    assert torch.isfinite(out).all()


def test_registry_rejects_unknown_backbone():
    with pytest.raises(ValueError, match="Available"):
        build_backbone("NotARealNet")
    assert "OfficialPointNet" in available_backbones()  # legacy alias kept


def _cbm(concept_bank, pooling="topk_lse"):
    bb = build_backbone("PointNetLite", d_f=32, in_channel=6, width=16)
    return ConceptBottleneck3D(bb, d_f=32, d_t=concept_bank.shape[1], d_hidden=16, num_concepts=6, num_classes=2, pooling=pooling)


def test_cbm_outputs(concept_bank):
    model = _cbm(concept_bank).eval()
    logits, C, A = model(torch.randn(4, 256, 6), concept_bank)
    assert logits.shape == (4, 2) and C.shape == (4, 6) and A.shape == (4, 256, 6)
    assert (A >= 0).all() and (A <= 1).all()
    # logits are *exactly* a linear function of the concept scores (eval mode, no dropout)
    torch.testing.assert_close(logits, model.predict_from_concepts(C))


def test_pooling_ordering():
    A = torch.rand(3, 100, 5)
    beta = torch.tensor(10.0)
    mean, mx = pool_concepts(A, "mean", beta), pool_concepts(A, "max", beta)
    lse = pool_concepts(A, "lse", beta)
    assert (mean <= mx + 1e-6).all()
    assert (lse >= mx - 1e-6).all()  # log-sum-exp is a soft upper bound on the max
    assert (lse <= mx + torch.log(torch.tensor(100.0)) / beta + 1e-6).all()


def test_all_parameters_receive_gradients(concept_bank):
    model = _cbm(concept_bank).train()
    logits, C, A = model(torch.randn(4, 128, 6), concept_bank)
    (logits.sum() + C.sum()).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing


def test_blackbox_head():
    bb = build_backbone("PointNetLite", d_f=32, in_channel=6, width=16)
    logits, C, A = BlackBoxClassifier(bb, d_f=32, d_hidden=16, num_classes=2)(torch.randn(3, 64, 6))
    assert logits.shape == (3, 2) and C is None and A is None


def test_legacy_checkpoint_parameter_names():
    """Checkpoints from the original course-project code must still load."""
    cfg = {"model": {"backbone": "OfficialPointNet", "d_f": 128, "d_hidden": 256, "in_channel": 6}}
    keys = set(build_model(cfg, d_t=768, num_concepts=6).state_dict())
    for k in [
        "backbone.stn.fc3.weight",
        "backbone.fstn.conv1.weight",
        "backbone.proj.0.weight",
        "feat_norm.weight",
        "ccam.proj_features.weight",
        "ccam.proj_text.weight",
        "predictor.weight",
        "beta",
    ]:
        assert k in keys


def test_tiny_overfit(concept_bank):
    """Sanity check: the full CBM can memorise 8 labelled clouds."""
    torch.manual_seed(0)
    X = torch.randn(8, 128, 6)
    y = torch.tensor([0, 1] * 4)
    X[y == 1, :, :3] *= 2.0  # make classes separable by scale
    model = _cbm(concept_bank)
    model.dropout.p = 0.0
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(150):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(X, concept_bank)[0], y)
        loss.backward()
        opt.step()
    model.eval()
    acc = (model(X, concept_bank)[0].argmax(1) == y).float().mean()
    assert acc == 1.0, f"failed to overfit (acc={acc:.2f}, loss={loss.item():.3f})"
