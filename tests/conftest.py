import os

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "processed_IntrA_6d")


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def concept_bank():
    E = torch.randn(6, 32)
    return E / E.norm(dim=-1, keepdim=True)


requires_data = pytest.mark.skipif(not os.path.isdir(os.path.join(DATA, "points")), reason="IntrA data not present")
