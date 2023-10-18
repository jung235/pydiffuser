import pytest

from pydiffuser.models.bm import BrownianMotion, BrownianMotionConfig


def test_bm():
    model = BrownianMotion()
    model.generate(dimension=3)
    _, _, dim, _ = model.generate_info.values()
    assert dim == 3

    config = BrownianMotionConfig(dimension=1)
    assert config.name == model.name

    model_v2 = BrownianMotion.from_config(config)
    model_v2.generate(dimension=3)
    _, _, dim, _ = model_v2.generate_info.values()
    with pytest.raises(AssertionError):
        assert dim == 3
    assert dim == 1
