from inspect import signature

import pytest

from pydiffuser.models.core.base import MODEL_KWARGS


@pytest.mark.parametrize(
    "name",
    [
        ("abp"),
        ("aoup"),
        ("bm"),
        ("levy"),
        ("mips"),
        ("rtp"),
        ("smoluchowski"),
        ("vicsek"),
    ],
)
def test_models(config_dict, model_dict, name):
    config = config_dict[name]()
    model = model_dict[name].from_config(config=config)  # instantiation from config
    ens = model.generate()
    shape = (config.realization, config.length, config.dimension)
    assert ens.microstate.shape == shape

    ens = model.generate(realization=5, length=50, dimension=2)
    with pytest.raises(AssertionError):
        assert ens.microstate.shape == (5, 50, 2)
    assert ens.microstate.shape == shape

    kwargs = config_dict[name].show_fields()
    params = list(signature(model_dict[name].__init__).parameters.keys())
    args = (
        [kwargs[param] for param in params[1:]]
        if MODEL_KWARGS not in params
        else ([kwargs[param] for param in params[1:-1]])
    )
    model_v2 = model_dict[name](*args)  # instantiation without config

    ens_v2 = model_v2.generate(realization=5, length=50, dimension=2)
    assert ens_v2.microstate.shape == (5, 50, 2)
