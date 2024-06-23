import pytest

from pydiffuser.models import CONFIG_REGISTRY, MODEL_REGISTRY


@pytest.fixture
def config_dict():
    return CONFIG_REGISTRY


@pytest.fixture
def model_dict():
    return MODEL_REGISTRY
