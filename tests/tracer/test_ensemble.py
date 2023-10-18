import jax.numpy as jnp
import pytest

from pydiffuser.tracer.ensemble import Ensemble
from pydiffuser.tracer.trajectory import Trajectory


def test_ensemble():
    ens = Ensemble(dt=1.0)

    tracer = Trajectory(dt=0.1, position_x1=[0, 1, 2], position_x2=[0, -1, -2])
    with pytest.raises(AssertionError):
        ens.add(tracer)

    tracer = Trajectory(dt=1.0, position_x1=[0, 1, 2], position_x2=[0, -1, -2])
    ens.add(tracer)
    microstate = jnp.array([[[3, -3], [4, -4], [5, -5]]])
    ens.update_microstate(microstate)
