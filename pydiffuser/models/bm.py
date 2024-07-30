import jax.numpy as jnp
from scipy.stats import norm

from pydiffuser.models.core import BaseDiffusion
from pydiffuser.tracer import Ensemble, Trajectory
from pydiffuser.utils import BaseDiffusionConfig, helpers, jitted


class BrownianMotionConfig(BaseDiffusionConfig):
    name: str = "bm"

    def __init__(self, diffusivity: float = 1.0, **kwargs):
        super(BrownianMotionConfig, self).__init__(**kwargs)
        self.diffusivity = diffusivity


class BrownianMotion(BaseDiffusion):
    name: str = "bm"

    def __init__(self, diffusivity: float = 1.0):
        super(BrownianMotion, self).__init__()
        self.diffusivity = diffusivity

    @property
    def std(self) -> float:
        return float(jnp.sqrt(2 * self.diffusivity))

    def generate(
        self,
        realization: int = 10,
        length: int = 1000,
        dimension: int = 2,
        dt: float = 1.0,
        **generate_kwargs,
    ) -> Ensemble:
        ens = super().generate(realization, length, dimension, dt, **generate_kwargs)
        realization, length, dimension, dt = self.generate_info.values()

        x = self._get_initial_position()
        noise = jitted.get_noise(
            generator=norm.rvs,
            size=realization * (length - 1) * dimension,
            shape=(realization, -1, dimension),
        )
        dx = noise * self.std * jnp.sqrt(dt)
        x = jnp.concatenate((x, dx), axis=1)
        x = jnp.cumsum(x, axis=1)
        ens.update_microstate(microstate=x)
        return ens

    @helpers.deprecated
    def create(
        self, realization: int, length: int, dimension: int, dt: float
    ) -> Ensemble:
        import numpy as np

        ens = Ensemble(dt=dt)
        for id in range(realization):
            x = np.zeros((1, dimension))  # init
            for _ in range(1, length):
                noise = norm.rvs(size=dimension)
                dx = noise * self.std * np.sqrt(dt)
                next_x = x[-1] + dx
                x = np.vstack((x, next_x))
            ens[id] = Trajectory(dt, *x.T)
        return ens
