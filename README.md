<p align="center">
    <img src=https://github.com/jung235/pydiffuser/assets/96967431/a9f54ef6-d1d4-4fcf-88bf-8c336dd671c0 width="30%">
    <h1 align="center">Pydiffuser</h1>
</p>

[![pypi](https://img.shields.io/badge/pypi-v0.0.3-blue)](https://pypi.org/project/pydiffuser/)
[![python](https://img.shields.io/badge/python-3.10_|_3.11_|_3.12-blue)](https://pypi.org/project/pydiffuser/)
[![doi](https://zenodo.org/badge/703392021.svg)](https://zenodo.org/doi/10.5281/zenodo.10017027)
[![codecov](https://codecov.io/gh/jung235/pydiffuser/graph/badge.svg?token=UAT5VEBK0O)](https://codecov.io/gh/jung235/pydiffuser)
[![ci](https://github.com/jung235/pydiffuser/actions/workflows/ci.yml/badge.svg)](https://github.com/jung235/pydiffuser/actions/workflows/ci.yml)
[![docs](https://img.shields.io/badge/docs-dev-black)](https://github.com/jung235/pydiffuser/blob/main/README.md)
[![status](https://img.shields.io/badge/status-alpha-blueviolet)](https://github.com/jung235/pydiffuser)

Pydiffuser is a numerical simulation framework for nonequilibrium statistical physics based on [JAX](https://github.com/google/jax).

This package mainly aims:
- to share code to implement a numerical simulation on physical models written in various forms of [stochastic differential equations](https://en.wikipedia.org/wiki/Stochastic_differential_equation).
- to revisit recent research highlights in nonequilibrium statistical physics.
- to reduce the repeated code on time-series data analysis, e.g., statistical analysis of [single-particle trajectory](https://en.wikipedia.org/wiki/Single-particle_trajectory) for [SPT](https://en.wikipedia.org/wiki/Single-particle_tracking) experiments.
- to provide the skeleton of stochastic modeling for anyone interested in stochastic processes.

## Installation

### Requirements

Python 3.10+, [`jax>=0.4.18`](https://pypi.org/project/jax/), and [`jaxlib>=0.4.18`](https://pypi.org/project/jaxlib/).

### From [PyPI](https://pypi.org/project/pydiffuser/)

```console
$ pip install pydiffuser
```

If properly installed, you can run:

```console
$ pydiffuser --version
pydiffuser, version 0.0.3
```

## Quickstart

Pydiffuser provides various stochastic models that implement a numerical simulation based on the [Monte Carlo method](https://en.wikipedia.org/wiki/Monte_Carlo_method).
All Pydiffuser's models inherit an abstract class `pydiffuser.models.BaseDiffusion` and initiate the simulation after a method `generate` is called.
For the simplest case, you can produce a non-interacting [Brownian motion](https://en.wikipedia.org/wiki/Brownian_motion) at low Reynolds numbers as follows:

```python
from pydiffuser.models import BrownianMotion
from pydiffuser.tracer import Ensemble, Trajectory


model = BrownianMotion()
ensemble: Ensemble = model.generate()
tracer: Trajectory = ensemble[0]  # the 0th particle
```

Relevant stochastic observables, such as [mean-squared displacement](https://en.wikipedia.org/wiki/Mean_squared_displacement) and normalized [velocity autocorrelation function](https://en.wikipedia.org/wiki/Autocorrelation), can be calculated through the [methods](#observables) of `Trajectory` and `Ensemble`.

```python
tamsd = tracer.get_mean_squared_displacement(lagtime=1, rolling=True)
eamsd = ensemble.get_mean_squared_displacement(lagtime=1, rolling=False)
eatamsd = ensemble.get_mean_squared_displacement(lagtime=1, rolling=True)
```

You can visualize the trajectory using [matplotlib](https://github.com/matplotlib/matplotlib):

<p align="center" width="100%">
    <img width="25%" src=https://github.com/jung235/pydiffuser/assets/96967431/544f3d2f-1b51-46f6-8036-ee7f296fddad>
</p>

It is obtained by `matplotlib.pyplot.plot(tracer.position_x1, tracer.position_x2)`.

## CLI

List all stochastic models supported by Pydiffuser.

```console
$ pydiffuser model list
NAME            MODEL                           CONFIG                          DIMENSION       
abp             ActiveBrownianParticle          ActiveBrownianParticleConfig    2d              
aoup            ActiveOUParticle                ActiveOUParticleConfig          1d, 2d, 3d      
bm              BrownianMotion                  BrownianMotionConfig            1d, 2d, 3d      
levy            LevyWalk                        LevyWalkConfig                  1d, 2d, 3d      
mips            PhaseSeparation                 PhaseSeparationConfig           1d, 2d, 3d      
rtp             RunAndTumbleParticle            RunAndTumbleParticleConfig      1d, 2d, 3d      
smoluchowski    SmoluchowskiEquation            SmoluchowskiEquationConfig      1d, 2d          
vicsek          VicsekModel                     VicsekModelConfig               2d              
```

## Features

### How fast is it?

When generating $N$ realizations consisting of $L$ footprints, we have:

```julia
═════════════════════════════════════════════════════════════════════════════════════════════════
Model               Method              Running time [s] for N x L =                             
                                        10² x 10⁵          10³ x 10⁴          10⁴ x 10³          
─────────────────────────────────────────────────────────────────────────────────────────────────
`loop` [*]                              3.62 (0.19)        3.45 (0.23)        3.37 (0.21)        
─────────────────────────────────────────────────────────────────────────────────────────────────
`abp`               `generate`          1.95 (0.14)        1.74 (0.12)        1.59 (0.11)        
`aoup`              `generate`          1.61 (0.08)        1.61 (0.15)        1.55 (0.09)        
`bm`                `generate`          1.45 (0.11)        1.46 (0.13)        1.46 (0.14)        
`smoluchowski`      `generate`          1.71 (0.12)        1.67 (0.15)        1.64 (0.13)        
─────────────────────────────────────────────────────────────────────────────────────────────────
`bm`                `create`            1440.72 (158.16)   964.90 (83.06)     1195.41 (94.28)    
═════════════════════════════════════════════════════════════════════════════════════════════════
```

<details>
<summary>[*]</summary>

```python
def loop(N: int, L: int) -> float:
    """Even the most straightforward loop requires over 3 [s] for all (N, L) conditions.
    """

    t1 = time.time()

    xes = []
    for _ in range(N):
        x = []
        for _ in range(1, L):
            x.append([])
        xes.append(x)

    t2 = time.time()
    return t2 - t1
```
</details>

The represented running time is a mean $\mu$ (standard deviation $\sigma$) of five trials.

### Observables

*class* `pydiffuser.tracer.Trajectory` ∈ *class* `pydiffuser.tracer.Ensemble`

- `get_increments`
- `get_displacement_moment`
- `get_mean_squared_displacement`
- `get_cosine_moment`
- `get_velocity_autocorrelation`
- `get_real_time`

The above methods are defined in both `Trajectory` and `Ensemble` to enhance transparency.
Using `Trajectory`, the statistical analysis of [single-particle trajectory](https://en.wikipedia.org/wiki/Single-particle_trajectory) can be accelerated.

### Configuration

We introduce a configuration to deal with extensive parameter manipulation.
For instance, see [`config.json`](https://github.com/jung235/pydiffuser/blob/main/docs/features/configs/config.json), which contains all parameters demanded to instantiate `pydiffuser.ActiveBrownianParticle`.
Every JSON of the configurations listed in [CLI](#cli) can be obtained as follows:

```python
import pydiffuser as pyd
from pydiffuser.models import ActiveBrownianParticle, ActiveBrownianParticleConfig


config = ActiveBrownianParticleConfig()
config.to_json(json_path=<JSON_PATH>)
```

We suggest a research pipeline:

```python
┌────┐     ┌─────────────────────┐     ┌───────────────┐     ┌──────────┐     ┌────────────┐
│JSON├──>──┤`BaseDiffusionConfig`├──>──┤`BaseDiffusion`├──>──┤`Ensemble`├──>──┤NPY | PICKLE│
└────┘ [1] └─────────────────────┘ [2] └───────────────┘ [3] └──────────┘ [4] └────────────┘
```

It can be automized as follows:

```python
config = ActiveBrownianParticleConfig.from_json(json_path=<JSON_PATH>)  # [1]
model = ActiveBrownianParticle.from_config(config=config)  # [2]
ensemble = model.generate()  # [3]
ensemble.to_npy(npy_path=<NPY_PATH>)  # [4]
```

You can save and load any picklable object through `pydiffuser.save` and `pydiffuser.load`.

```python
MODEL_PATH = "model.pickle"


pyd.save(obj=model, pickle_path=MODEL_PATH)  # Here, <PICKLE_PATH> = MODEL_PATH
model = pyd.load(pickle_path=MODEL_PATH)
```

## Related Works

[Hyperdiffusion of Poissonian run-and-tumble particles in two dimensions](https://arxiv.org/abs/2308.00554)

## License

[Apache License 2.0](https://github.com/jung235/pydiffuser/blob/main/LICENSE)

## Citation

```bibtex
@misc{jung2023pydiffuser,
  title = {Pydiffuser: a simulation framework for nonequilibrium statistical physics},
  author = {Jung, Yurim},
  year = {2023},
  note = {doi: 10.5281/zenodo.10017027},
}
```
