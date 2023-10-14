<p align="center">
    <img src=docs/overrides/.icons/pydiffuser_logo_small.png width="30%">
    <h1 align="center">Pydiffuser</h1>
</p>

[![pypi](https://img.shields.io/badge/pypi-v0.0.1-blue)](https://pypi.org/)
[![python](https://img.shields.io/badge/python-3.10_|_3.11_|_3.12-blue)](https://pypi.org/)
[![doi](https://img.shields.io/badge/DOI--blue)](https://zenodo.org/)
[![codecov](https://codecov.io/gh/jung235/pydiffuser/graph/badge.svg?token=XXXXXXXXXX)](https://codecov.io/gh/jung235/pydiffuser)
[![ci](https://github.com/jung235/pydiffuser/actions/workflows/ci.yml/badge.svg)](https://github.com/jung235/pydiffuser)
[![docs](https://img.shields.io/badge/docs-dev-black)](https://github.com/jung235/pydiffuser)
[![status](https://img.shields.io/badge/status-alpha-blueviolet)](https://github.com/jung235/pydiffuser)

Pydiffuser is a numerical simulation framework for nonequilibrium statistical physics based on [JAX](https://github.com/google/jax).

**This package mainly aims:**
- to share code to implement a numerical simulation on physical models written in various forms of [stochastic differential equations](https://en.wikipedia.org/wiki/Stochastic_differential_equation).
- to revisit recent research highlights in non-equilibrium statistical physics.
- to reduce the repeated code on time-series data analysis, e.g., statistical analysis of [single-particle trajectory](https://en.wikipedia.org/wiki/Single-particle_trajectory) for [SPT](https://en.wikipedia.org/wiki/Single-particle_tracking) experiments.
- to provide the skeleton of stochastic model simulation for anyone interested in stochastic processes.

## Installation

### Requirements

Python 3.10+, [`jax>=0.4.18`](https://pypi.org/project/jax/), and [`jaxlib>=0.4.18`](https://pypi.org/project/jaxlib/).

### From [PyPI](https://pypi.org)

```console
$ pip install pydiffuser
```

If properly installed, you can run:

```console
$ pydiffuser --version
pydiffuser, version 0.0.1
```

### From source

```console
$ git clone https://github.com/jung235/pydiffuser.git
$ cd pydiffuser
$ pip install .
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
tracer: Trajectory = ensemble[0]  # 0th particle
```

Relevant stochastic observables, such as [mean-squared displacement](https://en.wikipedia.org/wiki/Mean_squared_displacement) $\left \langle \mathbf{r}^{2}(t) \right \rangle$ and normalized [velocity autocorrelation function](https://en.wikipedia.org/wiki/Autocorrelation), can be calculated through the [methods](#observables) of `Trajectory` and `Ensemble`.
For example:

```python
tamsd = tracer.get_mean_squared_displacement(lagtime=1, rolling=True)
eamsd = ensemble.get_mean_squared_displacement(lagtime=1, rolling=False)
eatamsd = ensemble.get_mean_squared_displacement(lagtime=1, rolling=True)
```

You can visualize the results using [matplotlib](https://github.com/matplotlib/matplotlib).

<p align="center" width="100%">
    <img width="25%" src=https://github.com/jung235/pydiffuser/assets/96967431/ee9d4442-edb2-4c24-8c1b-614a8e385cd4>
</p>

The trajectory is obtained by `matplotlib.pyplot.plot(tracer.position_x1, tracer.position_x2)`.

## CLI

List all stochastic models supported by Pydiffuser.

```console
$ pydiffuser list
NAME            MODEL                           CONFIG                          DIMENSION       
abp             ActiveBrownianParticle          ActiveBrownianParticleConfig    2d              
aoup            ActiveOUParticle                ActiveOUParticleConfig          1d, 2d, 3d      
bm              BrownianMotion                  BrownianMotionConfig            1d, 2d, 3d      
levy            LevyWalk                        LevyWalkConfig                  1d, 2d, 3d      
rtp             RunAndTumbleParticle            RunAndTumbleParticleConfig      1d, 2d, 3d      
smoluchowski    SmoluchowskiEquation            SmoluchowskiEquationConfig      1d, 2d          
```

Here, every model is the subclass of `pydiffuser.models.BaseDiffusion`, and every configuration is the subclass of `pydiffuser.utils.BaseDiffusionConfig`.

## Features

### Observables

*class* `pydiffuser.tracer.Trajectory` ∈ *class* `pydiffuser.tracer.Ensemble`

- `get_increments`
- `get_displacement_moment`
- `get_mean_squared_displacement`
- `get_cosine_moment`
- `get_velocity_autocorrelation`
- `get_real_time`

The above methods are defined in both `Trajectory` and `Ensemble` to enhance transparency.
Using the methods of `Trajectory`, the statistical analysis of [single-particle trajectory](https://en.wikipedia.org/wiki/Single-particle_trajectory) can be accelerated.

### Configuration

We introduce a configuration file to deal with extensive parameter manipulation.
For instance, see [`config.json`](https://github.com/jung235/pydiffuser/blob/main/docs/features/configs/config.json), which contains all parameters demanded to instantiate `pydiffuser.ActiveBrownianParticle`.
Every JSON file of the configurations listed in [CLI](#cli) can be obtained as follows:

```python
import pydiffuser as pyd
from pydiffuser.models import ActiveBrownianParticle, ActiveBrownianParticleConfig


config = ActiveBrownianParticleConfig()
config.to_json(json_path=<JSON_PATH>)
```

We suggest a research pipeline.

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

After calculating the stochastic [observables](#observables), you can plot:

<p align="center" width="100%">
    <img width="30%" src=https://github.com/jung235/pydiffuser/assets/96967431/e92010cf-201d-4920-b508-fc1432544b77>
</p>

It is possible to save & load a picklable object through `pydiffuser.save` and `pydiffuser.load`.

```python
MODEL_PATH = "model.pickle"

pyd.save(obj=model, pickle_path=MODEL_PATH)  # <PICKLE_PATH> = MODEL_PATH
model = pyd.load(pickle_path=MODEL_PATH)
```

## Related Works

[Hyperdiffusion of Poissonian run-and-tumble particles in two dimensions](https://arxiv.org/abs/2308.00554)

## License

[Apache License 2.0](https://github.com/jung235/pydiffuser/blob/main/LICENSE)

## Citation

```bibtex
```
