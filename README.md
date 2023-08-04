# Bicyclus
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Bicyclus is a Bayesian Inference module for Cyclus.
Specifically, it is an interface written in Python3 that connects
[Cyclus](https://fuelcycle.org/), a nuclear fuel simulator, with
[PyMC](https://www.pymc.io/) (formerly PyMC3), a Python package for
Bayesian statistics using Markov chain Monte Carlo algorithms.

> Bicyclus, Bicyclus, Bicyclus  
> I want to use my Bicyclus  
> I want to use Cyclus  
> I want to infer with Cyclus  
> I want to infer what I like.  
>
> _Freddie Mercury (somewhen, somewhere, probably)_

## Features
- Run inference processes on nuclear fuel cycles simulated with Cyclus.
- Store quantities of interest and simulation output that are not part of the
  inference process, in separate files.
  These can be used for later analysis.
- Use PyMC distributions as priors (WIP, currently restricted to a subset of
  distributions).
- Features utility functions for post-processing and plotting posterior
  distributions.

## How-to
### Requirements
The following table lists the software requirements.
All dependencies listed below will be installed automatically through `pip3`
except for Cyclus, which the user must install themself.

| Name | Tested with version | Notes |
|:-----|---:|:---|
| [Python](https://www.python.org/) | `3.10.6` | |
| [PyMC](https://www.pymc.io/welcome.html) | `4.2.0` | must be used with PyMC v4.x or larger |
| [Aesara](https://aesara.readthedocs.io/en/latest/) | `2.8.2` | only for PyMC v4.x |
| [PyTensor](https://pytensor.readthedocs.io/en/latest/) | | only for PyMC v5.x or larger |
| [Arviz](https://python.arviz.org/en/latest/index.html) | `0.12.1` | |
| [NumPy](https://numpy.org/doc/stable/index.html) | `1.23.3` | |
| [Scipy](https://docs.scipy.org/doc/scipy/index.html) | `1.9.1` | |
| [Pandas](https://pandas.pydata.org/) | `1.5.0` | only for plotting |
| [matplotlib](https://matplotlib.org/) | `3.6.0` | only for plotting |
| [seaborn](https://seaborn.pydata.org/) | `0.12.0` | only for plotting |
| [Cyclus](https://fuelcycle.org/) | `1.5.5-59-gb1a858e3` | Must be installed by the user |

### Installation
Install Bicyclus and *all* dependencies.
```bash
$ git clone https://github.com/Nuclear-Verification-and-Disarmament/bicyclus.git
$ cd bicyclus
$ pip3 install ".[plotting]"
```

If you do not want to install dependencies needed for plotting, run the
following commands:
```bash
$ git clone https://github.com/Nuclear-Verification-and-Disarmament/bicyclus.git
$ cd bicyclus
$ pip3 install .
```
You can still run the inference process, but features from
`bicyclus/postprocessing` may not be available.

### Tutorial
#### Inference mode
A minimum working example (MWE) can be found in the [`examples`](/examples)
directory.
At the moment, Bicyclus can be used through a driver script that has to be
written on a case-by-case basis.
However, the MWE provided should be a good starting point for any new driver script.

#### Forward mode
Bicyclus can be used to perform large-scale forward simulations with Cyclus,
which can be useful, e.g., to perform sensity analyses.
This so-called forward mode uses Quasi Monte Carlo sampling (specifically, Sobol
sequences) to efficiently sample the input parameter space.
Furthermore, the driver script and file structure used in the inference mode can
largely be reused here.

An MWE might be provided at a later stage.

## Pitfalls
- Depending on the runtime of one Cyclus run, `subprocess`'s timeout value has
  to be adapted.
  It is defined in `bicyclus/blackbox/blackbox.py`
  (`CyclusCliModel.simulate`) and is currently set to 300 seconds (as of
  September 2022).

## Legacy code
Originally, this work was developed as part of a Bachelor's thesis by Lewin
Bormann.
If you are interested in the git blame and log beyond this repository's initial
commit, please visit the
[original repository](https://git.rwth-aachen.de/nvd/fuel-cycle/bayesian-cycle/).
That repository also contains two applications of complex nuclear fuel cycles
and reconstruction scenarios.

## Contributing: Usage of pre-commit hooks
We follow the [`Black`](https://black.readthedocs.io/en/stable/) code style.
Run the following command from the root directory to enable use of the
pre-commit hook.
This will automatically run `black` when comitting and thus will ensure proper
formatting of the committed code.
```bash
$ git config --local core.hooksPath .githooks/
```
