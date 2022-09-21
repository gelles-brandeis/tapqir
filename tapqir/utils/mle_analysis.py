# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.distributions.constraints as constraints
from pyro.ops.indexing import Vindex
from pyroapi import distributions as dist
from pyroapi import handlers, infer, optim, pyro


def train(model, guide, lr=1e-3, n_steps=1000, jit=True, verbose=False, **kwargs):

    pyro.clear_param_store()
    optimizer = optim.Adam({"lr": lr})
    elbo = (
        infer.JitTraceEnum_ELBO(max_plate_nesting=2)
        if jit
        else infer.TraceEnum_ELBO(max_plate_nesting=2)
    )
    svi = infer.SVI(model, guide, optimizer, elbo)

    for step in range(n_steps):
        svi.step(**kwargs)
        if step % 100 == 99 and verbose:
            values = tuple(f"{k}: {v}" for k, v in pyro.get_param_store().items())
            print(values)


def ttfb_model(data, control, Tmax):
    r"""
    Eq. 4 and Eq. 7 in::

      @article{friedman2015multi,
        title={Multi-wavelength single-molecule fluorescence analysis of transcription mechanisms},
        author={Friedman, Larry J and Gelles, Jeff},
        journal={Methods},
        volume={86},
        pages={27--36},
        year={2015},
        publisher={Elsevier}
      }

    :param data: time prior to the first binding at the target location
    :param control: time prior to the first binding at the control location
    :param Tmax: entire observation interval
    """
    ka = pyro.param(
        "ka",
        lambda: torch.full((data.shape[0], 1), 0.001),
        constraint=constraints.positive,
    )
    kns = pyro.param(
        "kns",
        lambda: torch.full((data.shape[0], 1), 0.001),
        constraint=constraints.positive,
    )
    Af = pyro.param(
        "Af",
        lambda: torch.full((data.shape[0], 1), 0.9),
        constraint=constraints.unit_interval,
    )
    k = torch.stack([kns, ka + kns])

    # on-target data
    mask = (data < Tmax) & (data > 0)
    tau = data.masked_fill(~mask, 1.0)
    with pyro.plate("bootstrap", data.shape[0], dim=-2) as bdx:
        with pyro.plate("N", data.shape[1], dim=-1):
            active = pyro.sample(
                "active", dist.Bernoulli(Af), infer={"enumerate": "parallel"}
            )
            with handlers.mask(mask=(data == Tmax)):
                pyro.factor("Tmax", -Vindex(k)[active.long().squeeze(-1), bdx] * Tmax)
                # pyro.factor("Tmax", -k * Tmax)
            with handlers.mask(mask=mask):
                pyro.sample(
                    "tau",
                    dist.Exponential(Vindex(k)[active.long().squeeze(-1), bdx]),
                    obs=tau,
                )
                # pyro.sample("tau", dist.Exponential(k), obs=tau)

    # negative control data
    if control is not None:
        mask = (control < Tmax) & (control > 0)
        tauc = control.masked_fill(~mask, 1.0)
        with pyro.plate("bootstrapc", control.shape[0], dim=-2):
            with pyro.plate("Nc", control.shape[1], dim=-1):
                with handlers.mask(mask=(control == Tmax)):
                    pyro.factor("Tmaxc", -kns * Tmax)
                with handlers.mask(mask=mask):
                    pyro.sample("tauc", dist.Exponential(kns), obs=tauc)


def ttfb_guide(data, control, Tmax):
    pass  # MLE


def exp_model(data, K):
    with pyro.plate("components", K):
        k = pyro.param(
            "k", lambda: torch.logspace(-K + 1, 0, K), constraint=constraints.positive
        )
    A = pyro.param("A", lambda: torch.ones(K), constraint=constraints.simplex)

    with pyro.plate("data", len(data)):
        m = pyro.sample("m", dist.Categorical(A), infer={"enumerate": "parallel"})
        pyro.sample("obs", dist.Exponential(k[m.long()]), obs=data)


def exp_guide(data, K):
    pass  # MLE
