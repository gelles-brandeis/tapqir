import torch
import torch.distributions.constraints as constraints
from pyroapi import distributions as dist
from pyroapi import infer, optim, pyro


def train(model, guide, lr=1e-3, n_steps=1000, **kwargs):

    pyro.clear_param_store()
    optimizer = optim.Adam({"lr": lr})
    elbo = infer.TraceEnum_ELBO(max_plate_nesting=1)
    svi = infer.SVI(model, guide, optimizer, elbo)

    for step in range(n_steps):
        svi.step(**kwargs)
        if step % 100 == 99:
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
    ka = pyro.param("ka", lambda: torch.tensor(0.05), constraint=constraints.positive)
    kns = pyro.param("kns", lambda: torch.tensor(0.01), constraint=constraints.positive)
    Af = pyro.param(
        "Af", lambda: torch.tensor(0.5), constraint=constraints.unit_interval
    )
    k = torch.stack([kns, ka + kns])

    # on-target data
    n = sum(data == Tmax)  # no binding has occured
    tau = data[(data < Tmax) & (data > 0)]
    with pyro.plate("n", n):
        active1 = pyro.sample(
            "active1", dist.Bernoulli(Af), infer={"enumerate": "parallel"}
        )
        pyro.factor("Tmax", -k[active1.long()] * Tmax)
    with pyro.plate("N-n-nz", len(tau)):
        active2 = pyro.sample(
            "active2", dist.Bernoulli(Af), infer={"enumerate": "parallel"}
        )
        pyro.sample("tau", dist.Exponential(k[active2.long()]), obs=tau)

    # negative control data
    if control is not None:
        nc = sum(control == Tmax)  # no binding has occured
        tauc = control[(control < Tmax) & (control > 0)]
        with pyro.plate("nc", nc):
            pyro.factor("Tmaxc", -kns * Tmax)
        with pyro.plate("Nc-nc-ncz", len(tauc)):
            pyro.sample("tauc", dist.Exponential(kns), obs=tauc)


def ttfb_guide(data, control, Tmax):
    pass  # MLE


def double_exp_model(data):
    k1 = pyro.param("k1", lambda: torch.tensor(0.01), constraint=constraints.positive)
    k2 = pyro.param("k2", lambda: torch.tensor(0.05), constraint=constraints.positive)
    A = pyro.param("A", lambda: torch.tensor(0.5), constraint=constraints.unit_interval)
    k = torch.stack([k1, k2])

    with pyro.plate("data", len(data)):
        m = pyro.sample("m", dist.Bernoulli(A), infer={"enumerate": "parallel"})
        pyro.sample("obs", dist.Exponential(k[m.long()]), obs=data)


def double_exp_guide(data):
    pass  # MLE
