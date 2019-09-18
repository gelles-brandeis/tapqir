from __future__ import print_function
import os
from collections import defaultdict
import numpy as np
import scipy.stats
import torch
from torch.distributions import constraints
from matplotlib import pyplot

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete

#smoke_test = ('CI' in os.environ)
#assert pyro.__version__.startswith('0.4.0')
pyro.enable_validation(True)

data = torch.tensor([0., 1., 10., 11., 12.])

K = 2  # Fixed number of components.

@config_enumerate
def model(data):
    # Global variables.
    weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(K)))
    scale = pyro.sample('scale', dist.LogNormal(0., 2.))
    with pyro.plate('components', K):
        locs = pyro.sample('locs', dist.Normal(0., 10.))

    with pyro.plate('data', len(data)):
        # Local variables.
        assignment = pyro.sample('assignment', dist.Categorical(weights))
        pyro.sample('obs', dist.Normal(locs[assignment], scale), obs=data)

@config_enumerate
def full_guide(data):
    weights_concentration = pyro.param("weights_concentration", 5*torch.ones(K))
    pyro.sample('weights', dist.Dirichlet(weights_concentration))
    scale_v = pyro.param("scale_v", torch.ones(1))
    scale = pyro.sample('scale', dist.Delta(scale_v))
    # Global variables.
    locs_v = pyro.param("locs_v", torch.ones(1))
    with pyro.plate('components', K):
        locs = pyro.sample('locs', dist.Delta(locs_v))

    # Local variables.
    with pyro.plate('data', len(data)):
        assignment_probs = pyro.param('assignment_probs', torch.ones(len(data), K) / K,
                                      constraint=constraints.unit_interval)
        pyro.sample('assignment', dist.Categorical(assignment_probs))

optim = pyro.optim.Adam({'lr': 0.2, 'betas': [0.8, 0.99]})
elbo = TraceEnum_ELBO(max_plate_nesting=1)
svi = SVI(model, full_guide, optim, loss=elbo)


def initialize(seed):
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    # Initialize weights to uniform.
    pyro.param('weights_concentration', 5 * torch.ones(K), constraint=constraints.simplex)
    # Assume half of the data variance is due to intra-component noise.
    pyro.param('scale_v', (data.var() / 2).sqrt(), constraint=constraints.positive)
    # Initialize means from a subsample of data.
    pyro.param('locs_v', data[torch.multinomial(torch.ones(len(data)) / len(data), K)]);
    loss = svi.loss(model, full_guide, data)
    return loss

# Choose the best among 100 random initializations.
loss, seed = min((initialize(seed), seed) for seed in range(100))
initialize(seed)
print('seed = {}, initial_loss = {}'.format(seed, loss))


losses = []
for i in range(400):
    loss = svi.step(data)
    losses.append(loss)
    print('.' if i % 100 else '\n', end='')

#map_estimates = full_guide(data)
weights = pyro.param("weights_concentration").detach().numpy() 
locs = pyro.param("locs_v").detach().numpy()
scale = pyro.param("scale_v").detach().numpy()
print('weights = {}'.format(weights))
print('locs = {}'.format(locs))
print('scale = {}'.format(scale))
