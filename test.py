import os
import torch
import pyro
import pyro.distributions as dist
from torch.distributions import constraints
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate, infer_discrete
from pyro.ops.indexing import Vindex

pyro.set_rng_seed(0)

@config_enumerate
def model():
    b = pyro.sample("b", dist.Binomial(100, torch.tensor([0 , .2, .8, 1])))
    #p = pyro.param("p", torch.randn(5, 4, 3, 2).exp(), constraint=constraints.simplex)
    #x = pyro.sample("x", dist.Categorical(torch.ones(4)))
    #y = pyro.sample("y", dist.Categorical(torch.ones(3)))
    #with pyro.plate("z_plate", 5):
    #    p_xy = Vindex(p)[..., x, y, :]
    #    z = pyro.sample("z", dist.Categorical(p_xy))
    print('   b.shape = {}'.format(b.shape))
    #print('   p.shape = {}'.format(p.shape))
    #print('   x.shape = {}'.format(x.shape))
    #print('   y.shape = {}'.format(y.shape))
    #print('p_xy.shape = {}'.format(p_xy.shape))
    #print('   z.shape = {}'.format(z.shape))
    #return x, y, z

def guide():
    pass

pyro.clear_param_store()
elbo = TraceEnum_ELBO(max_plate_nesting=1)
elbo.loss(model, guide);
