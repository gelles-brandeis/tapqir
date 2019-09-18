import numpy as np
import torch
import torch.distributions.constraints as constraints
import pyro
from pyro import poutine
from pyro.infer import  config_enumerate
import pyro.distributions as dist

# template functions for classes A (spot) and B (no spot)
def classA(height, width, background, D):
    loc = torch.ones(2) * (D-1)/2
    #precision = torch.ones(2) / width**2
    x, y = torch.meshgrid(torch.arange(D), torch.arange(D))
    pos = torch.stack((x, y), dim=2).float()
    rv = dist.MultivariateNormal(loc, scale_tril=torch.eye(2)*width)
    #rv = dist.MultivariateNormal(loc, precision_matrix=torch.eye(2)*precision)
    gaussian_spot = torch.exp(rv.expand([D,D]).log_prob(pos)) * 2 * np.pi * width**2
    #gaussian_spot = torch.exp(rv.expand([D,D]).log_prob(pos)) * 2 * np.pi / precision
    return height * gaussian_spot + background

def classB(background, D):
    return torch.ones(D,D)*background

def generate(height, width, background, pi, D, K, N):
    data = torch.zeros(N,D,D)
    states = torch.zeros(N)
    # class templates
    locs = torch.zeros(K,D,D)
    locs[0,:,:] = classB(background, D)
    locs[1,:,:] = classA(height, width, background, D)
    
    transition = torch.tensor([[0.4, 0.6], [0.4, 0.6]])
    z = pyro.sample("z_0", dist.Categorical(pi))
    #with pyro.plate("sample_size", N):
    for t in range(N):
        # hidden states
        #z = pyro.sample("z", dist.Categorical(pi))
        states[t] = z
        data[t,:,:] = pyro.sample("data_{}".format(t), dist.Poisson(locs[z,:,:]).to_event(2))
        z = pyro.sample("z_{}".format(t+1), dist.Categorical(transition[z]))
        # add normal noise (std = 20) to template images
        #data = pyro.sample("data", dist.Normal(locs[z,:,:], noise).to_event(2))
    return data, states

def hmm_model(data, K):
    sample_size, N, _ = data.shape
    # prior distributions
    #weights = pyro.sample("weights", dist.Dirichlet(0.5 * torch.ones(K)))
    height = pyro.sample('height', dist.Uniform(0., 300.))
    background = pyro.sample('background', dist.Uniform(0., 1000.))
    # class templates
    locs = torch.zeros(K,N,N)
    locs[0,:,:] = classB(background)
    locs[1,:,:] = classA(height, 2, background)
    with pyro.plate("hidden_state", K):
        transition = pyro.sample("transition", dist.Dirichlet(0.5 * torch.ones(K)))

    #with pyro.plate("sample_size", sample_size):
    z = 0
    for t in range(sample_size):
        # hidden states
        z = pyro.sample("z_{}".format(t), dist.Categorical(transition[z]), infer={"enumerate": "parallel"})
        # likelihood / conditioning on data
        pyro.sample("obs_{}".format(t), dist.Normal(locs[z,:,:], 20.).to_event(2), obs=data[t])

@config_enumerate
def model(data, K):
    N, D, _ = data.shape
    # prior distributions
    weights = pyro.sample("weights", dist.Dirichlet(0.5 * torch.ones(K)))
    
    height = pyro.sample("height", dist.HalfNormal(500.))
    background = pyro.sample("background", dist.HalfNormal(1000.))
    width = pyro.sample("width", dist.Gamma(1., 0.1)) # dist.HalfNormal(100.)
    # class templates
    locs = torch.zeros((K,D,D))
    locs[0,:,:] = classB(background, D)
    locs[1,:,:] = classA(height, width, background, D)
    
    beta = pyro.param("beta", torch.tensor(50.), constraint=constraints.positive)

    with pyro.plate("sample_size", N):
        # hidden states
        z = pyro.sample("z", dist.Categorical(weights))
        # likelihood / conditioning on data
        pyro.sample("obs", dist.Gamma(locs[z,:,:]*beta, beta).to_event(2), obs=data)
        
def guide(data, K):
    N, D, _ = data.shape
    # posterior approximations
    weights_conc = pyro.param("weights_conc", torch.ones(K), constraint=constraints.positive)
    weights = pyro.sample("weights", dist.Dirichlet(weights_conc))
    
    beta = pyro.param("beta", torch.tensor(1.), constraint=constraints.positive)    
    
    height_loc = pyro.param("height_loc", torch.tensor(100.), constraint=constraints.positive)
    height_scale = pyro.param("height_scale", torch.tensor(20.), constraint=constraints.positive)
    height = pyro.sample("height", dist.Normal(height_loc, height_scale)) # true distribution
    
    background_loc = pyro.param("background_loc", torch.tensor(360.), constraint=constraints.positive)
    background_scale = pyro.param("background_scale", torch.tensor(50.), constraint=constraints.positive)
    background = pyro.sample('background', dist.Normal(background_loc, background_scale))
    
    width_alpha = pyro.param("width_alpha", torch.tensor(10.), constraint=constraints.positive)
    width_beta = pyro.param("width_beta", torch.tensor(20.), constraint=constraints.positive)
    width = pyro.sample("width", dist.Gamma(width_alpha, width_beta))
    

@config_enumerate
def normal_model(data, K):
    N, D, _ = data.shape
    # prior distributions
    weights = pyro.sample("weights", dist.Dirichlet(0.5 * torch.ones(K)))
    
    locs_loc = torch.zeros((K,D,D))
    locs_loc[0,:,:] = classB(500., D)
    locs_loc[1,:,:] = classA(100., 0.5, 500., D)
    locs = pyro.sample("locs", dist.Normal(locs_loc, 500.).to_event(3))
    sigma = pyro.sample("sigma", dist.Gamma(1, 0.001))
    with pyro.plate("sample_size", N):
        # hidden states
        z = pyro.sample("z", dist.Categorical(weights))
        # likelihood / conditioning on data
        pyro.sample("obs", dist.Normal(locs[z,:,:], sigma).to_event(2), obs=data)
        
        
def normal_guide(data, K):
    N, D, _ = data.shape
    # posterior approximations
    weights_conc = pyro.param("weights_conc", torch.ones(K)*0.5, constraint=constraints.positive)
    weights = pyro.sample("weights", dist.Dirichlet(weights_conc))
    
    height = pyro.param("height", torch.tensor(100.), constraint=constraints.positive)
    width = pyro.param("width", torch.tensor(0.5), constraint=constraints.positive)
    background = pyro.param("background", torch.tensor(500.), constraint=constraints.positive)
    
    locs_scale = pyro.param("locs_scale", torch.tensor(50.))
    
    locs_loc = torch.zeros((K,D,D))
    locs_loc[0,:,:] = classB(background, D)
    locs_loc[1,:,:] = classA(height, width, background, D)
    locs = pyro.sample("locs", dist.Normal(locs_loc, locs_scale).to_event(3))
    
    sigma_alpha = pyro.param("sigma_alpha", torch.tensor(20.), constraint=constraints.positive)
    sigma_beta = pyro.param("sigma_beta", torch.tensor(4.), constraint=constraints.positive)
    sigma = pyro.sample("sigma", dist.Gamma(sigma_alpha, sigma_beta))