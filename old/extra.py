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