import torch
import numpy as np
import os
from torch.distributions.transforms import AffineTransform
import pyro
import pyro.distributions as dist
from cosmos.models.noise import _noise, _noise_fn
from cosmos.utils.utils import write_summary
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Model:
    """ Gaussian Spot Model """
    def __init__(self, data, control, K, lr, n_batch, jit, noise="GammaOffset"):
        # D - number of pixel along axis
        # K - number of states
        # data - number of frames, y axis, x axis
        self.data = data
        self.control = control 
        self.K = K
        self.D = data.D
        self._params = _noise[noise]
        self.CameraUnit = _noise_fn[noise]
        self.lr = lr
        self.n_batch = n_batch
        self.path = os.path.join(self.data.path,"runs", "{}".format(self.data.name), "{}".format(self.__name__), "K{}".format(self.K), "lr{}".format(self.lr))
        self.writer = SummaryWriter(log_dir=self.path)
        
        # create meshgrid of DxD pixel positions
        x_pixel, y_pixel = torch.meshgrid(torch.arange(self.D), torch.arange(self.D))
        self.pixel_pos = torch.stack((x_pixel, y_pixel), dim=-1).float()
        
        # drift locs for 2D gaussian spot
        self.target_locs = torch.tensor((self.data.drift[["dx", "dy"]].values.reshape(1,self.data.F,2) + self.data.target[["x", "y"]].values.reshape(self.data.N,1,2)), dtype=torch.float32)
        self.target_locs = self.target_locs.reshape(self.data.N,self.data.F,1,1,1,2).repeat(1,1,1,1,self.K,1)# N,F,1,1,M,K,2

        if self.control:
            # drift locs for 2D gaussian spot
            self.control_locs = torch.tensor((self.control.drift[["dx", "dy"]].values.reshape(1,self.control.F,2) + self.control.target[["x", "y"]].values.reshape(self.control.N,1,2)), dtype=torch.float32)
            self.control_locs = self.control_locs.reshape(self.control.N,self.control.F,1,1,1,2).repeat(1,1,1,1,self.K,1)# N,F,1,1,M,K,2

    def Location(self, mean, size, loc, scale):
        """
        Location(mode, size, loc, scale) = loc + scale * Beta((mode - loc) / scale, size)
        mode(Location) = mode
        var(Location) = <Location ** 2> - <Location> ** 2
        <Location> = loc + scale * <Beta>
        <Location> ** 2 = loc ** 2 + (scale ** 2) * (<Beta> ** 2) + 2 * loc * scale * <Beta>
        Location ** 2 = loc ** 2 + (scale ** 2) * (Beta ** 2) + 2 * loc * scale * Beta
        <Location ** 2> = loc ** 2 + (scale ** 2) * <Beta ** 2> + 2 * loc * scale * <Beta>
        var(Location) = (scale ** 2) * (<Beta ** 2> - <Beta> ** 2)
        """ 
        mean = (mean - loc) / scale
        concentration1 = mean * size
        concentration0 = (1 - mean) * size
        base_distribution = dist.Beta(concentration1, concentration0)
        transforms =  [AffineTransform(loc=loc, scale=scale)]
        return dist.TransformedDistribution(base_distribution, transforms)

    def Location2(self, concentration0, concentration1, loc, scale):
        base_distribution = dist.Beta(concentration1, concentration0)
        transforms =  [AffineTransform(loc=loc, scale=scale)]
        return dist.TransformedDistribution(base_distribution, transforms)

    # Ideal 2D gaussian spot
    def gaussian_spot(self, spot_locs, height, width, x0, y0):
        # return gaussian spot with height, width, and drift adjusted position xy
        spot_locs[...,0] += x0 # N,F,1,1,K,2
        spot_locs[...,1] += y0 # N,F,1,1,K,2
        spot = torch.zeros(height.shape[0],height.shape[1],self.D,self.D)
        for k in range(self.K):
            #w = width.reshape(1,1,1,1)
            w = width[...,k] # N,F,1,1
            rv = dist.MultivariateNormal(spot_locs[...,k,:], scale_tril=torch.eye(2) * w.view(w.size()+(1,1)))
            gaussian_spot = torch.exp(rv.log_prob(self.pixel_pos)) # N,F,D,D
            spot += height[...,k] * gaussian_spot # N,F,D,D
        return spot

    def epoch(self, num_epochs):
        for epoch in tqdm(range(num_epochs)):
            #with torch.autograd.detect_anomaly():
            epoch_loss = self.svi.step()
            if not (self.epoch_count % 1000):    
                write_summary(self.epoch_count, epoch_loss, self, self.svi, self.writer, feature=False, mcc=self.mcc)
                self.save(verbose=False)
            self.epoch_count += 1
        self.save()

    def save(self, verbose=True):
        self.optim.save(os.path.join(self.path, "optimizer"))
        pyro.get_param_store().save(os.path.join(self.path, "params"))
        np.savetxt(os.path.join(self.path, "epoch_count"), np.array([self.epoch_count]))
        if verbose:
            print("Classification results were saved in {}...".format(self.path))

    def load(self):
        try:
            self.epoch_count = int(np.loadtxt(os.path.join(self.path, "epoch_count")))
            self.optim.load(os.path.join(self.path, "optimizer"))
            pyro.get_param_store().load(os.path.join(self.path, "params"))
            print("loaded from {}".format(self.path))
        except:
            pass

def m_param(pi, lamda, K):
    bernoulli = lambda x: dist.Bernoulli(pi[1]).log_prob(torch.tensor([float(x)])).exp()
    poisson = lambda x: dist.Poisson(lamda).log_prob(torch.tensor([float(x)])).exp()
    m_pi = torch.zeros(2**K)
    m_pi[0] = bernoulli(0.) * poisson(0.)
    k = 1
    for m in range(1,K+1):
        r = int(math.factorial(K)/(math.factorial(K-m)*math.factorial(m)))
        for _ in range(r):
            m_pi[k] = (bernoulli(1) * poisson(m-1) + bernoulli(0) * poisson(m)) / r
            k += 1
    return m_pi

def theta_param(pi, lamda, K):
    bernoulli = lambda x: dist.Bernoulli(pi[1]).log_prob(torch.tensor([float(x)])).exp()
    poisson = lambda x: dist.Poisson(lamda).log_prob(torch.tensor([float(x)])).exp()
    theta_pi = torch.zeros(2**K,K+1)
    theta_pi[0,0] = 1
    theta_pi[1,0] = bernoulli(0) * poisson(1) / (m_param(pi, lamda, K)[1] * 2)
    theta_pi[1,1] = bernoulli(1) * poisson(0) / (m_param(pi, lamda, K)[1] * 2)
    theta_pi[2,0] = bernoulli(0) * poisson(1) / (m_param(pi, lamda, K)[2] * 2)
    theta_pi[2,2] = bernoulli(1) * poisson(0) / (m_param(pi, lamda, K)[2] * 2)
    theta_pi[3,0] = bernoulli(0) * poisson(2) / m_param(pi, lamda, K)[3]
    theta_pi[3,1] = bernoulli(1) * poisson(1) / (m_param(pi, lamda, K)[3] * 2)
    theta_pi[3,2] = bernoulli(1) * poisson(1) / (m_param(pi, lamda, K)[3] * 2)
    return theta_pi
    #k = 1
    #for m in range(1,K+1):
    #    r = int(math.factorial(K)/(math.factorial(K-m)*math.factorial(m)))
    #    for _ in range(r):
    #        theta_pi[k,0] = bernoulli(0) * poisson(m) / m_param(pi, lamda, K)[m]
    #        theta_pi[k,1:k+1] = bernoulli(1) * poisson(m-1) / m_param(pi, lamda, K)[m]
    #        k += 1
