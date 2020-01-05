import torch
import torch.distributions.constraints as constraints
import numpy as np
import os
from torch.distributions.transforms import AffineTransform
import pyro
import pyro.distributions as dist
from pyro.infer import SVI
from pyro import param 
from pyro.contrib.autoname import scope
from cosmos.models.noise import _noise, _noise_fn
from cosmos.utils.utils import write_summary
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math
from cosmos.models.helper import Location, m_param, theta_param

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
        self.path = os.path.join(self.data.path,"runs", "{}".format(self.data.name), "{}".format(self.__name__), "K{}".format(self.K), "{}".format("jit" if jit else "nojit"), "lr{}".format(self.lr))
        self.writer = SummaryWriter(log_dir=self.path)
        
        # create meshgrid of DxD pixel positions
        x_pixel, y_pixel = torch.meshgrid(torch.arange(self.D), torch.arange(self.D))
        self.pixel_pos = torch.stack((x_pixel, y_pixel), dim=-1).float()
        
        # drift locs for 2D gaussian spot
        self.data.target_locs = torch.tensor((self.data.drift[["dx", "dy"]].values.reshape(1,self.data.F,2) + self.data.target[["x", "y"]].values.reshape(self.data.N,1,2)), dtype=torch.float32)
        self.data.target_locs = self.data.target_locs.reshape(self.data.N,self.data.F,1,1,1,2).repeat(1,1,1,1,self.K,1)# N,F,1,1,M,K,2

        if self.control:
            # drift locs for 2D gaussian spot
            self.control.target_locs = torch.tensor((self.control.drift[["dx", "dy"]].values.reshape(1,self.control.F,2) + self.control.target[["x", "y"]].values.reshape(self.control.N,1,2)), dtype=torch.float32)
            self.control.target_locs = self.control.target_locs.reshape(self.control.N,self.control.F,1,1,1,2).repeat(1,1,1,1,self.K,1)# N,F,1,1,M,K,2
        
        pyro.clear_param_store()
        self.parameters()
        self.epoch_count = 0
        #self.optim = pyro.optim.ClippedAdam({"lr": self.lr, "betas": [0.9, 0.999], "clip_norm": 10.0})
        self.optim = pyro.optim.Adam({"lr": self.lr, "betas": [0.9, 0.999]})
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)

        self.m_matrix = torch.tensor([[0, 0], [1,0], [0,1], [1,1]])
        self.theta_matrix = torch.tensor([[0,0], [1,0], [0,1]]) # K+1,K
        self.scale = torch.tensor([10., 0.5])
        self.width_size = torch.tensor([3., 15.])
        #size = torch.tensor([2., (((self.D+3)/(2*0.5))**2 - 1)])

    def model(self):
        raise NotImplementedError

    def guide(self):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    def spot_model(self, data, m_pi, theta_pi, prefix):
        with scope(prefix=prefix):
            with pyro.plate("N_plate", data.N, dim=-5) as batch_idx:
                with pyro.plate("F_plate", data.F, dim=-4):
                    background = pyro.sample("background", dist.Gamma(param("background_loc") * param("background_beta"), param("background_beta")))
                    m = pyro.sample("m", dist.Categorical(m_pi)) # N,F,1,1,1
                    if theta_pi is not None:
                        theta = pyro.sample("theta", dist.Categorical(theta_pi[m])) # N,F,1,1,1
                        theta = self.theta_matrix[theta.squeeze(dim=-1)]
                    else:
                        theta = 0 # N,F,1,1,K   K+1,1,1,1,1,1
                    m = self.m_matrix[m.squeeze(dim=-1)] # N,F,1,1,K
                    with pyro.plate("K_plate", self.K, dim=-1):
                        with pyro.poutine.mask(mask=m.bool()):
                            height = pyro.sample("height", dist.Gamma(param("height_loc") * param("height_beta"), param("height_beta"))) # K,N,F,1,1
                            height = height.masked_fill(~m.bool(), 0.)
                            width = pyro.sample("width", Location(1.3, self.width_size[theta], 0.5, 2.5))
                            x0 = pyro.sample("x0", dist.Normal(0., self.scale[theta]))
                            y0 = pyro.sample("y0", dist.Normal(0., self.scale[theta]))
                            #width = pyro.sample("width", Location(param("width_mode")[theta], param("width_size")[theta], 0.5, 2.5))
                            #x0 = pyro.sample("x0", Location(0., size[theta], -(self.D+3)/2, self.D+3))
                            #y0 = pyro.sample("y0", Location(0., size[theta], -(self.D+3)/2, self.D+3))

                    spot_locs = data.target_locs[batch_idx] # N,F,1,1,M,K,2 select target locs for given indices
                    locs = self.gaussian_spot(spot_locs, height, width, x0, y0) + background
                    with pyro.plate("x_plate", size=self.D, dim=-3):
                        with pyro.plate("y_plate", size=self.D, dim=-2):
                            pyro.sample("data", self.CameraUnit(locs, param("gain"), param("offset")), obs=data[batch_idx].unsqueeze(dim=-1))

    def spot_guide(self, data, theta, prefix):
        with scope(prefix=prefix):
            with pyro.plate("N_plate", data.N, subsample_size=self.n_batch, dim=-5) as batch_idx:
                with pyro.plate("F_plate", data.F, dim=-4):
                    pyro.sample("background", dist.Gamma(param("{}/b_loc".format(prefix))[batch_idx] * param("b_beta"), param("b_beta")))
                    m = pyro.sample("m", dist.Categorical(param("{}/m_probs".format(prefix))[batch_idx]))
                    m = self.m_matrix[m.squeeze(dim=-1)] # N,F,1,1,K
                    if theta:
                        pyro.sample("theta", dist.Categorical(param("{}/theta_probs".format(prefix))[batch_idx])) # N,F,1,1
                    with pyro.plate("K_plate", self.K, dim=-1):
                        with pyro.poutine.mask(mask=m.bool()):
                            pyro.sample("height", dist.Gamma(param("{}/h_loc".format(prefix))[batch_idx] * param("h_beta"), param("h_beta")))
                            pyro.sample("width", Location(param("{}/w_mode".format(prefix))[batch_idx], param("{}/w_size".format(prefix))[batch_idx], 0.5, 2.5))
                            pyro.sample("x0", dist.Normal(param("{}/x_mean".format(prefix))[batch_idx], param("{}/scale".format(prefix))[batch_idx]))
                            pyro.sample("y0", dist.Normal(param("{}/y_mean".format(prefix))[batch_idx], param("{}/scale".format(prefix))[batch_idx]))
                            #pyro.sample("x0", Location(param("x_mode")[batch_idx], param("size")[batch_idx], -(self.D+3)/2, self.D+3)) # N,F,1,1,M,K
                            #pyro.sample("y0", Location(param("y_mode")[batch_idx], param("size")[batch_idx], -(self.D+3)/2, self.D+3))

    def spot_parameters(self, data, theta, prefix):
        param("{}/m_probs".format(prefix), torch.ones(data.N,data.F,1,1,1,4), constraint=constraints.simplex)
        if theta:
            param("{}/theta_probs".format(prefix), torch.ones(data.N,data.F,1,1,1,self.K+1), constraint=constraints.simplex)
        param("{}/b_loc".format(prefix), torch.ones(data.N,data.F,1,1,1)*30., constraint=constraints.positive)
        param("{}/h_loc".format(prefix), torch.ones(data.N,data.F,1,1,self.K)*1000., constraint=constraints.positive)
        param("{}/w_mode".format(prefix), torch.ones(data.N,data.F,1,1,self.K)*1.35, constraint=constraints.interval(0.5,3.))
        param("{}/w_size".format(prefix), torch.ones(data.N,data.F,1,1,self.K)*100., constraint=constraints.greater_than(2.))
        param("{}/x_mean".format(prefix), torch.zeros(data.N,data.F,1,1,self.K), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))
        param("{}/y_mean".format(prefix), torch.zeros(data.N,data.F,1,1,self.K), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))
        scale = torch.ones(data.N,data.F,1,1,self.K)*5.
        scale[...,1] = 0.5 # 30 is better than 100
        param("{}/scale".format(prefix), scale, constraint=constraints.positive)
        #param("c_x_mode", torch.zeros(self.control.N,self.control.F,1,1,self.K), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))
        #param("c_y_mode", torch.zeros(self.control.N,self.control.F,1,1,self.K), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))
        #size = torch.ones(self.control.N,self.control.F,1,1,self.K)*5.
        #size[...,1] = (((self.D+3)/(2*0.5))**2 - 1) # 30 is better than 100
        #param("c_size", size, constraint=constraints.greater_than(2.))

    # Ideal 2D gaussian spot
    def gaussian_spot(self, spot_locs, height, width, x0, y0):
        # return gaussian spot with height, width, and drift adjusted position xy
        spot_locs[...,0] += x0 # N,F,1,1,K,2   4,N,F,1,1,K
        spot_locs[...,1] += y0 # N,F,1,1,K,2
        spot = []
        for k in range(self.K):
            #w = width.reshape(1,1,1,1)
            w = width[...,k] # N,F,1,1   4,N,F,1,1
            rv = dist.MultivariateNormal(spot_locs[...,k,:], scale_tril=torch.eye(2) * w.view(w.size()+(1,1)))
            gaussian_spot = torch.exp(rv.log_prob(self.pixel_pos)) # N,F,D,D
            spot.append(height[...,k] * gaussian_spot) # N,F,D,D
        return torch.stack(spot, dim=-1).sum(dim=-1, keepdim=True)

    def train(self, num_epochs):
        for epoch in tqdm(range(num_epochs)):
            with torch.autograd.detect_anomaly():
                epoch_loss = self.svi.step()
            if not (self.epoch_count % 1000):    
                write_summary(self.epoch_count, epoch_loss, self, self.svi, self.writer, feature=False, mcc=self.mcc)
                self.save_checkpoint(verbose=False)
            self.epoch_count += 1
        #self.save_checkpoint()

    def save_checkpoint(self, verbose=True):
        self.optim.save(os.path.join(self.path, "optimizer"))
        pyro.get_param_store().save(os.path.join(self.path, "params"))
        np.savetxt(os.path.join(self.path, "epoch_count"), np.array([self.epoch_count]))
        if verbose:
            print("done saving model and optimizer checkpoints in {}...".format(self.path))

    def load_checkpoint(self):
        try:
            self.epoch_count = int(np.loadtxt(os.path.join(self.path, "epoch_count")))
            self.optim.load(os.path.join(self.path, "optimizer"))
            pyro.get_param_store().load(os.path.join(self.path, "params"))
            print("done loading model and optimizer states from {}".format(self.path))
        except:
            pass
