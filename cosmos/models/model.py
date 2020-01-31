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
from cosmos.models.helper import m_param, theta_param, ScaledBeta
import pandas as pd
import logging
from pyro.ops.indexing import Vindex
from pyro.distributions.util import broadcast_shape
from itertools import product

class Model:
    """ Gaussian Spot Model """
    def __init__(self, data, control, K, lr, n_batch, jit, noise="GammaOffset"):
        # D - number of pixel along axis
        # K - number of states
        # data - number of frames, y axis, x axis
        self.logger = logging.getLogger(__name__)
        self.data = data
        self.control = control 
        self.K = K
        self.D = data.D
        self._params = _noise[noise]
        self.CameraUnit = _noise_fn[noise]
        self.lr = lr
        self.n_batch = n_batch
        
        # create meshgrid of DxD pixel positions
        x_pixel, y_pixel = torch.meshgrid(torch.arange(self.D), torch.arange(self.D))
        self.pixel_pos = torch.stack((x_pixel, y_pixel), dim=-1).float()
        
        # drift locs for 2D gaussian spot
        self.data.target_locs = torch.tensor((self.data.drift[["dx", "dy"]].values.reshape(1,self.data.F,1,1,1,2) + self.data.target[["x", "y"]].values.reshape(self.data.N,1,1,1,1,2)))
        if self.control:
            # drift locs for 2D gaussian spot
            self.control.target_locs = torch.tensor((self.control.drift[["dx", "dy"]].values.reshape(1,self.control.F,1,1,1,2) + self.control.target[["x", "y"]].values.reshape(self.control.N,1,1,1,1,2)))
        
        pyro.clear_param_store()
        self.epoch_count = 0
        self.optim_fn = pyro.optim.Adam
        self.optim_args = {"lr": self.lr, "betas": [0.9, 0.999]}
        self.optim = self.optim_fn(self.optim_args)
        #self.svi = SVI(pyro.poutine.block(self.model, hide=["width_loc", "width_beta"]), self.guide, self.optim, loss=self.elbo)
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)
        self.logger.debug("D - {}".format(self.D))
        self.logger.debug("K - {}".format(self.K))
        self.logger.debug("data.N - {}".format(self.data.N))
        self.logger.debug("data.F - {}".format(self.data.F))
        if self.control:
            self.logger.debug("control.N - {}".format(self.control.N))
            self.logger.debug("control.F - {}".format(self.control.F))
        self.logger.info("Optimizer - {}".format(self.optim_fn.__name__))
        self.logger.info("Learning rate - {}".format(self.lr))
        self.logger.info("Batch size - {}".format(self.n_batch))
        self.logger.info("{}".format("jit" if jit else "nojit"))

        """
        m_matrix
        tensor([[0, 0],
        [0, 1],
        [1, 0],
        [1, 1]])

        theta_matrix:
        tensor([[0, 0],
        [1, 0],
        [0, 1]])
        """
        self.m_matrix = torch.tensor([p for p in product((0,1), repeat=self.K)]).long() # 2**K,K
        self.theta_matrix = torch.eye(self.K+1)[:,1:].long() # K+1,K
        self.size = torch.tensor([2., (((self.D+3)/(2*0.5))**2 - 1)])

        self.path = os.path.join(self.data.path,"runs", "{}".format(self.data.name), "{}".format(self.__name__), "K{}".format(self.K), "{}".format("jit" if jit else "nojit"), "lr{}".format(self.lr), "{}".format(self.optim_fn.__name__), "{}".format(self.n_batch))
        self.writer_scalar = SummaryWriter(log_dir=os.path.join(self.path, "scalar"))
        self.writer_hist = SummaryWriter(log_dir=os.path.join(self.path, "hist"))

    def model(self):
        raise NotImplementedError

    def guide(self):
        raise NotImplementedError

    def spot_model(self, data, m_pi, theta_pi, prefix):
        """
        N plate:
        F plate:
            background      (N,F,1,1,1) (N,F,1,1,1)
            m (enum)        (N,F,1,1,1) (2^K,1,1,1,1,1)
            theta (enum)    (N,F,1,1,1) (K+1,1,1,1,1,1,1)
            m_mask          (2^K,1,1,1,1,K)
            theta_mask      (K+1,1,1,1,1,1,K)
        K plate: 
            height          (N,F,1,1,K) (K+1,2^K,N,F,1,1,K)
            width           (N,F,1,1,K) (K+1,2^K,N,F,1,1,K)
            x               (N,F,1,1,K) (K+1,2^K,N,F,1,1,K)
        """
        #self.size = torch.cat((torch.tensor([2.]), param("proximity")), 0)
        with scope(prefix=prefix):
            with pyro.plate("N_plate", data.N, dim=-5) as batch_idx:
                with pyro.plate("F_plate", data.F, dim=-4):
                    background = pyro.sample("background", dist.Gamma(param("{}/background_loc".format(prefix))[batch_idx] * param("background_beta"), param("background_beta")))
                    m = pyro.sample("m", dist.Categorical(m_pi)) # 4,1,1,1,1,1
                    if theta_pi is not None:
                        theta = pyro.sample("theta", dist.Categorical(theta_pi[m])).squeeze(dim=-1) # 3,1,1,1,1,1,1
                        theta_mask = self.theta_matrix[theta] # 3,1,1,1,1,K
                    else:
                        theta = torch.tensor([0]).long() # N,F,1,1,K   K+1,1,1,1,1,1
                        theta_mask = torch.tensor([0]).long() # N,F,1,1,K   K+1,1,1,1,1,1
                    m_mask = self.m_matrix[m.squeeze(dim=-1)].bool() # 4,1,1,1,1,K
                    with pyro.plate("K_plate", self.K, dim=-1):
                        with pyro.poutine.mask(mask=m_mask):
                            m_shape = broadcast_shape(m_mask.shape, param("{}/h_loc".format(prefix))[batch_idx].shape)
                            theta_shape = broadcast_shape(m_mask.shape, theta_mask.shape, param("{}/h_loc".format(prefix))[batch_idx].shape)
                            height = pyro.sample("height", dist.Gamma(param("height_loc")[theta_mask] * param("height_beta")[theta_mask], param("height_beta")[theta_mask]).expand(theta_shape)) # 4,K,N,F,1,1
                            #height = pyro.sample("height", dist.Gamma(param("height_loc") * param("height_beta"), param("height_beta")).expand(m_shape)) # 4,K,N,F,1,1
                            height = height.masked_fill(~m_mask, 0.)
                            #width = pyro.sample("width", dist.Gamma(param("width_loc") * param("width_beta"), param("width_beta")).expand(m_shape))
                            width = pyro.sample("width", dist.Gamma(param("width_loc")[theta_mask] * param("width_beta")[theta_mask], param("width_beta")[theta_mask]).expand(theta_shape))
                            x0 = pyro.sample("x0", ScaledBeta(0., self.size[theta_mask], -(self.D+3)/2, self.D+3).expand(theta_shape)) * (self.D+3) - (self.D+3)/2
                            y0 = pyro.sample("y0", ScaledBeta(0., self.size[theta_mask], -(self.D+3)/2, self.D+3).expand(theta_shape)) * (self.D+3) - (self.D+3)/2

                    target_locs = data.target_locs[batch_idx] # N,F,1,1,M,K,2 select target locs for given indices
                    locs = self.gaussian_spot(target_locs, height, width, x0, y0) + background
                    with pyro.plate("x_plate", size=self.D, dim=-3):
                        with pyro.plate("y_plate", size=self.D, dim=-2):
                            pyro.sample("data", self.CameraUnit(locs, param("gain"), param("offset")), obs=data[batch_idx].unsqueeze(dim=-1))

    def spot_guide(self, data, theta, m, prefix):
        """
        ixij = Vindex(x)[i, :, j]
        batch_shape = broadcast_shape(i.shape, j.shape)
        event_shape = (x.size(1),)
        assert xij.shape == batch_shape + event_shape

        N plate:
        F plate:
            background      (N,F,1,1,1)               (N,F,1,1,1)
            m (enum)        (N,F,1,1,1,2^K)       (2^K,1,1,1,1,1)
            theta (enum)    (N,F,1,1,1,2^K,K+1) (K+1,1,1,1,1,1,1) batch_shape = (2^K,N,F,1,1,1); event_shape = (K+1,)
            m_mask          (2^K,K)               (2^K,1,1,1,1,K)
            theta_mask      (K+1,K)             (K+1,1,1,1,1,1,K)
        K plate: 
            height          (N,F,1,1,K+1,K)   (K+1,2^K,N,F,1,1,K) batch_shape = (K+1,1,N,F,1,1); event_shape = (K,)
            width           (N,F,1,1,K+1,K)   (K+1,2^K,N,F,1,1,K) batch_shape = (K+1,1,N,F,1,1); event_shape = (K,)
            x               (N,F,1,1,K+1,K)   (K+1,2^K,N,F,1,1,K) batch_shape = (K+1,1,N,F,1,1); event_shape = (K,)

        N plate:
        F plate:
            background      (N,F,1,1,1)               (N,F,1,1,1)
            m (enum)        (N,F,1,1,1,2^K)       (2^K,1,1,1,1,1)
            m (enum)        (N,F,1,1,K,2)         (2^K,1,1,1,1,1)
            theta (enum)    (N,F,1,1,1,2^K,K+1) (K+1,1,1,1,1,1,1) batch_shape = (2^K,N,F,1,1,1); event_shape = (K+1,)
            theta (enum)    (N,F,1,1,K,2) (K+1,1,1,1,1,1,1) batch_shape = (2^K,N,F,1,1,1); event_shape = (K+1,)
            m_mask          (2^K,K)               (2^K,1,1,1,1,K)
            theta_mask      (K+1,K)             (K+1,1,1,1,1,1,K)
        K plate: 
            height          (N,F,1,1,K,2)   (K+1,2^K,N,F,1,1,K) batch_shape = (K+1,1,N,F,1,1,K); event_shape = (,)
            width           (N,F,1,1,K,2)   (K+1,2^K,N,F,1,1,K) batch_shape = (K+1,1,N,F,1,1,K); event_shape = (,)
            x               (N,F,1,1,K,2)   (K+1,2^K,N,F,1,1,K) batch_shape = (K+1,1,N,F,1,1,K); event_shape = (,)
            Vindex(height[batch_idx])[...,theta]
        """
        with scope(prefix=prefix):
            with pyro.plate("N_plate", data.N, subsample_size=self.n_batch, dim=-5) as batch_idx:
                with pyro.plate("F_plate", data.F, dim=-4):
                    pyro.sample("background", dist.Gamma(param("{}/b_loc".format(prefix))[batch_idx] * param("b_beta"), param("b_beta")))
                    if m:
                        m = pyro.sample("m", dist.Categorical(param("{}/m_probs".format(prefix))[batch_idx])) # 4,1,1,1,1,1
                    if theta:
                        theta = pyro.sample("theta", dist.Categorical(Vindex(param("{}/theta_probs".format(prefix))[batch_idx])[...,m,:])).squeeze(dim=-1) # N,F,1,1
                        #theta_mask = self.theta_matrix[theta] # 3,1,1,1,1,K
                    #else:
                        #theta = torch.tensor([0]).long()
                        #theta_mask = torch.tensor([0]).long()
                    m_mask = self.m_matrix[m.squeeze(dim=-1)].bool() # N,F,1,1,K
                    with pyro.plate("K_plate", self.K, dim=-1):
                        with pyro.poutine.mask(mask=m_mask):
                            #theta_shape = broadcast_shape(theta_mask.shape, m_mask.shape, param("{}/h_loc".format(prefix))[batch_idx].shape)
                            #m_shape = broadcast_shape(m_mask.shape, param("{}/h_loc".format(prefix))[batch_idx].shape)
                            #pyro.sample("height", dist.Gamma(Vindex(param("{}/h_loc".format(prefix))[batch_idx])[...,theta_mask] * param("h_beta"), param("h_beta")).expand(theta_shape))
                            pyro.sample("height", dist.Gamma(param("{}/h_loc".format(prefix))[batch_idx] * param("h_beta"), param("h_beta")))
                            pyro.sample("width", dist.Gamma(param("{}/w_loc".format(prefix))[batch_idx] * param("{}/w_beta".format(prefix))[batch_idx], param("{}/w_beta".format(prefix))[batch_idx]))
                            #pyro.sample("width", ScaledBeta(param("{}/w_mode".format(prefix))[batch_idx],
                            #                                param("{}/w_size".format(prefix))[batch_idx], 0.75, 1.5)) * 1.5 + 0.75
                            pyro.sample("x0", ScaledBeta(param("{}/x_mode".format(prefix))[batch_idx],
                                                         param("{}/size".format(prefix))[batch_idx], -(self.D+3)/2, self.D+3)) * (self.D+3) - (self.D+3)/2
                            pyro.sample("y0", ScaledBeta(param("{}/y_mode".format(prefix))[batch_idx],
                                                         param("{}/size".format(prefix))[batch_idx], -(self.D+3)/2, self.D+3)) * (self.D+3) - (self.D+3)/2


    def spot_parameters(self, data, theta, m, prefix):
        param("{}/background_loc".format(prefix), torch.ones(data.N,1,1,1,1)*100., constraint=constraints.positive)
        param("{}/b_loc".format(prefix), torch.ones(data.N,data.F,1,1,1)*30., constraint=constraints.positive)
        #height = torch.ones(data.N,data.F,1,1,self.K,2)*1000.
        height = torch.ones(data.N,data.F,1,1,self.K)*100.
        #height[...,1] = 10.
        param("{}/h_loc".format(prefix), height, constraint=constraints.positive)
        if m:
            param("{}/m_probs".format(prefix), torch.ones(data.N,data.F,1,1,1,2**self.K), constraint=constraints.simplex)
        if theta:
            theta_probs = torch.ones(data.N,data.F,1,1,1,2**self.K,self.K+1)
            theta_probs[...,0,1:] = 0.
            theta_probs[...,1,1] = 0.
            theta_probs[...,2,2] = 0.
            param("{}/theta_probs".format(prefix), theta_probs, constraint=constraints.simplex)
        param("{}/w_loc".format(prefix), torch.ones(data.N,data.F,1,1,self.K)*1.3, constraint=constraints.positive)
        param("{}/w_beta".format(prefix), torch.ones(data.N,data.F,1,1,self.K)*100., constraint=constraints.positive)
        #param("{}/w_mode".format(prefix), torch.ones(data.N,data.F,1,1,self.K)*1.3, constraint=constraints.interval(0.75,2.25))
        #param("{}/w_size".format(prefix), torch.ones(data.N,data.F,1,1,self.K)*100., constraint=constraints.greater_than(2.))
        param("{}/x_mode".format(prefix), torch.zeros(data.N,data.F,1,1,self.K), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))
        param("{}/y_mode".format(prefix), torch.zeros(data.N,data.F,1,1,self.K), constraint=constraints.interval(-(self.D+3)/2,(self.D+3)/2))
        size = torch.ones(data.N,data.F,1,1,self.K)*5.
        size[...,1] = (((self.D+3)/(2*0.5))**2 - 1) # 30 is better than 100
        param("{}/size".format(prefix), size, constraint=constraints.greater_than(2.))

    # Ideal 2D gaussian spot
    def gaussian_spot(self, target_locs, height, width, x0, y0):
        # return gaussian spot with height, width, and drift adjusted position xy
        spot_locs = torch.zeros(broadcast_shape(x0.unsqueeze(dim=-1).shape, target_locs.shape))
        spot_locs[...,0] = target_locs[...,0] + x0 # N,F,1,1,K,2   3,4,N,F,1,1,K,2
        spot_locs[...,1] = target_locs[...,1] + y0 # N,F,1,1,K,2   3,4,N,F,1,1,K,2
        spot = []
        for k in range(self.K):
            w = width[...,k] # N,F,1,1   4,N,F,1,1
            rv = dist.MultivariateNormal(spot_locs[...,k,:], scale_tril=torch.eye(2) * w.view(w.size()+(1,1)))
            gaussian_spot = torch.exp(rv.log_prob(self.pixel_pos)) # N,F,D,D
            spot.append(height[...,k] * gaussian_spot) # N,F,D,D
        return torch.stack(spot, dim=-1).sum(dim=-1, keepdim=True)

    def train(self, num_steps):
        for epoch in tqdm(range(num_steps)):
            #with torch.autograd.detect_anomaly():
            epoch_loss = self.svi.step()
            if not self.epoch_count % 100:    
                write_summary(self.epoch_count, epoch_loss, self, self.svi, self.writer_scalar, self.writer_hist, feature=False, mcc=self.mcc)
                self.save_checkpoint()
            self.epoch_count += 1

    def save_checkpoint(self):
        if not any([torch.isnan(v).any() for v in pyro.get_param_store().values()]):
            self.optim.save(os.path.join(self.path, "optimizer"))
            pyro.get_param_store().save(os.path.join(self.path, "params"))
            np.savetxt(os.path.join(self.path, "epoch_count"), np.array([self.epoch_count]))
            self.logger.debug("Step #{}. Saved model params and optimizer state in {}".format(self.epoch_count, self.path))
        else:
            self.logger.warning("Step #{}. Detected NaN values in parameters")

    def load_checkpoint(self):
        try:
            self.epoch_count = int(np.loadtxt(os.path.join(self.path, "epoch_count")))
            self.optim.load(os.path.join(self.path, "optimizer"))
            pyro.get_param_store().load(os.path.join(self.path, "params"), map_location=self.data.device)
            self.logger.info("Step #{}. Loaded model params and optimizer state from {}".format(self.epoch_count, self.path))
        except:
            pass
