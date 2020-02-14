import torch
import torch.distributions.constraints as constraints
import numpy as np
import os
import pyro
import pyro.distributions as dist
from pyro.infer import SVI
from pyro.infer import JitTraceEnum_ELBO, TraceEnum_ELBO
from pyro import param
from pyro.contrib.autoname import scope
from cosmos.models.noise import _noise_fn
from cosmos.utils.utils import write_summary
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from cosmos.models.helper import ScaledBeta
import logging
from pyro.ops.indexing import Vindex
from pyro.distributions.util import broadcast_shape
from itertools import product
from pyro import poutine


class Model:
    """ Gaussian Spot Model """
    def __init__(self, data, control,
                 K, lr, n_batch, jit,
                 noise="GammaOffset"):
        # D - number of pixel along axis
        # K - number of states
        self.logger = logging.getLogger(__name__)
        self.data = data
        self.control = control
        self.K = K
        self.D = data.D
        self.CameraUnit = _noise_fn[noise]
        self.lr = lr
        self.n_batch = n_batch

        # create meshgrid of DxD pixel positions
        x_pixel, y_pixel = torch.meshgrid(
            torch.arange(self.D), torch.arange(self.D))
        self.pixel_pos = torch.stack((x_pixel, y_pixel), dim=-1).float()

        # drift locs for 2D gaussian spot
        self.data.target_locs = torch.tensor(
            self.data.drift[["dx", "dy"]].values
            .reshape(1, self.data.F, 1, 1, 1, 2)
            + self.data.target[["x", "y"]].values
            .reshape(self.data.N, 1, 1, 1, 1, 2))
        if self.control:
            self.control.target_locs = torch.tensor(
                self.control.drift[["dx", "dy"]].values
                .reshape(1, self.control.F, 1, 1, 1, 2)
                + self.control.target[["x", "y"]].values
                .reshape(self.control.N, 1, 1, 1, 1, 2))

        pyro.clear_param_store()
        self.epoch_count = 0
        self.optim_fn = pyro.optim.Adam
        self.optim_args = {"lr": self.lr, "betas": [0.9, 0.999]}
        self.optim = self.optim_fn(self.optim_args)
        self.elbo = (JitTraceEnum_ELBO if jit else TraceEnum_ELBO) \
            (max_plate_nesting=5, ignore_jit_warnings=True)
        self.param_path = os.path.join(
            self.data.path, "runs", "{}".format(self.data.name),
            "marginalfixed", "K{}".format(self.K),
            "{}".format("jit" if jit else "nojit"),
            "lr0.005", "{}".format(self.optim_fn.__name__),
            "{}".format(self.n_batch))
        pyro.get_param_store().load(
            os.path.join(self.param_path, "params"),
            map_location=self.data.device)
        self.svi = SVI(
            poutine.block(
                self.model,
                hide=[param for param in pyro.get_param_store().keys()]),
            poutine.block(
                self.guide,
                hide=[param for param in pyro.get_param_store().keys()]),
            self.optim, loss=self.elbo)
        # self.svi = SVI(
        #     poutine.block(
        #         self.model,
        #         hide=["width_mode", "width_size"]),
        #     self.guide, self.optim, loss=self.elbo)
        # self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)
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
        M_MATRIX
        tensor([[0, 0],
        [0, 1],
        [1, 0],
        [1, 1]])

        THETA_MATRIX:
        tensor([[0, 0],
        [1, 0],
        [0, 1]])
        """
        self.M_MATRIX = torch.tensor(
            [p for p in product((0, 1), repeat=self.K)]).long()  # 2**K,K
        self.THETA_MATRIX = torch.eye(
            self.K+1)[:, 1:].long()  # K+1,K
        self.size = torch.tensor([2., (((self.D+3)/(2*0.5))**2 - 1)])

        self.path = os.path.join(
            self.data.path, "runs", "{}".format(self.data.name),
            "{}fixed".format(self.__name__), "K{}".format(self.K),
            "{}".format("jit" if jit else "nojit"),
            "lr{}".format(self.lr), "{}".format(self.optim_fn.__name__),
            "{}".format(self.n_batch))
        self.writer_scalar = SummaryWriter(
            log_dir=os.path.join(self.path, "scalar"))
        self.writer_hist = SummaryWriter(
            log_dir=os.path.join(self.path, "hist"))

    def model(self):
        raise NotImplementedError

    def guide(self):
        raise NotImplementedError

    def spot_model(self, data, m_pi, theta_pi, data_mask, prefix):
        N_plate = pyro.plate("N_plate", data.N, dim=-5)
        F_plate = pyro.plate("F_plate", data.F, dim=-4)
        X_plate = pyro.plate("X_plate", self.D, dim=-3)
        Y_plate = pyro.plate("Y_plate", self.D, dim=-2)
        K_plate = pyro.plate("K_plate", self.K, dim=-1)

        with N_plate as batch_idx, F_plate:
            with scope(prefix=prefix), poutine.mask(mask=data_mask[batch_idx]):
                background = pyro.sample(
                    "background", dist.Gamma(
                        param("{}/background_loc".format(prefix))[batch_idx]
                        * param("background_beta"), param("background_beta")))
                m = pyro.sample("m", dist.Categorical(m_pi))
                m_mask = self.M_MATRIX[m.squeeze(dim=-1)].bool()
                if theta_pi is not None:
                    theta = pyro.sample(
                        "theta", dist.Categorical(theta_pi[m])).squeeze(dim=-1)
                    theta_mask = self.THETA_MATRIX[theta]
                else:
                    theta = torch.tensor([0]).long()
                    theta_mask = torch.tensor([0]).long()
                with K_plate:
                    height = pyro.sample(
                        "height", dist.Gamma(
                            param("height_loc") * param("height_beta"),
                            param("height_beta")))
                    height = height.masked_fill(~m_mask, 0.)
                    width = pyro.sample(
                        "width", ScaledBeta(
                            param("width_mode"), param("width_size"), 0.5, 2.))
                    width = width * 2. + 0.5
                    x0 = pyro.sample(
                        "x0", ScaledBeta(
                            0, self.size[theta_mask], -(self.D+3)/2, self.D+3))
                    x0 = x0 * (self.D+3) - (self.D+3)/2
                    y0 = pyro.sample(
                        "y0", ScaledBeta(
                            0, self.size[theta_mask], -(self.D+3)/2, self.D+3))
                    y0 = y0 * (self.D+3) - (self.D+3)/2

                target_locs = data.target_locs[batch_idx]
                locs = self.gaussian_spot(target_locs, height, width, x0, y0) \
                    + background
                with X_plate, Y_plate:
                    pyro.sample(
                        "data", self.CameraUnit(
                            locs, param("gain"), param("offset")),
                        obs=data[batch_idx].unsqueeze(dim=-1))

    def spot_guide(self, data, theta, m, data_mask, prefix):
        """
        ixij = Vindex(x)[i, :, j]
        batch_shape = broadcast_shape(i.shape, j.shape)
        event_shape = (x.size(1),)
        assert xij.shape == batch_shape + event_shape
        """
        N_plate = pyro.plate("N_plate", data.N,
                             subsample_size=self.n_batch, dim=-5)
        F_plate = pyro.plate("F_plate", data.F, dim=-4)
        K_plate = pyro.plate("K_plate", self.K, dim=-1)

        with N_plate as batch_idx, F_plate:
            with scope(prefix=prefix), poutine.mask(mask=data_mask[batch_idx]):
                pyro.sample(
                    "background", dist.Gamma(
                        param("{}/b_loc".format(prefix))[batch_idx]
                        * param("b_beta"), param("b_beta")))
                if m:
                    m = pyro.sample(
                        "m", dist.Categorical(
                            param("{}/m_probs".format(prefix))[batch_idx]))
                if theta:
                    pyro.sample(
                        "theta", dist.Categorical(
                            Vindex(param("{}/theta_probs"
                                   .format(prefix))[batch_idx])[..., m, :]))
                with K_plate:
                    pyro.sample(
                        "height", dist.Gamma(
                            param("{}/h_loc".format(prefix))[batch_idx]
                            * param("h_beta"), param("h_beta")))
                    pyro.sample(
                        "width", ScaledBeta(
                            param("{}/w_mode".format(prefix))[batch_idx],
                            param("{}/w_size".format(prefix))[batch_idx],
                            0.5, 2.))
                    pyro.sample(
                        "x0", ScaledBeta(
                            param("{}/x_mode".format(prefix))[batch_idx],
                            param("{}/size".format(prefix))[batch_idx],
                            -(self.D+3)/2, self.D+3))
                    pyro.sample(
                        "y0", ScaledBeta(
                            param("{}/y_mode".format(prefix))[batch_idx],
                            param("{}/size".format(prefix))[batch_idx],
                            -(self.D+3)/2, self.D+3))

    def spot_parameters(self, data, theta, m, prefix):
        param("{}/background_loc".format(prefix),
              torch.ones(data.N, 1, 1, 1, 1) * 100.,
              constraint=constraints.positive)
        param("{}/b_loc".format(prefix),
              torch.ones(data.N, data.F, 1, 1, 1) * 30.,
              constraint=constraints.positive)
        param("{}/h_loc".format(prefix),
              torch.ones(data.N, data.F, 1, 1, self.K) * 1000.,
              constraint=constraints.positive)
        if m:
            param("{}/m_probs".format(prefix),
                  torch.ones(data.N, data.F, 1, 1, 1, 2**self.K),
                  constraint=constraints.simplex)
        if theta:
            theta_probs = torch.ones(
                data.N, data.F, 1, 1, 1, 2**self.K, self.K+1)
            theta_probs[..., 0, 1:] = 0.
            theta_probs[..., 1, 1] = 0.
            theta_probs[..., 2, 2] = 0.
            param("{}/theta_probs".format(prefix),
                  theta_probs, constraint=constraints.simplex)
        param("{}/w_mode".format(prefix),
              torch.ones(data.N, data.F, 1, 1, self.K) * 1.3,
              constraint=constraints.interval(0.5, 2.5))
        param("{}/w_size".format(prefix),
              torch.ones(data.N, data.F, 1, 1, self.K) * 100.,
              constraint=constraints.greater_than(2.))
        param("{}/x_mode".format(prefix),
              torch.zeros(data.N, data.F, 1, 1, self.K),
              constraint=constraints.interval(-(self.D+3)/2, (self.D+3)/2))
        param("{}/y_mode".format(prefix),
              torch.zeros(data.N, data.F, 1, 1, self.K),
              constraint=constraints.interval(-(self.D+3)/2, (self.D+3)/2))
        size = torch.ones(data.N, data.F, 1, 1, self.K)*5.
        size[..., 1] = ((self.D+3) / (2*0.5)) ** 2 - 1
        param("{}/size".format(prefix),
              size, constraint=constraints.greater_than(2.))

    # Ideal 2D gaussian spot
    def gaussian_spot(self, target_locs, height, width, x0, y0):
        # return gaussian spot with
        # height, width, and drift adjusted position xy
        spot_locs = torch.zeros(
            broadcast_shape(
                x0.unsqueeze(dim=-1).shape,
                target_locs.shape))
        spot_locs[..., 0] = target_locs[..., 0] + x0
        spot_locs[..., 1] = target_locs[..., 1] + y0
        spot = []
        for k in range(self.K):
            w = width[..., k]  # N,F,1,1   4,N,F,1,1
            rv = dist.MultivariateNormal(
                spot_locs[..., k, :],
                scale_tril=torch.eye(2) * w.view(w.size()+(1, 1)))
            gaussian_spot = torch.exp(rv.log_prob(self.pixel_pos))  # N,F,D,D
            spot.append(height[..., k] * gaussian_spot)  # N,F,D,D
        return torch.stack(spot, dim=-1).sum(dim=-1, keepdim=True)

    def train(self, num_steps):
        for epoch in tqdm(range(num_steps)):
            # with torch.autograd.detect_anomaly():
            epoch_loss = self.svi.step()
            if not self.epoch_count % 100:
                write_summary(self.epoch_count, epoch_loss,
                              self, self.svi, self.writer_scalar,
                              self.writer_hist, feature=False, mcc=self.mcc)
                self.save_checkpoint()
            self.epoch_count += 1

    def save_checkpoint(self):
        if not any([torch.isnan(v).any()
                   for v in pyro.get_param_store().values()]):
            self.optim.save(os.path.join(self.path, "optimizer"))
            pyro.get_param_store().save(os.path.join(self.path, "params"))
            np.savetxt(os.path.join(self.path, "epoch_count"),
                       np.array([self.epoch_count]))
            if self.mcc:
                z_probs = torch.sum(
                    param("d/m_probs").squeeze()
                    * param("d/theta_probs").squeeze()[..., 1:].sum(dim=-1),
                    dim=-1)
                self.data.labels["probs"] = z_probs.reshape(-1).data.cpu()
                self.data.labels["binary"] = self.data.labels["probs"] > 0.5
                self.data.labels.to_csv(os.path.join(self.path, "labels.csv"))
            self.logger.debug(
                    "Step #{}. Saved model params and optimizer state in {}"
                    .format(self.epoch_count, self.path))
        else:
            self.logger.warning("Step #{}. Detected NaN values in parameters")

    def load_checkpoint(self):
        try:
            self.epoch_count = int(
                np.loadtxt(os.path.join(self.path, "epoch_count")))
            self.optim.load(os.path.join(self.path, "optimizer"))
            pyro.get_param_store().load(
                    os.path.join(self.path, "params"),
                    map_location=self.data.device)
            self.logger.info(
                    "Step #{}. Loaded model params and optimizer state from {}"
                    .format(self.epoch_count, self.path))
        except:
            pass
