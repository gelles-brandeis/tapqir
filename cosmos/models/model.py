import torch
import torch.nn as nn
import torch.distributions.constraints as constraints
import numpy as np
import os
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, infer_discrete, config_enumerate, NUTS, MCMC
from pyro.infer import JitTraceEnum_ELBO, TraceEnum_ELBO
from pyro import param
from pyro.contrib.autoname import scope
from cosmos.models.noise import _noise_fn
from cosmos.utils.utils import write_summary
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from tqdm import tqdm
from cosmos.models.helper import ScaledBeta
import logging
from pyro.ops.indexing import Vindex
from pyro.distributions.util import broadcast_shape
from itertools import product
from pyro import poutine


class GaussianSpot(nn.Module):
    def __init__(self, data, K):
        super().__init__()
        self.K = K
        # create meshgrid of DxD pixel positions
        x_pixel, y_pixel = torch.meshgrid(
            torch.arange(data.D), torch.arange(data.D))
        self.pixel_pos = torch.stack((x_pixel, y_pixel), dim=-1).float()

        # drift locs for 2D gaussian spot
        self.target_locs = torch.tensor(
            data.drift[["dx", "dy"]].values.reshape(1, data.F, 1, 1, 1, 2)
            + data.target[["x", "y"]].values.reshape(data.N, 1, 1, 1, 1, 2))
        
    # Ideal 2D gaussian spot
    def forward(self, batch_idx, height, width, x0, y0, background):
        target_locs = self.target_locs[batch_idx]
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
        return torch.stack(spot, dim=-1).sum(dim=-1, keepdim=True) + background


class Model(nn.Module):
    """ Gaussian Spot Model """
    def __init__(self, data, control,
                 K, lr, n_batch, jit,
                 noise="GammaOffset"):
        # D - number of pixel along axis
        # K - number of states
        self.data = data
        self.control = control
        self.K = K
        self.CameraUnit = _noise_fn[noise]
        self.lr = lr
        self.n_batch = n_batch
        self.jit = jit

        self.data.loc = GaussianSpot(self.data, K)
        if self.control:
            self.control.loc = GaussianSpot(self.control, K)

        pyro.clear_param_store()
        self.epoch_count = 0
        self.optim_fn = pyro.optim.Adam
        self.optim_args = {"lr": self.lr, "betas": [0.9, 0.999]}
        self.optim = self.optim_fn(self.optim_args)
        self.elbo = (JitTraceEnum_ELBO if jit else TraceEnum_ELBO) \
            (max_plate_nesting=5, ignore_jit_warnings=True)
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)

        self.THETA_MATRIX = torch.tensor([[0, 0], [1, 0], [0, 1]])
        self.size = torch.tensor([2., (((data.D+3)/(2*0.5))**2 - 1)])

        self.log()

    def model(self):
        raise NotImplementedError

    def guide(self):
        raise NotImplementedError

    def spot_model(self, data, z, theta_pi, j_pi, data_mask, prefix):
        with scope(prefix=prefix):
            N_plate = pyro.plate("N_plate", data.N, dim=-5)
            F_plate = pyro.plate("F_plate", data.F, dim=-4)
            X_plate = pyro.plate("X_plate", data.D, dim=-3)
            Y_plate = pyro.plate("Y_plate", data.D, dim=-2)
            K_plate = pyro.plate("K_plate", self.K, dim=-1)
            with N_plate as batch_idx, F_plate:
                with poutine.mask(mask=data_mask[batch_idx]):
                    background = pyro.sample(
                        "background", dist.Gamma(
                            param("{}/background_loc".format(prefix))[batch_idx]
                            * param("background_beta"), param("background_beta")))
                    if z:
                        z = pyro.sample("z", dist.Categorical(param("pi")))
                    else:
                        z = 0
                    theta = pyro.sample("theta", dist.Categorical(theta_pi[z]))
                    theta_mask = self.THETA_MATRIX[theta.squeeze(dim=-1)] 
                    #j_mask = self.J_MATRIX[j.squeeze(dim=-1)]

                    with K_plate:
                        j = pyro.sample("j", dist.Categorical(j_pi[theta_mask]))
                        m_mask = (theta_mask + j).bool()
                        height = pyro.sample(
                            "height", dist.Gamma(
                                param("height_loc") * param("height_beta"),
                                param("height_beta")))
                        width = pyro.sample(
                            "width", ScaledBeta(
                                param("width_mode")[theta_mask], param("width_size")[theta_mask], 0.5, 2.))
                        x0 = pyro.sample(
                            "x0", ScaledBeta(
                                0, self.size[theta_mask], -(data.D+3)/2, data.D+3))
                        y0 = pyro.sample(
                            "y0", ScaledBeta(
                                0, self.size[theta_mask], -(data.D+3)/2, data.D+3))

                    height = height.masked_fill(~m_mask, 0.)
                    width = width * 2. + 0.5
                    x0 = x0 * (data.D+3) - (data.D+3)/2
                    y0 = y0 * (data.D+3) - (data.D+3)/2

                    locs = data.loc(batch_idx, height, width, x0, y0, background)
                    with X_plate, Y_plate:
                        pyro.sample(
                            "data", self.CameraUnit(
                                locs, param("gain"), param("offset")),
                            obs=data[batch_idx].unsqueeze(dim=-1))

    def spot_guide(self, data, z, j, data_mask, prefix):
        with scope(prefix=prefix):
            N_plate = pyro.plate("N_plate", data.N,
                                 subsample_size=self.n_batch, dim=-5)
            F_plate = pyro.plate("F_plate", data.F, dim=-4)
            K_plate = pyro.plate("K_plate", self.K, dim=-1)
            with N_plate as batch_idx, F_plate:
                with poutine.mask(mask=data_mask[batch_idx]):
                    self.batch_idx = batch_idx
                    pyro.sample(
                        "background", dist.Gamma(
                            param("{}/b_loc".format(prefix))[batch_idx]
                            * param("{}/b_beta".format(prefix))[batch_idx],
                            param("{}/b_beta".format(prefix))[batch_idx]))
                    if z:
                        z = pyro.sample(
                            "z", dist.Categorical(
                                param("{}/z_probs".format(prefix))[batch_idx])).squeeze(dim=-1)
                    else:
                        z = 0
                    #theta = pyro.sample(
                    #    "theta", dist.Categorical(
                    #        param("{}/theta_probs".format(prefix))[batch_idx]))
                    with K_plate:
                        if j:
                            pyro.sample(
                                "j", dist.Categorical(
                                    #param("{}/j_probs".format(prefix))[batch_idx]))
                                    Vindex(param("{}/j_probs".format(prefix))[batch_idx])[...,z,:,:]))
                        pyro.sample(
                            "height", dist.Gamma(
                                param("{}/h_loc".format(prefix))[batch_idx]
                                * param("{}/h_beta".format(prefix))[batch_idx],
                                param("{}/h_beta".format(prefix))[batch_idx]))
                        pyro.sample(
                            "width", ScaledBeta(
                                param("{}/w_mode".format(prefix))[batch_idx],
                                param("{}/w_size".format(prefix))[batch_idx],
                                0.5, 2.))
                        pyro.sample(
                            "x0", ScaledBeta(
                                param("{}/x_mode".format(prefix))[batch_idx],
                                param("{}/size".format(prefix))[batch_idx],
                                -(data.D+3)/2, data.D+3))
                        pyro.sample(
                            "y0", ScaledBeta(
                                param("{}/y_mode".format(prefix))[batch_idx],
                                param("{}/size".format(prefix))[batch_idx],
                                -(data.D+3)/2, data.D+3))

    def spot_parameters(self, data, z, j, prefix):
        param("{}/background_loc".format(prefix),
              torch.ones(data.N, 1, 1, 1, 1) * 100.,
              constraint=constraints.positive)
        if z:
            param("{}/z_probs".format(prefix),
                  torch.ones(data.N, data.F, 1, 1, 1, 2),
                  constraint=constraints.simplex)
        if j:
            j_probs = torch.ones(data.N, data.F, 1, 1, 2, self.K, 2)
            j_probs[..., 1, 0, :] = torch.tensor([1., 0.])
            param("{}/j_probs".format(prefix),
                  #torch.ones(data.N, data.F, 1, 1, self.K, 2),
                  #constraint=constraints.simplex)
                  j_probs, constraint=constraints.simplex)
        param("{}/b_loc".format(prefix),
              torch.ones(data.N, data.F, 1, 1, 1) * 30.,
              constraint=constraints.positive)
        param("{}/b_beta".format(prefix),
              torch.ones(data.N, data.F, 1, 1, 1) * 30,
              constraint=constraints.positive)
        param("{}/h_loc".format(prefix),
              torch.ones(data.N, data.F, 1, 1, self.K) * 1000.,
              constraint=constraints.positive)
        param("{}/h_beta".format(prefix),
              torch.ones(data.N, data.F, 1, 1, self.K),
              constraint=constraints.positive)
        param("{}/w_mode".format(prefix),
              torch.ones(data.N, data.F, 1, 1, self.K) * 1.3,
              constraint=constraints.interval(0.5, 2.5))
        param("{}/w_size".format(prefix),
              torch.ones(data.N, data.F, 1, 1, self.K) * 100.,
              constraint=constraints.greater_than(2.))
        param("{}/x_mode".format(prefix),
              torch.zeros(data.N, data.F, 1, 1, self.K),
              constraint=constraints.interval(-(data.D+3)/2, (data.D+3)/2))
        param("{}/y_mode".format(prefix),
              torch.zeros(data.N, data.F, 1, 1, self.K),
              constraint=constraints.interval(-(data.D+3)/2, (data.D+3)/2))
        size = torch.ones(data.N, data.F, 1, 1, self.K) * 5.
        size[..., 0] = ((data.D+3) / (2*0.5)) ** 2 - 1
        param("{}/size".format(prefix),
              size, constraint=constraints.greater_than(2.))

    def model_parameters(self):
        # Global Parameters
        # param("proximity", torch.tensor([(((self.D+3)/(2*0.5))**2 - 1)]),
        #       constraint=constraints.greater_than(30.))
        param("background_beta", torch.tensor([1.]),
              constraint=constraints.positive)
        param("height_loc", torch.tensor([1000.]),
              constraint=constraints.positive)
        param("height_beta", torch.tensor([0.01]),
              constraint=constraints.positive)
        param("width_mode", torch.tensor([1.25, 1.25]),
              constraint=constraints.interval(0.5, 2.5))
        param("width_size",
              torch.tensor([3., 15.]), constraint=constraints.positive)
        #param("A", torch.ones(2,2), constraint=constraints.simplex)
        param("pi", torch.ones(2), constraint=constraints.simplex)
        param("j_pi", torch.ones(2), constraint=constraints.simplex)
        #param("j_pi", torch.tensor([0.1]),
        #      constraint=constraints.greater_than(0.))

        if self.control:
            self.offset_max = torch.where(
                self.data[:].min() < self.control[:].min(),
                self.data[:].min() - 0.1,
                self.control[:].min() - 0.1)
        else:
            self.offset_max = self.data[:].min() - 0.1
        param("offset", self.offset_max-50,
              constraint=constraints.interval(0, self.offset_max))
        param("gain", torch.tensor(5.), constraint=constraints.positive)

    def train(self, num_steps):
        for epoch in tqdm(range(num_steps)):
            # with torch.autograd.detect_anomaly():
            #import pdb; pdb.set_trace()
            self.epoch_loss = self.svi.step()
            if not self.epoch_count % 100:
                self.save_checkpoint()
            self.epoch_count += 1

    def mcmc(self):
        if self.control:
            self.offset_max = torch.where(
                self.data[:].min() < self.control[:].min(),
                self.data[:].min() - 0.1,
                self.control[:].min() - 0.1)
        else:
            self.offset_max = self.data[:].min() - 0.1
        nuts_kernel = NUTS(poutine.lift(
            self.model,
            prior={"d/background_loc": dist.HalfNormal(500.),
                   "background_beta": dist.HalfNormal(100.),
                   "height_loc": dist.HalfNormal(2000.),
                   "height_beta": dist.HalfNormal(10.),
                   "width_mode": dist.Uniform(0.5, 2.5),
                   "width_size": dist.HalfNormal(50.),
                   "gain": dist.HalfNormal(30.),
                   "offset": dist.Uniform(0., self.offset_max),
                   "pi": dist.Dirichlet(torch.ones(2)),
                   "lamda": dist.Exponential(.5)}))
        mcmc = MCMC(nuts_kernel, num_samples=10000, warmup_steps=500)
        mcmc.run()

    def log(self):
        self.logger = logging.getLogger(__name__)
        self.logger.debug("D - {}".format(self.data.D))
        self.logger.debug("K - {}".format(self.K))
        self.logger.debug("data.N - {}".format(self.data.N))
        self.logger.debug("data.F - {}".format(self.data.F))
        if self.control:
            self.logger.debug("control.N - {}".format(self.control.N))
            self.logger.debug("control.F - {}".format(self.control.F))
        self.logger.info("Optimizer - {}".format(self.optim_fn.__name__))
        self.logger.info("Learning rate - {}".format(self.lr))
        self.logger.info("Batch size - {}".format(self.n_batch))
        self.logger.info("{}".format("jit" if self.jit else "nojit"))

        self.path = os.path.join(
            self.data.path, "runs", "{}".format(self.data.name),
            "{}nn".format(self.__name__), "K{}".format(self.K),
            "{}".format("jit" if self.jit else "nojit"),
            "lr{}".format(self.lr), "{}".format(self.optim_fn.__name__),
            "{}".format(self.n_batch))
        self.writer_scalar = SummaryWriter(
            log_dir=os.path.join(self.path, "scalar"))
        self.writer_hist = SummaryWriter(
            log_dir=os.path.join(self.path, "hist"))

    def save_checkpoint(self):
        if not any([torch.isnan(v).any()
                   for v in pyro.get_param_store().values()]):
            self.optim.save(os.path.join(self.path, "optimizer"))
            pyro.get_param_store().save(os.path.join(self.path, "params"))
            np.savetxt(os.path.join(self.path, "epoch_count"),
                       np.array([self.epoch_count]))

            self.writer_scalar.add_scalar("-ELBO", self.epoch_loss, self.epoch_count)
            for p, value in pyro.get_param_store().named_parameters():
                if pyro.param(p).squeeze().dim() == 0:
                    self.writer_scalar.add_scalar(p, pyro.param(p).squeeze().item(), self.epoch_count)
                elif pyro.param(p).squeeze().dim() == 1 and pyro.param(p).squeeze().shape[0] <= self.K:
                    scalars = {str(i): pyro.param(p).squeeze()[i].item() for i in range(pyro.param(p).squeeze().size()[-1])}
                    self.writer_scalar.add_scalars("{}".format(p), scalars, self.epoch_count)
            if self.mcc:
                #import pdb; pdb.set_trace()
                if self.__name__ == "marginal":
                    guide_trace = poutine.trace(self.guide).get_trace()
                    trained_model = poutine.replay(poutine.enum(self.model), trace=guide_trace)
                    inferred_model = infer_discrete(trained_model, temperature=0,
                                                    first_available_dim=-6)
                    trace = poutine.trace(inferred_model).get_trace()
                    predictions = trace.nodes["d/z"]["value"].cpu().reshape(-1)
                    true_labels = self.data.labels["spotpicker"].values.reshape(self.data.N,self.data.F)[self.batch_idx.cpu()].reshape(-1)
                elif self.__name__ == "tracker":
                    self.data.labels["probs"] = param("d/z_probs")[...,1].reshape(-1).data.cpu()
                    self.data.labels["binary"] = self.data.labels["probs"] > 0.5
                    predictions = self.data.labels["binary"].values
                    true_labels = self.data.labels["spotpicker"].values
                mcc = matthews_corrcoef(true_labels, predictions)
                tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
                scalars = {}
                scalars["MCC"] = mcc
                scalars["TNR"] = tn / (tn + fp)
                scalars["TPR"] = tp / (tp + fn)
                self.writer_scalar.add_scalars("ACCURACY", scalars, self.epoch_count)
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
