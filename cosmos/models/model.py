import torch
import torch.nn as nn
import torch.distributions.constraints as constraints
import numpy as np
import os
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, infer_discrete
from pyro.infer import JitTraceEnum_ELBO, TraceEnum_ELBO
from pyro import param
from cosmos.models.noise import _noise_fn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from tqdm import tqdm
from cosmos.models.helper import ScaledBeta
import logging
from pyro.ops.indexing import Vindex
from pyro import poutine
from cosmos.models.helper import z_probs_calc
import cosmos
from git import Repo


class GaussianSpot(nn.Module):
    def __init__(self, data, K):
        super().__init__()
        self.K = K
        # create meshgrid of DxD pixel positions
        x_pixel, y_pixel, _ = torch.meshgrid(
            torch.arange(data.D), torch.arange(data.D), torch.arange(1))
        self.pixel_pos = torch.stack((x_pixel, y_pixel), dim=-1).float()

        # drift locs for 2D gaussian spot
        self.target_locs = torch.tensor(
            data.drift[["dx", "dy"]].values.reshape(data.F, 1, 1, 1, 2)
            + data.target[["x", "y"]].values.reshape(data.N, 1, 1, 1, 1, 2)) \
            .float()

    # Ideal 2D gaussian spots
    def forward(self, batch_idx, m_mask, height, width, x0, y0, background):
        height = height.masked_fill(~m_mask, 0.)
        spot_locs = self.target_locs[batch_idx] + torch.stack((x0, y0), dim=-1)
        rv = dist.MultivariateNormal(
            spot_locs,
            scale_tril=torch.eye(2) * width[..., None, None])
        gaussian_spot = torch.exp(rv.log_prob(self.pixel_pos))  # N,F,D,D
        return (height * gaussian_spot).sum(dim=-1, keepdim=True) + background


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

        self.size = torch.tensor([2., (((data.D+3) / (2*0.5)) ** 2 - 1)])
        self.m_matrix = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]])
        self.theta_matrix = torch.tensor([[0, 0], [1, 0], [0, 1]])

        self.data.loc = GaussianSpot(self.data, self.K)
        if self.control:
            self.control.loc = GaussianSpot(self.control, self.K)

        pyro.clear_param_store()
        self.epoch_count = 0
        self.optim_fn = pyro.optim.Adam
        self.optim_args = {"lr": self.lr, "betas": [0.9, 0.999]}
        self.optim = self.optim_fn(self.optim_args)
        self.elbo = (JitTraceEnum_ELBO if jit else TraceEnum_ELBO)(
            max_plate_nesting=5, ignore_jit_warnings=True)
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)

        self.log()

    def model(self):
        raise NotImplementedError

    def guide(self):
        raise NotImplementedError

    def spot_model(self, data, m_pi, theta_pi, prefix):
        N_plate = pyro.plate("N_plate", data.N, dim=-5)
        F_plate = pyro.plate("F_plate", data.F, dim=-4)
        X_plate = pyro.plate("X_plate", data.D, dim=-3)
        Y_plate = pyro.plate("Y_plate", data.D, dim=-2)
        K_plate = pyro.plate("K_plate", self.K, dim=-1)

        with N_plate as batch_idx, F_plate:
            background = pyro.sample(
                "background", dist.Gamma(
                    param(f"{prefix}/background_loc")[batch_idx]
                    * param("background_beta"), param("background_beta")))

            m = pyro.sample("m", dist.Categorical(m_pi))
            if theta_pi is not None:
                theta = pyro.sample("theta", dist.Categorical(theta_pi[m]))
                theta_mask = self.theta_matrix[theta.squeeze(dim=-1)]
            else:
                theta_mask = 0
            m_mask = self.m_matrix[m.squeeze(dim=-1)].bool()

            with K_plate, pyro.poutine.mask(mask=m_mask):
                height = pyro.sample(
                    "height", dist.Gamma(
                        param("height_loc") * param("height_beta"),
                        param("height_beta")))
                width = pyro.sample(
                    "width", ScaledBeta(
                        param("width_mode"),
                        param("width_size"), 0.5, 2.5))
                x0 = pyro.sample(
                    "x0", ScaledBeta(
                        0, self.size[theta_mask], -(data.D+3)/2, data.D+3))
                y0 = pyro.sample(
                    "y0", ScaledBeta(
                        0, self.size[theta_mask], -(data.D+3)/2, data.D+3))

            width = width * 2.5 + 0.5
            x0 = x0 * (data.D+3) - (data.D+3)/2
            y0 = y0 * (data.D+3) - (data.D+3)/2

            locs = data.loc(batch_idx, m_mask,
                            height, width, x0, y0, background)
            with X_plate, Y_plate:
                pyro.sample(
                    "data", self.CameraUnit(
                        locs, param("gain"), param("offset")),
                    obs=data[batch_idx].unsqueeze(dim=-1))

    def spot_guide(self, data, theta, prefix):
        N_plate = pyro.plate("N_plate", data.N,
                             subsample_size=self.n_batch, dim=-5)
        F_plate = pyro.plate("F_plate", data.F, dim=-4)
        K_plate = pyro.plate("K_plate", self.K, dim=-1)

        with N_plate as batch_idx, F_plate:
            self.batch_idx = batch_idx.cpu()
            pyro.sample(
                "background", dist.Gamma(
                    param(f"{prefix}/b_loc")[batch_idx]
                    * param(f"{prefix}/b_beta")[batch_idx],
                    param(f"{prefix}/b_beta")[batch_idx]))
            m = pyro.sample("m", dist.Categorical(
                param(f"{prefix}/m_probs")[batch_idx]))
            if theta:
                pyro.sample("theta", dist.Categorical(
                    Vindex(param(
                        f"{prefix}/theta_probs")[batch_idx])[..., m, :]))
            m_mask = self.m_matrix[m.squeeze(dim=-1)].bool()

            with K_plate, pyro.poutine.mask(mask=m_mask):
                pyro.sample(
                    "height", dist.Gamma(
                        param(f"{prefix}/h_loc")[batch_idx]
                        * param(f"{prefix}/h_beta")[batch_idx],
                        param(f"{prefix}/h_beta")[batch_idx]))
                pyro.sample(
                    "width", ScaledBeta(
                        param(f"{prefix}/w_mode")[batch_idx],
                        param(f"{prefix}/w_size")[batch_idx],
                        0.5, 2.5))
                pyro.sample(
                    "x0", ScaledBeta(
                        param(f"{prefix}/x_mode")[batch_idx],
                        param(f"{prefix}/size")[batch_idx],
                        -(data.D+3)/2, data.D+3))
                pyro.sample(
                    "y0", ScaledBeta(
                        param(f"{prefix}/y_mode")[batch_idx],
                        param(f"{prefix}/size")[batch_idx],
                        -(data.D+3)/2, data.D+3))

    def spot_parameters(self, data, m, theta, prefix):
        param(f"{prefix}/background_loc",
              torch.ones(data.N, 1, 1, 1, 1) * 50.,
              constraint=constraints.positive)
        if m:
            param(f"{prefix}/m_probs",
                  torch.ones(data.N, data.F, 1, 1, 1, 2**self.K),
                  constraint=constraints.simplex)
        if theta:
            theta_probs = torch.ones(
                data.N, data.F, 1, 1, 1, 2**self.K, self.K+1)
            theta_probs[..., 0, 1:] = 0
            theta_probs[..., 1, 2] = 0
            theta_probs[..., 2, 1] = 0
            param(f"{prefix}/theta_probs", theta_probs,
                  constraint=constraints.simplex)
        param(f"{prefix}/b_loc",
              torch.ones(data.N, data.F, 1, 1, 1) * 50.,
              constraint=constraints.positive)
        param(f"{prefix}/b_beta",
              torch.ones(data.N, data.F, 1, 1, 1) * 30,
              constraint=constraints.positive)
        param(f"{prefix}/h_loc",
              torch.ones(data.N, data.F, 1, 1, self.K) * 1000.,
              constraint=constraints.positive)
        param(f"{prefix}/h_beta",
              torch.ones(data.N, data.F, 1, 1, self.K),
              constraint=constraints.positive)
        param(f"{prefix}/w_mode",
              torch.ones(data.N, data.F, 1, 1, self.K) * 1.3,
              constraint=constraints.interval(0.5, 3.))
        param(f"{prefix}/w_size",
              torch.ones(data.N, data.F, 1, 1, self.K) * 100.,
              constraint=constraints.greater_than(2.))
        param(f"{prefix}/x_mode",
              torch.zeros(data.N, data.F, 1, 1, self.K),
              constraint=constraints.interval(-(data.D+3)/2, (data.D+3)/2))
        param(f"{prefix}/y_mode",
              torch.zeros(data.N, data.F, 1, 1, self.K),
              constraint=constraints.interval(-(data.D+3)/2, (data.D+3)/2))
        size = torch.ones(data.N, data.F, 1, 1, self.K) * 5.
        size[..., 0] = ((data.D+3) / (2*0.5)) ** 2 - 1
        param(f"{prefix}/size",
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
        param("width_mode", torch.tensor([1.3]),
              constraint=constraints.interval(0.5, 3.))
        param("width_size",
              torch.tensor([10.]), constraint=constraints.positive)
        param("pi", torch.ones(2), constraint=constraints.simplex)
        param("lamda", torch.tensor([0.1]), constraint=constraints.positive)

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
            # import pdb; pdb.set_trace()
            self.epoch_loss = self.svi.step()
            if not self.epoch_count % 100:
                self.save_checkpoint()
            self.epoch_count += 1

    def log(self):
        repo = Repo(cosmos.__path__[0], search_parent_directories=True)
        version = repo.git.describe()
        self.path = os.path.join(
            self.data.path, "runs",
            "{}{}".format(self.__name__, version),
            "{}".format("jit" if self.jit else "nojit"),
            "lr{}".format(self.lr), "{}".format(self.optim_fn.__name__),
            "{}".format(self.n_batch))
        self.writer_scalar = SummaryWriter(
            log_dir=os.path.join(self.path, "scalar"))
        self.writer_hist = SummaryWriter(
            log_dir=os.path.join(self.path, "hist"))

        self.logger = logging.getLogger(__name__)
        fh = logging.FileHandler(os.path.join(
            self.path, "run.log"))
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S %p")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
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

    def save_checkpoint(self):
        if not any([torch.isnan(v).any()
                   for v in pyro.get_param_store().values()]):
            self.optim.save(os.path.join(self.path, "optimizer"))
            pyro.get_param_store().save(os.path.join(self.path, "params"))
            np.savetxt(os.path.join(self.path, "epoch_count"),
                       np.array([self.epoch_count]))

            self.writer_scalar.add_scalar(
                "-ELBO", self.epoch_loss, self.epoch_count)
            for p in pyro.get_param_store().keys():
                if param(p).squeeze().dim() == 0:
                    self.writer_scalar.add_scalar(
                        p, pyro.param(p).squeeze().item(), self.epoch_count)
                elif param(p).squeeze().dim() == 1 and \
                        len(param(p).squeeze()) <= self.K:
                    scalars = {str(i): p
                               for i, p in enumerate(param(p).squeeze())}
                    self.writer_scalar.add_scalars(
                        "{}".format(p), scalars, self.epoch_count)
            if self.mcc:
                if self.__name__ == "marginal":
                    guide_trace = poutine.trace(self.guide).get_trace()
                    trained_model = poutine.replay(
                        poutine.enum(self.model), trace=guide_trace)
                    inferred_model = infer_discrete(
                        trained_model, temperature=0, first_available_dim=-6)
                    trace = poutine.trace(inferred_model).get_trace()
                    predictions = (
                        trace.nodes["d/theta"]["value"] > 0) \
                        .cpu().reshape(-1)
                    true_labels = self.data.labels["spotpicker"].values \
                        .reshape(self.data.N, self.data.F)[self.batch_idx] \
                        .reshape(-1)
                elif self.__name__ == "tracker":
                    z_probs = z_probs_calc(
                        pyro.param("d/m_probs"), pyro.param("d/theta_probs"))
                    self.data.predictions = self.data.labels.copy()
                    self.data.predictions["probs"] = z_probs.reshape(-1)
                    self.data.predictions["binary"] = \
                        self.data.predictions["probs"] > 0.5
                    mask = self.data.labels["spotpicker"].values < 2
                    predictions = self.data.predictions["binary"].values[mask]
                    true_labels = self.data.labels["spotpicker"].values[mask]
                mcc = matthews_corrcoef(true_labels, predictions)
                tn, fp, fn, tp = confusion_matrix(
                    true_labels, predictions).ravel()
                #scalars = {}
                #scalars["MCC"] = mcc
                #scalars["TPR"] = tp / (tp + fn)
                negatives = {}
                negatives["TN"] = tn
                negatives["FP"] = fp
                positives = {}
                positives["TP"] = tp
                positives["FN"] = fn
                self.writer_scalar.add_scalar(
                    "MCC", mcc, self.epoch_count)
                self.writer_scalar.add_scalars(
                    "NEGATIVES", negatives, self.epoch_count)
                self.writer_scalar.add_scalars(
                    "POSITIVES", positives, self.epoch_count)
                self.data.predictions.to_csv(os.path.join(self.path, "predictions.csv"))
                if self.data.name.startswith("FL") and \
                        self.data.name.endswith("OD"):
                    atten_labels = self.data.predictions.copy()
                    atten_labels.loc[
                        self.data.labels["spotpicker"] != self.data.predictions["binary"],
                        "spotpicker"] = 2
                    atten_labels["probs"] = 0
                    atten_labels["binary"] = 0
                    atten_labels.to_csv(os.path.join(self.path, "atten_labels.csv"))

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
