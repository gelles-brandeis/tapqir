import torch
import torch.nn as nn
import numpy as np
import pandas as pd
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
import logging
from pyro import poutine
from cosmos.models.helper import z_probs_calc, k_probs_calc
import cosmos
from git import Repo


class GaussianSpot(nn.Module):
    def __init__(self, target, drift, D, K):
        super().__init__()
        self.K = K
        # create meshgrid of DxD pixel positions
        x_pixel, y_pixel, _ = torch.meshgrid(
            torch.arange(D), torch.arange(D), torch.arange(1))
        self.pixel_pos = torch.stack((x_pixel, y_pixel), dim=-1).float()

        # drift locs for 2D gaussian spot
        N, F = len(target), len(drift)
        self.target_locs = torch.tensor(
            drift[["dx", "dy"]].values.reshape(F, 1, 1, 1, 2)
            + target[["x", "y"]].values.reshape(N, 1, 1, 1, 1, 2)) \
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
    def __init__(self, data, control, path,
                 K, lr, n_batch, jit,
                 noise="GammaOffset"):
        # D - number of pixel along axis
        # K - number of states
        self.data = data
        self.control = control
        self.path = path
        self.K = K
        self.CameraUnit = _noise_fn[noise]
        self.lr = lr
        self.n_batch = n_batch
        self.jit = jit

        self.size = torch.tensor([2., (((data.D+3) / (2*0.5)) ** 2 - 1)])
        self.m_matrix = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]])
        self.theta_matrix = torch.tensor([[0, 0], [1, 0], [0, 1]])

        self.data.predictions = np.zeros(
            (data.N, data.F),
            dtype=[("aoi", int), ("frame", int), ("z", bool), ("z_prob", float),
                   ("m0", bool), ("m1", bool), ("m0_prob", float), ("m1_prob", float)])
        self.data.predictions["aoi"] = data.target.index.values.reshape(-1, 1)
        self.data.predictions["frame"] = data.drift.index.values


        self.data.loc = GaussianSpot(
                self.data.target, self.data.drift,
                self.data.D, self.K)
        if self.control:
            self.control.loc = GaussianSpot(
                    self.control.target, self.control.drift,
                    self.control.D, self.K)

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

    def train(self, num_steps):
        for epoch in tqdm(range(num_steps)):
            # with torch.autograd.detect_anomaly():
            # import pdb; pdb.set_trace()
            self.epoch_loss = self.svi.step()
            if not self.epoch_count % 100:
                self.infer()
                self.save_checkpoint()
            self.epoch_count += 1

    def log(self):
        repo = Repo(cosmos.__path__[0], search_parent_directories=True)
        version = repo.git.describe()
        self.path = os.path.join(
            self.path, "runs",
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

    def infer(self):
        if self.__name__ == "marginal":
            #n_batch_save = self.n_batch
            #self.n_batch = self.data.N
            guide_trace = poutine.trace(self.guide).get_trace()
            trained_model = poutine.replay(
                poutine.enum(self.model), trace=guide_trace)
            inferred_model = infer_discrete(
                trained_model, temperature=0, first_available_dim=-6)
            trace = poutine.trace(inferred_model).get_trace()
            self.data.predictions["z"][self.batch_idx] = (
                trace.nodes["d/theta"]["value"] > 0) \
                .cpu().data.squeeze()
            self.data.predictions["m0"][self.batch_idx] = \
                self.m_matrix[trace.nodes["d/m"]["value"]].squeeze().cpu().data[..., 0]
            self.data.predictions["m1"][self.batch_idx] = \
                self.m_matrix[trace.nodes["d/m"]["value"]].squeeze().cpu().data[..., 1]
        elif self.__name__ == "tracker":
            z_probs = z_probs_calc(
                pyro.param("d/m_probs"), pyro.param("d/theta_probs"))
            k_probs = k_probs_calc(pyro.param("d/m_probs"))
            self.data.predictions["z_prob"] = z_probs.squeeze()
            self.data.predictions["z"] = \
                self.data.predictions["z_prob"] > 0.5
            self.data.predictions["m0_prob"] = k_probs.squeeze()[..., 0]
            self.data.predictions["m1_prob"] = k_probs.squeeze()[..., 1]
            self.data.predictions["m0"] = self.data.predictions["m0_prob"] > 0.5
            self.data.predictions["m1"] = self.data.predictions["m1_prob"] > 0.5
        np.save(os.path.join(self.path, "predictions.npy"),
                self.data.predictions)

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
            if self.data.labels is not None:
                mask = self.data.labels["z"] < 2
                predictions = self.data.predictions["z"][mask]
                true_labels = self.data.labels["z"][mask]
                mcc = matthews_corrcoef(true_labels, predictions)
                tn, fp, fn, tp = confusion_matrix(
                    true_labels, predictions).ravel()
                scalars = {}
                scalars["MCC"] = mcc
                scalars["TPR"] = tp / (tp + fn)
                scalars["TNR"] = tn / (tn + fp)
                negatives = {}
                negatives["TN"] = tn
                negatives["FP"] = fp
                positives = {}
                positives["TP"] = tp
                positives["FN"] = fn
                self.writer_scalar.add_scalars(
                    "ACCURACY", scalars, self.epoch_count)
                self.writer_scalar.add_scalars(
                    "NEGATIVES", negatives, self.epoch_count)
                self.writer_scalar.add_scalars(
                    "POSITIVES", positives, self.epoch_count)
                #if self.data.name.startswith("FL") and \
                #        self.data.name.endswith("OD"):
                #    atten_labels = self.data.labels.copy()
                #    atten_labels.loc[
                #        self.data.labels["spotpicker"] != self.data.predictions["binary"],
                #        "binary"] = 2
                #    atten_labels["spotpicker"] = 0
                #    atten_labels.to_csv(os.path.join(self.path, "atten_labels.csv"))

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
            pyro.clear_param_store()
            pyro.get_param_store().load(
                    os.path.join(self.path, "params"),
                    map_location=self.data.device)
            self.logger.info(
                    "Step #{}. Loaded model params and optimizer state from {}"
                    .format(self.epoch_count, self.path))
        except:
            pass
