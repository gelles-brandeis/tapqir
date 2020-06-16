import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import pyro
import pyro.distributions as dist
from pyro.infer import SVI
from pyro import param
from cosmos.models.noise import _noise_fn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from tqdm import tqdm
import logging
import cosmos
from git import Repo


class GaussianSpot(nn.Module):
    def __init__(self, target, drift, D):
        super().__init__()
        # create meshgrid of DxD pixel positions
        i_pixel, j_pixel = torch.meshgrid(
            torch.arange(D), torch.arange(D))
        self.ij_pixel = torch.stack((i_pixel, j_pixel), dim=-1).float()

        # drift locs for 2D gaussian spot
        self.target_locs = torch.tensor(
            drift[["dx", "dy"]].values.reshape(-1, 2)
            + target[["x", "y"]].values.reshape(-1, 1, 2)) \
            .float()

    # Ideal 2D gaussian spots
    def forward(self, height, width, x, y, background, n_idx, f=None):
        if f is not None:
            spot_locs = self.target_locs[n_idx, f] + torch.stack((x, y), -1)
        else:
            spot_locs = self.target_locs[n_idx] + torch.stack((x, y), -1)
        rv = dist.MultivariateNormal(
            spot_locs[..., None, None, :],
            scale_tril=torch.eye(2) * width[..., None, None, None, None])
        gaussian_spot = torch.exp(rv.log_prob(self.ij_pixel))  # N,F,D,D
        result = (height[..., None, None] * gaussian_spot).sum(dim=-5, keepdim=True) \
            + background[..., None, None]
        return result


class Model(nn.Module):
    """ Gaussian Spot Model """
    def __init__(self, data, control, path,
                 K, lr, n_batch, jit,
                 noise="GammaOffset"):
        super().__init__()
        # D - number of pixels along axis
        # K - max number of spots
        # pyro.set_rng_seed(0)
        self.data = data
        self.control = control
        self.path = path
        self.K = K
        self.S = 1
        self.CameraUnit = _noise_fn[noise]
        self.lr = lr
        self.n_batch = n_batch
        self.jit = jit

        self.size = torch.tensor([2., (((data.D+1) / (2*0.5)) ** 2 - 1)])
        self.m_matrix = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]]).T.reshape(2,1,1,4)
        self.theta_matrix = torch.tensor([[0, 0], [1, 0], [0, 1]]).T.reshape(2,1,1,3)
        """
        self.theta_matrix = \
            torch.tensor([[0, 0],
                          [1, 0],
                          #[1, 0],
                          #[0, 1],
                          #[0, 1]]).T.reshape(2, 1, 1, 5)
                          [0, 1]]).T.reshape(2, 1, 1, 3)
        self.m_matrix = \
            torch.tensor([[0, 0],
                          [1, 0],
                          #[2, 0],
                          #[0, 1],
                          #[0, 2]]).T.reshape(2, 1, 1, 5)
                          [0, 1]]).T.reshape(2, 1, 1, 3)
        """
        self.m_state = \
            torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1],
                          [1, 0], [1, 1], [0, 1], [1, 1]]) \
            .T.reshape(2, 1, 1, 8)
        self.theta_state = \
            torch.tensor([[0, 0], [0, 0], [0, 0], [0, 0],
                          [1, 0], [1, 0], [0, 1], [0, 1]]) \
            .T.reshape(2, 1, 1, 8)

        self.predictions = np.zeros(
            (data.N, data.F),
            dtype=[("aoi", int), ("frame", int), ("z", bool),
                   ("z_prob", float), ("m", bool, (2,)),
                   ("m_prob", float, (2,)), ("theta", int)])
        self.predictions["aoi"] = data.target.index.values.reshape(-1, 1)
        self.predictions["frame"] = data.drift.index.values

        self.data.loc = GaussianSpot(
            self.data.target, self.data.drift,
            self.data.D)
        if self.control:
            self.control.loc = GaussianSpot(
                self.control.target, self.control.drift,
                self.control.D)

        pyro.clear_param_store()
        self.epoch_count = 0
        self.optim_fn = pyro.optim.Adam
        self.optim_args = {"lr": self.lr, "betas": [0.9, 0.999]}
        self.optim = self.optim_fn(self.optim_args)
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)

        self.log()

    def model(self):
        raise NotImplementedError

    def guide(self):
        raise NotImplementedError

    def train(self, num_steps):
        for epoch in tqdm(range(num_steps)):
            # with torch.autograd.detect_anomaly():
            #import pdb; pdb.set_trace()
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

    def save_checkpoint(self):
        if not any([torch.isnan(v).any()
                   for v in pyro.get_param_store().values()]):
            self.optim.save(os.path.join(self.path, "optimizer"))
            pyro.get_param_store().save(os.path.join(self.path, "params"))
            np.savetxt(os.path.join(self.path, "epoch_count"),
                       np.array([self.epoch_count]))
            params_last = pd.Series(data={"epoch_count": self.epoch_count})

            self.writer_scalar.add_scalar(
                "-ELBO", self.epoch_loss, self.epoch_count)
            params_last["-ELBO"] = self.epoch_loss
            for p in pyro.get_param_store().keys():
                if param(p).squeeze().dim() == 0:
                    self.writer_scalar.add_scalar(
                        p, pyro.param(p).squeeze().item(), self.epoch_count)
                    params_last[p] = pyro.param(p).item()
                elif param(p).squeeze().dim() == 1 and \
                        len(param(p).squeeze()) <= self.S+1:
                    scalars = {str(i): p.item()
                               for i, p in enumerate(param(p).squeeze())}
                    self.writer_scalar.add_scalars(
                        "{}".format(p), scalars, self.epoch_count)
                    for key, value in scalars.items():
                        params_last["{}_{}".format(p, key)] = value
                elif param(p).squeeze().dim() == 2 and \
                        len(param(p).squeeze()) <= self.S+1:
                    scalars = {str(i): p.item()
                               for i, p in enumerate(param(p)[0].squeeze())}
                    self.writer_scalar.add_scalars(
                        "{}_0".format(p), scalars, self.epoch_count)
                    for key, value in scalars.items():
                        params_last["{}_{}_0".format(p, key)] = value
                    scalars = {str(i): p.item()
                               for i, p in enumerate(param(p)[1].squeeze())}
                    self.writer_scalar.add_scalars(
                        "{}_1".format(p), scalars, self.epoch_count)
                    for key, value in scalars.items():
                        params_last["{}_{}_1".format(p, key)] = value
            if self.data.labels is not None:
                mask = self.data.labels["z"] < 2
                predictions = self.predictions["z"][mask]
                true_labels = self.data.labels["z"][mask]
                mcc = matthews_corrcoef(true_labels, predictions)
                tn, fp, fn, tp = confusion_matrix(
                    true_labels, predictions).ravel()
                fuzzy_tp = self.predictions["z_prob"][self.data.labels["z"] == 1].sum()
                fuzzy_tn = (1 - self.predictions["z_prob"][self.data.labels["z"] == 0]).sum()
                fuzzy_fp = self.predictions["z_prob"][self.data.labels["z"] == 0].sum()
                fuzzy_fn = (1 - self.predictions["z_prob"][self.data.labels["z"] == 1]).sum()
                scalars = {}
                scalars["fuzzy_MCC"] = (fuzzy_tp*fuzzy_tn - fuzzy_fp*fuzzy_fn) \
                    / np.sqrt((fuzzy_tp+fuzzy_fp)*(fuzzy_tp+fuzzy_fn)*(fuzzy_tn+fuzzy_fp)*(fuzzy_tn+fuzzy_fn))
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
                for key, value in scalars.items():
                    params_last[key] = value
                for key, value in negatives.items():
                    params_last[key] = value
                for key, value in positives.items():
                    params_last[key] = value
                #import pdb; pdb.set_trace()
                #if self.data.name.startswith("FL") and \
                #        self.data.name.endswith("OD"):
                try:
                    atten_labels = np.copy(self.data.labels)
                    atten_labels["z"][
                        self.data.labels["spotpicker"] != self.predictions["z"]] = 2 
                    atten_labels["spotpicker"] = 0
                    np.save(os.path.join(self.path, "atten_labels.npy"),
                            atten_labels)
                except:
                    pass
                #    atten_labels.loc[
                #        self.data.labels["spotpicker"] != self.data.predictions["binary"],
                #        "binary"] = 2

            params_last.to_csv(os.path.join(self.path, "params_last.csv"))
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
