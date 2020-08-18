import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import pyro
import pyro.distributions as dist
from pyro.infer import SVI
from pyro.infer import JitTraceEnum_ELBO, TraceEnum_ELBO
from pyro.ops.indexing import Vindex
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import matthews_corrcoef, confusion_matrix, \
    recall_score, precision_score
import logging
from cosmos import __version__ as cosmos_version
from tqdm import tqdm
from cosmos.utils.dataset import load_data


class GaussianSpot(nn.Module):
    def __init__(self, target, drift, D):
        super().__init__()
        # create meshgrid of DxD pixel positions
        D_range = torch.arange(D, dtype=torch.float)
        i_pixel, j_pixel = torch.meshgrid(D_range, D_range)
        self.ij_pixel = torch.stack((i_pixel, j_pixel), dim=-1)

        # drift locs for 2D gaussian spot
        self.target_locs = torch.tensor(
            drift[["dx", "dy"]].values.reshape(-1, 2)
            + target[["x", "y"]].values.reshape(-1, 1, 2),
        ).float()

    # Ideal 2D gaussian spots
    def forward(self, height, width, x, y, n_idx, f_idx):
        spot_locs = Vindex(self.target_locs)[n_idx, f_idx, :] + torch.stack((x, y), -1)
        rv = dist.MultivariateNormal(
            spot_locs[..., None, None, :],
            scale_tril=torch.eye(2) * width[..., None, None, None, None]
        )
        gaussian_spot = torch.exp(rv.log_prob(self.ij_pixel))  # N,F,D,D
        return height[..., None, None] * gaussian_spot


class Model(nn.Module):
    """ Gaussian Spot Model """

    def __init__(self, S, K=2):
        super().__init__()
        self.K = K
        self.S = S
        self.batch_size = None
        # for plotting
        self.n = None
        self.frames = None

    def load(self, path, control, device):
        # set path
        self.data_path = path

        # set device
        self.device = torch.device(device)

        # load test data
        self.data = load_data(self.data_path, dtype="test", device=self.device)
        self.data_loc = GaussianSpot(
            self.data.target, self.data.drift,
            self.data.D)

        self.data_median = torch.median(self.data.data)
        self.offset_median = torch.median(self.data.offset)
        self.noise = (self.data.data.std(dim=(1, 2, 3)).mean() - self.data.offset.std()) * np.pi * (2 * 1.3) ** 2
        offset_max = np.percentile(self.data.offset.cpu().numpy(), 99.5)
        offset_min = np.percentile(self.data.offset.cpu().numpy(), 0.5)
        offset_weights, offset_samples = np.histogram(self.data.offset.cpu().numpy(),
                                                      range=(offset_min, offset_max),
                                                      bins=max(1, int(offset_max - offset_min)),
                                                      density=True)
        offset_samples = offset_samples[:-1]
        self.offset_samples = torch.from_numpy(offset_samples).float().to(device)
        self.offset_weights = torch.from_numpy(offset_weights).float().to(device)

        # load control data
        if control:
            self.control = load_data(self.data_path, dtype="control", device=self.device)
            self.control_loc = GaussianSpot(
                self.control.target, self.control.drift,
                self.control.D)
        else:
            self.control = control

        self.size = torch.ones(self.S+1) * (((self.data.D+1) / (2*0.5)) ** 2 - 1)
        self.size[0] = 2.
        self.theta_matrix = torch.zeros(1 + self.K * self.S, self.K).long()
        for s in range(self.S):
            for k in range(self.K):
                self.theta_matrix[1 + s*self.K + k, k] = s + 1

    def settings(self, lr, batch_size, jit=False):
        # K - max number of spots
        self.lr = lr
        self.batch_size = batch_size
        self.jit = jit

        self.optim_fn = pyro.optim.Adam
        self.optim_args = {"lr": self.lr, "betas": [0.9, 0.999]}
        self.optim = self.optim_fn(self.optim_args)
        self.log()

        try:
            self.load_checkpoint()
        except:
            pyro.clear_param_store()
            self.model_parameters()
            self.guide_parameters()

            self.iter = 0

            self.predictions = np.zeros(
                (self.data.N, self.data.F),
                dtype=[("z", bool),
                       ("z_prob", float), ("m", bool, (2,)),
                       ("m_prob", float, (2,)), ("theta", int)])

        self.elbo = (JitTraceEnum_ELBO if jit else TraceEnum_ELBO)(
            max_plate_nesting=2, ignore_jit_warnings=True)
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)

    def model(self):
        raise NotImplementedError

    def guide(self):
        raise NotImplementedError

    def run(self, num_iter):
        for i in tqdm(range(num_iter)):
            # with torch.autograd.detect_anomaly():
            # import pdb; pdb.set_trace()
            self.iter_loss = self.svi.step()
            if not self.iter % 100:
                self.infer()
                self.save_checkpoint()
            self.iter += 1

    def log(self):
        self.path = os.path.join(
            self.data_path, "runs",
            "{}".format(self.__name__),
            "{}".format(cosmos_version),
            "S{}".format(self.S),
            "{}".format("control" if self.control else "nocontrol"),
            "lr{}".format(self.lr),
            "bs{}".format(self.batch_size))
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.path, "scalar"))

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
        self.logger.info("Batch size - {}".format(self.batch_size))
        self.logger.info("{}".format("jit" if self.jit else "nojit"))

    def save_checkpoint(self):
        # save only if no NaN values
        if any([torch.isnan(v).any()
                for v in pyro.get_param_store().values()]):
            self.logger.warning("Step #{}. Detected NaN values in parameters".format(self.iter))
            return

        self.optim.save(os.path.join(self.path, "optimizer"))
        pyro.get_param_store().save(os.path.join(self.path, "params"))
        np.savetxt(os.path.join(self.path, "iter"),
                   np.array([self.iter]))
        params_last = pd.Series(data={"iter": self.iter})

        self.writer.add_scalar(
            "-ELBO", self.iter_loss, self.iter)
        params_last["-ELBO"] = self.iter_loss
        for name, val in pyro.get_param_store().items():
            if val.dim() == 0:
                self.writer.add_scalar(name, val.item(), self.iter)
                params_last[name] = val.item()
            elif val.dim() == 1 and len(val) <= self.S+1:
                scalars = {str(i): v.item() for i, v in enumerate(val)}
                self.writer.add_scalars(name, scalars, self.iter)
                for key, value in scalars.items():
                    params_last["{}_{}".format(name, key)] = value

        if self.data.labels is not None:
            mask = self.data.labels["z"] < 2
            pred_labels = self.predictions["z"][mask]
            true_labels = self.data.labels["z"][mask]

            metrics = {}
            with np.errstate(divide="ignore", invalid="ignore"):
                metrics["MCC"] = matthews_corrcoef(true_labels, pred_labels)
            metrics["Recall"] = recall_score(true_labels, pred_labels, zero_division=0)
            metrics["Precision"] = precision_score(true_labels, pred_labels, zero_division=0)

            neg, pos = {}, {}
            neg["TN"], neg["FP"], pos["FN"], pos["TP"] = confusion_matrix(
                true_labels, pred_labels).ravel()

            self.writer.add_scalars(
                "ACCURACY", metrics, self.iter)
            self.writer.add_scalars(
                "NEGATIVES", neg, self.iter)
            self.writer.add_scalars(
                "POSITIVES", pos, self.iter)
            for key, value in {**metrics, **pos, **neg}.items():
                params_last[key] = value
            try:
                atten_labels = np.copy(self.data.labels)
                atten_labels["z"][
                    self.data.labels["spotpicker"] != self.predictions["z"]] = 2
                atten_labels["spotpicker"] = 0
                np.save(os.path.join(self.path, "atten_labels.npy"),
                        atten_labels)
            except:
                pass

        params_last.to_csv(os.path.join(self.path, "params_last.csv"))
        self.logger.info("Step #{}.".format(self.iter))

    def load_checkpoint(self, path=None):
        if path is None:
            path = self.path
        self.iter = int(np.loadtxt(os.path.join(path, "iter")))
        self.optim.load(os.path.join(path, "optimizer"))
        pyro.clear_param_store()
        pyro.get_param_store().load(
            os.path.join(path, "params"),
            map_location=self.device)
        self.predictions = np.load(os.path.join(path, "predictions.npy"))
        self.logger.info(
            "Step #{}. Loaded model params and optimizer state from {}"
            .format(self.iter, path))

    def load_parameters(self, path=None):
        if path is None:
            path = self.path
        self.iter = int(
            np.loadtxt(os.path.join(path, "iter")))
        pyro.clear_param_store()
        pyro.get_param_store().load(
            os.path.join(path, "params"),
            map_location=self.device)
        self.predictions = np.load(os.path.join(path, "predictions.npy"))
