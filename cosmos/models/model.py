import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import pyro
import pyro.distributions as dist
from pyro.infer import SVI
from pyro.infer import JitTraceEnum_ELBO, TraceEnum_ELBO
from pyro import param
from pyro.ops.indexing import Vindex
from cosmos.models.noise import _noise_fn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import matthews_corrcoef, confusion_matrix, \
    recall_score, precision_score
from tqdm import tqdm
import logging
import cosmos
from cosmos.utils.glimpse_reader import load_data


class GaussianSpot(nn.Module):
    def __init__(self, target, drift, D, device):
        super().__init__()
        # create meshgrid of DxD pixel positions
        D_range = torch.arange(D, dtype=torch.float, device=device)
        i_pixel, j_pixel = torch.meshgrid(D_range, D_range)
        self.ij_pixel = torch.stack((i_pixel, j_pixel), dim=-1)

        # drift locs for 2D gaussian spot
        self.target_locs = torch.tensor(
            drift[["dx", "dy"]].values.reshape(-1, 2)
            + target[["x", "y"]].values.reshape(-1, 1, 2),
            device=device, dtype=torch.float)

    # Ideal 2D gaussian spots
    def forward(self, height, width, x, y, background, n_idx, f_idx):
        #if f is not None:
        #n_idx = n_idx[:, None]
        #import pdb; pdb.set_trace()
        spot_locs = Vindex(self.target_locs)[n_idx, f_idx, :] + torch.stack((x, y), -1)
        #else:
        #    spot_locs = self.target_locs[n_idx] + torch.stack((x, y), -1)
        rv = dist.MultivariateNormal(
            spot_locs[..., None, None, :],
            scale_tril=torch.eye(2, device=width.device) * width[..., None, None, None, None])
        gaussian_spot = torch.exp(rv.log_prob(self.ij_pixel))  # N,F,D,D
        result = (height[..., None, None] * gaussian_spot).sum(dim=-5, keepdim=True) \
            + background[..., None, None]
        return result


class Model(nn.Module):
    """ Gaussian Spot Model """
    def __init__(self, K=2):
        super().__init__()
        self.K = K
        self.S = 1
        self.batch_size = None
        # for plotting
        self.n = None
        self.frames = None

    def load(self, path, control, device):
        # set path
        self.path = path

        # set device
        self.device = torch.device(device)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

        # load test data
        self.data = load_data(self.path, dtype="test", device=self.device)
        self.data.loc = GaussianSpot(
            self.data.target, self.data.drift,
            self.data.D, self.device)

        self.offset_guess = self.data.offset_mean

        # load control data
        if control:
            self.control = load_data(self.path, dtype="control", device=self.device)
            self.control.loc = GaussianSpot(
                self.control.target, self.control.drift,
                self.control.D, self.device)
        else:
            self.control = control

        self.size = torch.tensor([2., (((self.data.D+1) / (2*0.5)) ** 2 - 1)], device=self.device)
        self.m_matrix = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], device=self.device).T.reshape(2,1,1,4)
        self.theta_matrix = torch.tensor([[0, 0], [1, 0], [0, 1]], device=self.device).T.reshape(2,1,1,3)

    def settings(self, lr, batch_size, jit=False):
        # K - max number of spots
        self.lr = lr
        self.batch_size = batch_size
        self.jit = jit

        self.predictions = np.zeros(
            (self.data.N, self.data.F),
            dtype=[("z", bool),
                   ("z_prob", float), ("m", bool, (2,)),
                   ("m_prob", float, (2,)), ("theta", int)])

        pyro.clear_param_store()
        self.iter = 0
        self.optim_fn = pyro.optim.Adam
        self.optim_args = {"lr": self.lr, "betas": [0.9, 0.999]}
        self.optim = self.optim_fn(self.optim_args)
        self.elbo = (JitTraceEnum_ELBO if jit else TraceEnum_ELBO)(
            max_plate_nesting=3, ignore_jit_warnings=True)
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)

        self.log()

        try:
            self.load_checkpoint()
        except:
            pass

    def model(self):
        raise NotImplementedError

    def guide(self):
        raise NotImplementedError

    def run(self, num_iter):
        for i in range(num_iter):
            # with torch.autograd.detect_anomaly():
            self.iter_loss = self.svi.step()
            if not self.iter % 100:
                self.infer()
                self.save_checkpoint()
            self.iter += 1

    def log(self):
        version = cosmos.__version__
        self.path = os.path.join(
            self.path, "runs",
            "{}{}".format(self.__name__, version),
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
            elif val.dim() == 1:
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
