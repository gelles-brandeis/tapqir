# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyroapi import distributions as dist
from pyroapi import infer, optim, pyro
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from torch.distributions.utils import lazy_property
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tapqir import __version__ as tapqir_version
from tapqir.utils.dataset import load
from tapqir.utils.stats import save_stats

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    fmt="%(asctime)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)
ch.setFormatter(formatter)
logger.addHandler(ch)


class GaussianSpot:
    r"""
    Calculates ideal shape of the 2D-Gaussian spot given spot parameters
    and target positions.

        :math:`\dfrac{h}{2 \pi \cdot w^2} \exp{\left( -\dfrac{(i-x)^2 + (j-y)^2}{2 \cdot w^2} \right)}`
    """

    def __init__(self, P):
        super().__init__()
        # create meshgrid of PxP pixel positions
        P_range = torch.arange(P)
        i_pixel, j_pixel = torch.meshgrid(P_range, P_range)
        self.ij_pixel = torch.stack((i_pixel, j_pixel), dim=-1)

    # Ideal 2D gaussian spots
    def __call__(self, height, width, x, y, target_locs):
        r"""
        :param height: integrated spot intensity.
        :param width: width of the 2D-Gaussian spot.
        :param x: relative :math:`x`-axis position relative to the target.
        :param y: relative :math:`y`-axis position relative to the target.
        :param target_locs: target position.
        :return: Ideal shape 2D-Gaussian spot.
        """

        spot_locs = target_locs + torch.stack((x, y), -1)
        rv = dist.MultivariateNormal(
            spot_locs[..., None, None, :],
            scale_tril=torch.eye(2) * width[..., None, None, None, None],
        )
        gaussian_spot = torch.exp(rv.log_prob(self.ij_pixel))  # N,F,D,D
        return height[..., None, None] * gaussian_spot


class Model:
    r"""
    Base class for tapqir models.

    Derived models must implement the methods

    * :meth:`model`
    * :meth:`guide`
    """

    def __init__(self, S=1, K=2, device="cpu", dtype="double"):
        self._S = S
        self._K = K
        self.batch_size = None
        # for plotting
        self.n = None
        self.data_path = None
        self.path = None
        self.run_path = None
        # logger
        self.logger = logger
        self.logger.info("Tapqir version - {}".format(tapqir_version))
        self.logger.info("Model - {}".format(self.name))
        # set device & dtype
        self.to(device, dtype)

    def to(self, device, dtype="double"):
        self.dtype = getattr(torch, dtype)
        self.device = torch.device(device)
        if device == "cuda" and dtype == "double":
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        elif device == "cuda" and dtype == "float":
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        elif device == "cpu" and dtype == "double":
            torch.set_default_tensor_type(torch.DoubleTensor)
        else:
            torch.set_default_tensor_type(torch.FloatTensor)
        # change loaded data device
        if hasattr(self, "data"):
            self.data.ontarget = self.data.ontarget._replace(device=self.device)
            self.data.offtarget = self.data.offtarget._replace(device=self.device)
            self.data.offset = self.data.offset._replace(
                samples=self.data.offset.samples.to(self.device),
                weights=self.data.offset.weights.to(self.device),
            )
            self.gaussian.ij_pixel = self.gaussian.ij_pixel.to(self.device)
        self.logger.info("Device - {}".format(self.device))
        self.logger.info("Floating precision - {}".format(self.dtype))

    @lazy_property
    def S(self):
        r"""
        Number of distinct molecular states for the binder molecules.
        """
        return self._S

    @lazy_property
    def K(self):
        r"""
        Maximum number of spots that can be present in a single image.
        """
        return self._K

    def load(self, path):
        # set path
        self.path = Path(path)
        self.run_path = self.path / f"{self.name}" / tapqir_version.split("+")[0]
        # logger
        self.writer = SummaryWriter(log_dir=self.run_path / "tb")
        fh = logging.FileHandler(self.run_path / "run.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # load data
        self.data = load(self.path, self.device)
        self.gaussian = GaussianSpot(self.data.P)
        self.logger.info(f"Loaded data from {self.path / 'data.tpqr'}")

        # load fit results
        if (self.path / "params.tpqr").is_file():
            self.params = torch.load(self.path / "params.tpqr")
            self.logger.info(f"Loaded parameters from {self.path / 'params.tpqr'}")
        if (self.path / "statistics.csv").is_file():
            self.statistics = pd.read_csv(self.path / "statistics.csv", index_col=0)
            self.logger.info(
                f"Loaded model statistics from {self.path / 'statistics.csv'}"
            )

    def TraceELBO(self, jit):
        """
        A trace implementation of ELBO-based SVI.
        """
        raise NotImplementedError

    def model(self):
        """
        Generative Model
        """
        raise NotImplementedError

    def guide(self):
        """
        Variational Model
        """
        raise NotImplementedError

    def run(self, num_iter=70000, lr=0.005, batch_size=0, jit=False, verbose=True):
        self.lr = lr
        # find max possible batch_size
        if batch_size == 0:
            batch_size = self._max_batch_size()
        self.batch_size = batch_size
        self.optim_fn = optim.Adam
        self.optim_args = {"lr": self.lr, "betas": [0.9, 0.999]}
        self.optim = self.optim_fn(self.optim_args)

        if verbose:
            self.logger.info("Optimizer - {}".format(self.optim_fn.__name__))
            self.logger.info("Learning rate - {}".format(self.lr))
            self.logger.info("Batch size - {}".format(self.batch_size))
            self.logger.info("{}".format("jit" if jit else "nojit"))

        if self.run_path is not None:
            try:
                self.load_checkpoint()
            except FileNotFoundError:
                pyro.clear_param_store()
                self.iter = 1
                self._rolling = {p: deque([], maxlen=100) for p in self.conv_params}
        else:
            pyro.clear_param_store()
            self.iter = 1
            self._rolling = {p: deque([], maxlen=100) for p in self.conv_params}

        self.elbo = self.TraceELBO(jit)
        self.svi = infer.SVI(self.model, self.guide, self.optim, loss=self.elbo)
        # pyro.enable_validation()
        self._stop = False
        for i in tqdm(range(num_iter)):
            self.iter_loss = self.svi.step()
            if not self.iter % 100 and self.iter != 0:
                self.save_checkpoint()
                if self._stop:
                    self.logger.info("Step #{} model converged.".format(self.iter))
                    break
            self.iter += 1

    def _max_batch_size(self):
        k = 0
        batch_size = 0
        while batch_size + 2 ** k < self.data.ontarget.N:
            try:
                self.run(1, lr=self.lr, batch_size=batch_size + 2 ** k, verbose=False)
            except RuntimeError as error:
                assert error.args[0].startswith("CUDA")
                if k == 0:
                    break
                else:
                    batch_size += 2 ** (k - 1)
                    k = 0
            else:
                k += 1
        else:
            batch_size += 2 ** (k - 1)

        return batch_size

    def save_checkpoint(self):
        # save only if no NaN values
        for k, v in pyro.get_param_store().items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                raise ValueError(
                    "Step #{}. Detected NaN values in {}".format(self.iter, k)
                )

        # update convergence criteria parameters
        for name in self.conv_params:
            if name == "-ELBO":
                self._rolling["-ELBO"].append(self.iter_loss)
            else:
                self._rolling[name].append(pyro.param(name).item())

        # check convergence status
        if len(self._rolling["-ELBO"]) == self._rolling["-ELBO"].maxlen:
            crit = all(
                torch.tensor(self._rolling[p]).std()
                / torch.tensor(self._rolling[p])[-50:].std()
                < 1.05
                for p in self.conv_params
            )
            if crit:
                self._stop = True

        # save the model state
        torch.save(
            {
                "iter": self.iter,
                "params": pyro.get_param_store().get_state(),
                "optimizer": self.optim.get_state(),
                "rolling": self._rolling,
                "convergence_status": self._stop,
            },
            self.run_path / "model",
        )

        # save global paramters for tensorboard
        self.writer.add_scalar("-ELBO", self.iter_loss, self.iter)
        for name, val in pyro.get_param_store().items():
            if val.dim() == 0:
                self.writer.add_scalar(name, val.item(), self.iter)
            elif val.dim() == 1 and len(val) <= self.S + 1:
                scalars = {str(i): v.item() for i, v in enumerate(val)}
                self.writer.add_scalars(name, scalars, self.iter)

        if self.data.ontarget.labels is not None and self.z_marginal is not None:
            pred_labels = self.z_map.cpu().numpy().ravel()
            true_labels = self.data.ontarget.labels["z"].ravel()

            metrics = {}
            with np.errstate(divide="ignore", invalid="ignore"):
                metrics["MCC"] = matthews_corrcoef(true_labels, pred_labels)
            metrics["Recall"] = recall_score(true_labels, pred_labels, zero_division=0)
            metrics["Precision"] = precision_score(
                true_labels, pred_labels, zero_division=0
            )

            neg, pos = {}, {}
            neg["TN"], neg["FP"], pos["FN"], pos["TP"] = confusion_matrix(
                true_labels, pred_labels, labels=(0, 1)
            ).ravel()

            self.writer.add_scalars("ACCURACY", metrics, self.iter)
            self.writer.add_scalars("NEGATIVES", neg, self.iter)
            self.writer.add_scalars("POSITIVES", pos, self.iter)

        self.logger.debug("Step #{}.".format(self.iter))

    def load_checkpoint(self, path=None, param_only=False):
        path = Path(path) if path else self.run_path
        pyro.clear_param_store()
        checkpoint = torch.load(path / "model", map_location=self.device)
        pyro.get_param_store().set_state(checkpoint["params"])
        if not param_only:
            self._stop = checkpoint["convergence_status"]
            self._rolling = checkpoint["rolling"]
            self.iter = checkpoint["iter"]
            self.optim.set_state(checkpoint["optimizer"])
            self.logger.info(
                "Step #{}. Loaded model params and optimizer state from {}".format(
                    self.iter, path
                )
            )

    def compute_stats(self, CI=0.95, save_matlab=False):
        save_stats(self, self.path, CI=CI, save_matlab=save_matlab)
        if self.path is not None:
            self.logger.info(f"Parameters were saved in {self.path / 'params.tpqr'}")
            if save_matlab:
                self.logger.info(f"Parameters were saved in {self.path / 'params.mat'}")

    def snr(self):
        r"""
        Calculate the signal-to-noise ratio.

        Total signal:

            :math:`\mu_{knf} =  \sum_{ij} I_{nfij}
            \mathcal{N}(i, j \mid x_{knf}, y_{knf}, w_{knf})`

        Noise:

            :math:`\sigma^2_{knf} = \sigma^2_{\text{offset}}
            + \mu_{knf} \text{gain}`

        Signal-to-noise ratio:

            :math:`\text{SNR}_{knf} =
            \dfrac{\mu_{knf} - b_{nf} - \mu_{\text{offset}}}{\sigma_{knf}}
            \text{ for } \theta_{nf} = k`
        """
        weights = self.gaussian(
            torch.ones(1),
            self.params["d/width"]["Mean"],
            self.params["d/x"]["Mean"],
            self.params["d/y"]["Mean"],
            self.data.ontarget.xy.to(self.dtype),
        )
        signal = (
            (
                self.data.ontarget.images
                - self.params["d/background"]["Mean"][..., None, None]
                - self.data.offset.mean
            )
            * weights
        ).sum(dim=(-2, -1))
        noise = (
            self.data.offset.var
            + self.params["d/background"]["Mean"] * self.params["gain"]["Mean"]
        ).sqrt()
        result = signal / noise
        mask = self.z_probs > 0.5
        return result[mask]
