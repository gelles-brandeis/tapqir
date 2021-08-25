# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
import os
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyroapi import infer, optim, pyro
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from torch.distributions.utils import lazy_property
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from tapqir import __version__ as tapqir_version
from tapqir.distributions.kspotgammanoise import _gaussian_spots
from tapqir.utils.dataset import load
from tapqir.utils.stats import save_stats

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    fmt="%(levelname)s - %(message)s",
)
ch.setFormatter(formatter)
logger.addHandler(ch)


class Model:
    r"""
    Base class for tapqir models.

    Derived models must implement the methods

    * :meth:`model`
    * :meth:`guide`
    """

    def __init__(self, S=1, K=2, device="cpu", dtype="double", verbose=True):
        self._S = S
        self._K = K
        self.batch_size = None
        # for plotting
        self.n = None
        self.data_path = None
        self.path = None
        self.run_path = None
        # logger
        self.verbose = verbose
        self.logger = logger
        # set device & dtype
        self.to(device, dtype)
        if self.verbose:
            self.logger.info("Tapqir version - {}".format(tapqir_version))
            self.logger.info("Model - {}".format(self.name))

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
        if self.verbose:
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

    def load(self, path, data_only=True):
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
        if self.verbose:
            self.logger.info(f"Loaded data from {self.path / 'data.tpqr'}")

        # load fit results
        if not data_only:
            self.params = torch.load(self.path / f"{self.name}-params.tpqr")
            self.statistics = pd.read_csv(self.path / "statistics.csv", index_col=0)
            if self.verbose:
                self.logger.info(f"Loaded parameters from {self.name}-params.tpqr")
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

    def init(self, lr=0.005, batch_size=0, jit=False):
        self.optim_fn = optim.Adam
        self.optim_args = {"lr": lr, "betas": [0.9, 0.999]}
        self.optim = self.optim_fn(self.optim_args)

        try:
            self.load_checkpoint()
        except (FileNotFoundError, TypeError):
            pyro.clear_param_store()
            self.iter = 0
            self._rolling = {p: deque([], maxlen=100) for p in self.conv_params}
            self.init_parameters()

        self.elbo = self.TraceELBO(jit)
        self.svi = infer.SVI(self.model, self.guide, self.optim, loss=self.elbo)

        # find max possible batch_size
        if batch_size == 0:
            batch_size = self._max_batch_size()
            if self.verbose:
                self.logger.info(
                    "Optimal batch size determined - {}".format(batch_size)
                )
        self.batch_size = batch_size

        if self.verbose:
            self.logger.info("Optimizer - {}".format(self.optim_fn.__name__))
            self.logger.info("Learning rate - {}".format(lr))
            self.logger.info("Batch size - {}".format(self.batch_size))
            self.logger.info("{}".format("jit" if jit else "nojit"))

    def run(self, num_iter=0, nrange=trange):
        slurm = os.environ.get("SLURM_JOB_ID")
        if slurm is not None:
            nrange = range
        # pyro.enable_validation()
        self._stop = False
        use_crit = False
        if not num_iter:
            use_crit = True
            num_iter = 100000
        for i in nrange(num_iter):
            self.iter_loss = self.svi.step()
            if not self.iter % 100:
                self.save_checkpoint()
                if use_crit and self._stop:
                    self.logger.info("Step #{} model converged.".format(self.iter))
                    break
            self.iter += 1

    def _max_batch_size(self):
        k = 0
        batch_size = 0
        while batch_size + 2 ** k < self.data.ontarget.N:
            try:
                torch.cuda.empty_cache()
                self.batch_size = batch_size + 2 ** k
                self.run(1, nrange=range)
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

        return max(1, batch_size // 4)

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

        if self.pspecific is not None and self.data.ontarget.labels is not None:
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

    def load_checkpoint(self, path=None, param_only=False, warnings=False):
        device = self.device
        path = Path(path) if path else self.run_path
        pyro.clear_param_store()
        checkpoint = torch.load(path / "model", map_location=device)
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
        if warnings and not checkpoint["convergence_status"]:
            self.logger.warning(f"Model at {path} has not been fully trained")

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
        weights = _gaussian_spots(
            torch.ones(1),
            self.params["d/width"]["Mean"],
            self.params["d/x"]["Mean"],
            self.params["d/y"]["Mean"],
            self.data.ontarget.xy.to(self.dtype),
            self.data.ontarget.P,
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
