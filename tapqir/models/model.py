# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

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
from tapqir.utils.dataset import load
from tapqir.utils.stats import save_stats

logger = logging.getLogger(__name__)


class Model:
    r"""
    Base class for tapqir models.

    Derived models must implement the methods

    * :meth:`model`
    * :meth:`guide`
    """

    def __init__(self, S=1, K=2, channels=(0,), device="cpu", dtype="double"):
        self._S = S
        self._K = K
        self.channels = channels
        self.nbatch_size = None
        self.fbatch_size = None
        # for plotting
        self.n = None
        self.f = None
        self.data_path = None
        self.path = None
        self.run_path = None
        logger.info("Tapqir version - {}".format(tapqir_version))
        logger.info("Model - {}".format(self.name))
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
        logger.info("Device - {}".format(self.device))
        logger.info("Floating precision - {}".format(self.dtype))

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
        self.run_path = self.path / ".tapqir"
        self.writer = SummaryWriter(log_dir=self.run_path / "logs" / self.full_name)

        # load data
        self.data = load(self.path, self.device)
        logger.info(f"Loaded data from {self.path / 'data.tpqr'}")

        # load fit results
        if not data_only:
            self.params = torch.load(self.path / f"{self.full_name}-params.tpqr")
            self.summary = pd.read_csv(
                self.path / f"{self.full_name}-summary.csv", index_col=0
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

    def init(self, lr=0.005, nbatch_size=5, fbatch_size=1000, jit=False):
        self.optim_fn = optim.Adam
        self.optim_args = {"lr": lr, "betas": [0.9, 0.999]}
        self.optim = self.optim_fn(self.optim_args)

        try:
            self.load_checkpoint()
        except (FileNotFoundError, TypeError):
            pyro.clear_param_store()
            self.epoch = 0
            self._rolling = {p: deque([], maxlen=100) for p in self.conv_params}
            self.init_parameters()

        self.elbo = self.TraceELBO(jit)
        self.svi = infer.SVI(self.model, self.guide, self.optim, loss=self.elbo)

        # find max possible batch_size
        if nbatch_size == 0:
            nbatch_size = self._max_batch_size()
            logger.info("Optimal batch size determined - {}".format(nbatch_size))
        self.nbatch_size = nbatch_size
        self.fbatch_size = fbatch_size

        logger.info("Optimizer - {}".format(self.optim_fn.__name__))
        logger.info("Learning rate - {}".format(lr))
        logger.info("AOI batch size - {}".format(self.nbatch_size))
        logger.info("Frame batch size - {}".format(self.fbatch_size))

    def run(self, num_epochs=0, nrange=trange):
        slurm = os.environ.get("SLURM_JOB_ID")
        if slurm is not None:
            nrange = range
        # pyro.enable_validation()
        self._stop = False
        use_crit = False
        if not num_epochs:
            use_crit = True
            num_epochs = 1000
        for i in nrange(num_epochs):
            losses = []
            for ndx in torch.split(torch.randperm(self.data.N), self.nbatch_size):
                for fdx in torch.split(torch.randperm(self.data.F), self.fbatch_size):
                    self.n = ndx
                    self.f = fdx
                    losses.append(self.svi.step())
            self.epoch_loss = sum(losses) / len(losses)
            self.save_checkpoint()
            if use_crit and self._stop:
                logger.info("Step #{} model converged.".format(self.epoch))
                break
            self.epoch += 1

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
                    "Step #{}. Detected NaN values in {}".format(self.epoch, k)
                )

        # update convergence criteria parameters
        for name in self.conv_params:
            if name == "-ELBO":
                self._rolling["-ELBO"].append(self.epoch_loss)
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
                "epoch": self.epoch,
                "params": pyro.get_param_store().get_state(),
                "optimizer": self.optim.get_state(),
                "rolling": self._rolling,
                "convergence_status": self._stop,
            },
            self.run_path / f"{self.full_name}-model.tpqr",
        )

        # save global paramters for tensorboard
        self.writer.add_scalar("-ELBO", self.epoch_loss, self.epoch)
        for name, val in pyro.get_param_store().items():
            if val.dim() == 0:
                self.writer.add_scalar(name, val.item(), self.epoch)
            elif val.dim() == 1 and len(val) <= self.S + 1:
                scalars = {str(i): v.item() for i, v in enumerate(val)}
                self.writer.add_scalars(name, scalars, self.epoch)

        if self.pspecific is not None and self.data.labels is not None:
            pred_labels = self.z_map[self.data.is_ontarget].cpu().numpy().ravel()
            true_labels = self.data.labels["z"].ravel()

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

            self.writer.add_scalars("ACCURACY", metrics, self.epoch)
            self.writer.add_scalars("NEGATIVES", neg, self.epoch)
            self.writer.add_scalars("POSITIVES", pos, self.epoch)

        logger.debug("Step #{}.".format(self.epoch))

    def load_checkpoint(self, path=None, param_only=False, warnings=False):
        device = self.device
        path = Path(path) if path else self.run_path
        pyro.clear_param_store()
        checkpoint = torch.load(
            path / f"{self.full_name}-model.tpqr", map_location=device
        )
        pyro.get_param_store().set_state(checkpoint["params"])
        if not param_only:
            self._stop = checkpoint["convergence_status"]
            self._rolling = checkpoint["rolling"]
            self.epoch = checkpoint["epoch"]
            self.optim.set_state(checkpoint["optimizer"])
            logger.info(
                "Step #{}. Loaded model params and optimizer state from {}".format(
                    self.epoch, path
                )
            )
        if warnings and not checkpoint["convergence_status"]:
            logger.warning(f"Model at {path} has not been fully trained")

    def compute_stats(self, CI=0.95, save_matlab=False):
        save_stats(self, self.path, CI=CI, save_matlab=save_matlab)
        if self.path is not None:
            logger.info(f"Parameters were saved in {self.path / 'params.tpqr'}")
            if save_matlab:
                logger.info(f"Parameters were saved in {self.path / 'params.mat'}")
