# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import logging
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Union

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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tapqir import __version__ as tapqir_version
from tapqir.exceptions import CudaOutOfMemoryError, TapqirFileNotFoundError
from tapqir.utils.dataset import load
from tapqir.utils.stats import save_stats

logger = logging.getLogger(__name__)


class Model:
    r"""
    Base class for tapqir models.

    Derived models must implement the methods

    * :meth:`model`
    * :meth:`guide`
    * :meth:`init_parameters`
    * :meth:`TraceELBO`

    :param S: Number of distinct molecular states for the binder molecules.
    :param K: Maximum number of spots that can be present in a single image.
    :param Q: Number of fluorescent dyes.
    :param device: Computation device (cpu or gpu).
    :param dtype: Floating point precision.
    :param priors: Dictionary of parameters of prior distributions.
    """

    def __init__(
        self,
        S: int = 1,
        K: int = 2,
        Q: int = None,
        device: str = "cpu",
        dtype: str = "double",
        priors: dict = None,
    ):
        self.S = S
        self.K = K
        self._Q = Q
        self.nbatch_size = None
        self.fbatch_size = None
        # priors settings
        self.priors = priors
        # for plotting
        self.n = None
        self.f = None
        self.data_path = None
        self.path = None
        self.run_path = None
        # set device & dtype
        self.to(device, dtype)

    def to(self, device: str, dtype: str = "double") -> None:
        """
        Change tensor device and dtype.

        :param device: Computation device, either "gpu" or "cpu".
        :param dtype: Floating point precision, either "double" or "float".
        """
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

    @property
    def Q(self):
        return self._Q or self.data.C

    def load(self, path: Union[str, Path], data_only: bool = True) -> None:
        """
        Load data and optionally parameters from a specified path

        :param path: Path to Tapqir analysis folder.
        :param data_only: Load only data or both data and parameters.
        """
        # set path
        self.path = Path(path)
        self.run_path = self.path / ".tapqir"

        # load data
        self.data = load(self.path, self.device)
        logger.debug(f"Loaded data from {self.path / 'data.tpqr'}")

        # load fit results
        if not data_only:
            try:
                self.params = torch.load(self.path / f"{self.name}-params.tpqr")
            except FileNotFoundError:
                raise TapqirFileNotFoundError(
                    "parameter", self.path / f"{self.name}-params.tpqr"
                )
            try:
                self.summary = pd.read_csv(
                    self.path / f"{self.name}-summary.csv", index_col=0
                )
            except FileNotFoundError:
                raise TapqirFileNotFoundError(
                    "summary", self.path / f"{self.name}-summary.csv"
                )

    def model(self):
        """
        Generative Model
        """
        raise NotImplementedError

    def guide(self):
        """
        Variational Distribution
        """
        raise NotImplementedError

    def TraceELBO(self, jit):
        """
        A trace implementation of ELBO-based SVI.
        """
        raise NotImplementedError

    def init_parameters(self):
        """
        Initialize variational parameters.
        """
        raise NotImplementedError

    def init(
        self,
        lr: float = 0.005,
        nbatch_size: int = 5,
        fbatch_size: int = 512,
        jit: bool = False,
    ) -> None:
        """
        Initialize SVI object.

        :param lr: Learning rate.
        :param nbatch_size: AOI batch size.
        :param fbatch_size: Frame batch size.
        :param jit: Use JIT compiler.
        """
        self.lr = lr
        self.optim_fn = optim.Adam
        self.optim_args = {"lr": lr, "betas": [0.9, 0.999]}
        self.optim = self.optim_fn(self.optim_args)

        try:
            self.load_checkpoint()
        except TapqirFileNotFoundError:
            pyro.clear_param_store()
            self.iter = 0
            self.converged = False
            self._rolling = defaultdict(lambda: deque([], maxlen=100))
            self.init_parameters()

        self.elbo = self.TraceELBO(jit)
        self.svi = infer.SVI(self.model, self.guide, self.optim, loss=self.elbo)

        self.nbatch_size = min(nbatch_size, self.data.Nt)
        self.fbatch_size = min(fbatch_size, self.data.F)

    def run(self, num_iter: int = 0, progress_bar=tqdm) -> None:
        """
        Run inference procedure for a specified number of iterations.
        If num_iter equals zero then run till model converges.

        :param num_iter: Number of iterations.
        """
        use_crit = False
        if not num_iter:
            use_crit = True
            num_iter = 100000

        logger.debug("Tapqir version - {}".format(tapqir_version))
        logger.debug("Model - {}".format(self.name))
        logger.debug("Device - {}".format(self.device))
        logger.debug("Floating precision - {}".format(self.dtype))
        logger.debug("Optimizer - {}".format(self.optim_fn.__name__))
        logger.debug("Learning rate - {}".format(self.lr))
        logger.debug("AOI batch size - {}".format(self.nbatch_size))
        logger.debug("Frame batch size - {}".format(self.fbatch_size))

        with SummaryWriter(log_dir=self.run_path / "logs" / self.name) as writer:
            for i in progress_bar(range(num_iter)):
                try:
                    self.iter_loss = self.svi.step()
                    # save a checkpoint every 200 iterations
                    if not self.iter % 200:
                        self.save_checkpoint(writer)
                        if use_crit and self.converged:
                            logger.info(f"Iteration #{self.iter} model converged.")
                            break
                    self.iter += 1
                except ValueError:
                    # load last checkpoint
                    self.init(
                        lr=self.lr,
                        nbatch_size=self.nbatch_size,
                        fbatch_size=self.fbatch_size,
                    )
                    # change rng seed
                    new_seed = random.randint(0, 100)
                    pyro.set_rng_seed(new_seed)
                    logger.warning(
                        f"Iteration #{self.iter} restarting with a new seed: {new_seed}."
                    )
                except RuntimeError as err:
                    assert err.args[0].startswith("CUDA out of memory")
                    raise CudaOutOfMemoryError()
            else:
                logger.warning(f"Iteration #{self.iter} model has not converged.")

    def save_checkpoint(self, writer: SummaryWriter = None):
        """
        Save checkpoint.

        :param writer: SummaryWriter object.
        """
        # save only if no NaN values
        for k, v in pyro.get_param_store().items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                raise ValueError(
                    "Iteration #{}. Detected NaN values in {}".format(self.iter, k)
                )

        # update convergence criteria parameters
        for name in self.conv_params:
            if name == "-ELBO":
                self._rolling["-ELBO"].append(self.iter_loss)
            elif pyro.param(name).ndim == 1:
                for i in range(len(pyro.param(name))):
                    self._rolling[f"{name}_{i}"].append(pyro.param(name)[i].item())
            else:
                self._rolling[name].append(pyro.param(name).item())

        # check convergence status
        self.converged = False
        if len(self._rolling["-ELBO"]) == self._rolling["-ELBO"].maxlen:
            crit = all(
                torch.tensor(value).std() / torch.tensor(value)[-50:].std() < 1.05
                for value in self._rolling.values()
            )
            if crit:
                self.converged = True

        # save the model state
        torch.save(
            {
                "iter": self.iter,
                "params": pyro.get_param_store().get_state(),
                "optimizer": self.optim.get_state(),
                "rolling": dict(self._rolling),
                "convergence_status": self.converged,
            },
            self.run_path / f"{self.name}-model.tpqr",
        )

        # save global parameters for tensorboard
        writer.add_scalar("-ELBO", self.iter_loss, self.iter)
        for name, val in pyro.get_param_store().items():
            if val.dim() == 0:
                writer.add_scalar(name, val.item(), self.iter)
            elif val.dim() == 1 and len(val) <= self.S + 1:
                scalars = {str(i): v.item() for i, v in enumerate(val)}
                writer.add_scalars(name, scalars, self.iter)

        if False and self.data.labels is not None:
            pred_labels = (
                self.pspecific_map[self.data.is_ontarget].cpu().numpy().ravel()
            )
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

            writer.add_scalars("ACCURACY", metrics, self.iter)
            writer.add_scalars("NEGATIVES", neg, self.iter)
            writer.add_scalars("POSITIVES", pos, self.iter)

        logger.debug(f"Iteration #{self.iter}: Successful.")

    def load_checkpoint(
        self,
        path: Union[str, Path] = None,
        param_only: bool = False,
        warnings: bool = False,
    ):
        """
        Load checkpoint.

        :param path: Path to model checkpoint.
        :param param_only: Load only parameters.
        :param warnings: Give warnings if loaded model has not been fully trained.
        """
        device = self.device
        path = Path(path) if path else self.run_path
        model_path = path / f"{self.name}-model.tpqr"
        try:
            checkpoint = torch.load(model_path, map_location=device)
        except FileNotFoundError:
            raise TapqirFileNotFoundError("model", model_path)

        pyro.clear_param_store()
        pyro.get_param_store().set_state(checkpoint["params"])
        if not param_only:
            self.converged = checkpoint["convergence_status"]
            self._rolling = checkpoint["rolling"]
            self.iter = checkpoint["iter"]
            self.optim.set_state(checkpoint["optimizer"])
            logger.info(
                f"Iteration #{self.iter}. Loaded a model checkpoint from {model_path}"
            )
        if warnings and not checkpoint["convergence_status"]:
            logger.warning(f"Model at {path} has not been fully trained")

    def compute_stats(self, CI: float = 0.95, save_matlab: bool = False):
        """
        Compute credible regions (CI) and other stats.

        :param CI: credible region.
        :param save_matlab: Save output in Matlab format as well.
        """
        try:
            save_stats(self, self.path, CI=CI, save_matlab=save_matlab)
        except RuntimeError as err:
            assert err.args[0].startswith("CUDA out of memory")
            raise CudaOutOfMemoryError()
        logger.debug("Computing stats: Successful.")
