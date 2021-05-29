# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
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


class GaussianSpot:
    r"""
    Calculates ideal shape of the 2D-Gaussian spot given spot parameters,
    target positions, and drift list.

        :math:`\dfrac{h_{knf}}{2 \pi w^2_{nfk}}
        \exp{\left ( -\dfrac{(i-x_{nfk})^2 + (j-y_{nfk})^2}{2w^2_{nfk}} \right)}`

    :param target: Target positions.
    :param drift: Frame drift list.
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
        :param ndx: AoI indices.
        :param fdx: Frame indices.
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

    **Implementing New Models**:

    Derived models must implement the methods
    :meth:`model`
    :meth:`guide`
    """

    def __init__(self, S=1, K=2, device="cpu", dtype="double"):
        super().__init__()
        self._S = S
        self._K = K
        self.batch_size = None
        self.status = None
        # for plotting
        self.n = None
        self.data_path = None
        # set device & dtype
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

        # status
        for line in reversed(list(open(self.run_path / "run.log"))):
            if "model converged" in line:
                self.status = "Trained"
                break

        # load data
        self.data = load(self.path, self.device)
        self.gaussian = GaussianSpot(self.data.ontarget.P)

        # load fit results
        if (self.path / "params.tpqr").is_file():
            self.params = torch.load(self.path / "params.tpqr")
        if (self.path / "theta_samples.pt").is_file():
            self.theta_samples = torch.load(self.path / "theta_samples.tpqr")

    def settings(self, lr=0.005, batch_size=0, jit=False):
        # K - max number of spots
        self.lr = lr
        # find max possible batch_size
        if batch_size == 0:
            batch_size = self._max_batch_size()
        self.batch_size = batch_size
        self.jit = jit

        self.optim_fn = optim.Adam
        self.optim_args = {"lr": self.lr, "betas": [0.9, 0.999]}
        self.optim = self.optim_fn(self.optim_args)
        self.log()

        if hasattr(self, "run_path"):
            try:
                self.load_checkpoint()
            except FileNotFoundError:
                pyro.clear_param_store()
                self.iter = 1
                self._rolling = None
        else:
            pyro.clear_param_store()
            self.iter = 1
            self._rolling = None

        if self.name == "hmm":
            self.elbo = (
                infer.JitTraceMarkovEnum_ELBO if jit else infer.TraceMarkovEnum_ELBO
            )(max_plate_nesting=3, ignore_jit_warnings=True)
        else:
            self.elbo = (infer.JitTraceEnum_ELBO if jit else infer.TraceEnum_ELBO)(
                max_plate_nesting=2, ignore_jit_warnings=True
            )
        self.svi = infer.SVI(self.model, self.guide, self.optim, loss=self.elbo)

    def model(self):
        r"""
        Generative Model
        """
        raise NotImplementedError

    def guide(self):
        r"""
        Variational Guide
        """
        raise NotImplementedError

    def run(self, num_iter):
        # pyro.enable_validation()
        self._stop = False
        for i in tqdm(range(num_iter)):
            self.iter_loss = self.svi.step()
            if not self.iter % 100 and self.iter != 0:
                self.save_checkpoint()
                if self._stop:
                    self.logger.debug("Step #{} model converged.".format(self.iter))
                    break
            self.iter += 1

    def _max_batch_size(self):
        k = 0
        batch_size = 0
        while batch_size + 2 ** k < self.data.ontarget.N:
            self.settings(self.lr, batch_size + 2 ** k)
            try:
                self.run(1)
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

    def log(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S %p",
        )
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if hasattr(self, "run_path"):
            self.writer = SummaryWriter(log_dir=self.run_path / "tb")
            fh = logging.FileHandler(self.run_path / "run.log")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        self.logger.info("Tapqir version - {}".format(tapqir_version))
        self.logger.info("Model - {}".format(self.name))
        self.logger.info("Optimizer - {}".format(self.optim_fn.__name__))
        self.logger.info("Learning rate - {}".format(self.lr))
        self.logger.info("Batch size - {}".format(self.batch_size))
        self.logger.info("Floating precision - {}".format(self.dtype))
        self.logger.info("{}".format("jit" if self.jit else "nojit"))

    def save_checkpoint(self):
        # save only if no NaN values
        for k, v in pyro.get_param_store().items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                raise ValueError(
                    "Step #{}. Detected NaN values in {}".format(self.iter, k)
                )

        # save parameters and optimizer state
        pyro.get_param_store().save(self.run_path / "params")
        self.optim.save(self.run_path / "optimizer")

        # save global paramters in csv file and for tensorboard
        global_params = pd.Series(dtype=float, name=self.iter)

        self.writer.add_scalar("-ELBO", self.iter_loss, self.iter)
        global_params["-ELBO"] = self.iter_loss
        for name, val in pyro.get_param_store().items():
            if val.dim() == 0:
                self.writer.add_scalar(name, val.item(), self.iter)
                global_params[name] = val.item()
            elif val.dim() == 1 and len(val) <= self.S + 1:
                scalars = {str(i): v.item() for i, v in enumerate(val)}
                self.writer.add_scalars(name, scalars, self.iter)
                for key, value in scalars.items():
                    global_params["{}_{}".format(name, key)] = value

        if self.data.ontarget.labels is not None and self.name == "hmm":
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
            for key, value in {**metrics, **pos, **neg}.items():
                global_params[key] = value

        if self._rolling is None:
            self._rolling = global_params.to_frame().T

        if global_params.name not in self._rolling.index:
            self._rolling = self._rolling.append(global_params)
        if len(self._rolling) > 100:
            self._rolling = self._rolling.drop(self._rolling.index[0])
            conv_params = ["-ELBO", "proximity_loc", "gain_loc", "lamda_loc"]
            crit = all(
                self._rolling[p].std() / self._rolling[p].iloc[-50:].std() < 1.05
                for p in conv_params
            )
            if crit:
                self._stop = True

        global_params.to_csv(self.run_path / "global_params.csv")
        self._rolling.to_csv(self.run_path / "rolling_params.csv")
        self.logger.debug("Step #{}.".format(self.iter))

    def load_checkpoint(self, path=None, param_only=False):
        path = Path(path) if path else self.run_path
        pyro.clear_param_store()
        pyro.get_param_store().load(path / "params", map_location=self.device)
        if not param_only:
            global_params = pd.read_csv(
                path / "global_params.csv", squeeze=True, index_col=0
            )
            self._rolling = pd.read_csv(path / "rolling_params.csv", index_col=0)
            self.iter = int(global_params.name)
            self.optim.load(path / "optimizer")
            self.logger.info(
                "Step #{}. Loaded model params and optimizer state from {}".format(
                    self.iter, path
                )
            )

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
        with torch.no_grad():
            weights = self.gaussian(
                torch.ones(1),
                pyro.param("d/w_mean"),
                pyro.param("d/x_mean"),
                pyro.param("d/y_mean"),
                self.data.ontarget.xy,
            )
            signal = (
                (
                    self.data.ontarget.data
                    - pyro.param("d/b_loc")[..., None, None]
                    - self.data.offset.mean
                )
                * weights
            ).sum(dim=(-2, -1))
            noise = (
                self.data.offset.var + pyro.param("d/b_loc") * pyro.param("gain_loc")
            ).sqrt()
            result = signal / noise
            mask = self.z_probs > 0.5
            return result[mask]
