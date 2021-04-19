import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pyro.ops.indexing import Vindex
from pyroapi import distributions as dist
from pyroapi import infer, optim, pyro
from scipy.io import savemat
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
from tapqir.utils.dataset import load_data


class GaussianSpot:
    r"""
    Calculates ideal shape of the 2D-Gaussian spot given spot parameters,
    target positions, and drift list.

        :math:`\dfrac{h_{knf}}{2 \pi w^2_{nfk}}
        \exp{\left ( -\dfrac{(i-x_{nfk})^2 + (j-y_{nfk})^2}{2w^2_{nfk}} \right)}`

    :param target: Target positions.
    :param drift: Frame drift list.
    """

    def __init__(self, target_locs, D):
        super().__init__()
        # create meshgrid of DxD pixel positions
        D_range = torch.arange(D, dtype=torch.float)
        i_pixel, j_pixel = torch.meshgrid(D_range, D_range)
        self.ij_pixel = torch.stack((i_pixel, j_pixel), dim=-1)
        self.target_locs = target_locs

    # Ideal 2D gaussian spots
    def __call__(self, height, width, x, y, ndx, fdx=None):
        r"""
        :param height: integrated spot intensity.
        :param width: width of the 2D-Gaussian spot.
        :param x: relative :math:`x`-axis position relative to the target.
        :param y: relative :math:`y`-axis position relative to the target.
        :param ndx: AoI indices.
        :param fdx: Frame indices.
        :return: Ideal shape 2D-Gaussian spot.
        """

        if fdx is not None:
            spot_locs = Vindex(self.target_locs)[ndx, fdx] + torch.stack((x, y), -1)
        else:
            spot_locs = Vindex(self.target_locs)[ndx] + torch.stack((x, y), -1)
        rv = dist.MultivariateNormal(
            spot_locs[..., None, None, :],
            scale_tril=torch.eye(2) * width[..., None, None, None, None],
        )
        gaussian_spot = torch.exp(rv.log_prob(self.ij_pixel))  # N,F,D,D
        return height[..., None, None] * gaussian_spot


class Model(nn.Module):
    r"""
    Base class for tapqir models.

    **Implementing New Models**:

    Derived models must implement the methods
    :meth:`model`
    :meth:`guide`
    """

    def __init__(self, S, K=2):
        super().__init__()
        self._S = S
        self._K = K
        self.batch_size = None
        # for plotting
        self.n = None
        self.data_path = None

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

    def load(self, path, control, device):
        # set path
        self.data_path = Path(path)

        # set device
        self.device = torch.device(device)

        # load test data
        self.data = load_data(self.data_path, dtype="test", device=self.device)
        self.data_loc = GaussianSpot(self.data.target_locs, self.data.D)

        # load control data
        if control:
            self.control = load_data(
                self.data_path, dtype="control", device=self.device
            )
            self.control_loc = GaussianSpot(self.control.target_locs, self.control.D)
        else:
            self.control = control

    def settings(self, lr, batch_size, jit=False):
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

        if self.data_path is not None:
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

    def run(self, num_iter, infer):
        # pyro.enable_validation()
        self._stop = False
        self.classify = False
        for i in tqdm(range(num_iter)):
            self.iter_loss = self.svi.step()
            if not self.iter % 100 and self.iter != 0:
                self.save_checkpoint()
                if self._stop:
                    self.logger.info("Step #{} converged.".format(self.iter))
                    break
            self.iter += 1

        self.classify = True
        for i in tqdm(range(infer)):
            self.iter_loss = self.svi.step()
            if not self.iter % 100:
                self.save_checkpoint()
            self.iter += 1

    def _max_batch_size(self):
        k = 0
        batch_size = 0
        while batch_size + 2 ** k < self.data.N:
            self.settings(self.lr, batch_size + 2 ** k)
            try:
                self.run(1, 1)
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

        if self.data_path is not None:
            self.path = (
                self.data_path
                / "runs"
                / f"{self.name}"
                / tapqir_version.split("+")[0]
                / f"S{self.S}"
                / f"{'control' if self.control else 'nocontrol'}"
                / f"lr{self.lr}"
                / f"bs{self.batch_size}"
            )
            self.writer = SummaryWriter(log_dir=self.path / "scalar")

            fh = logging.FileHandler(self.path / "run.log")
            fh.setLevel(logging.DEBUG)
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
        for k, v in pyro.get_param_store().items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                raise ValueError(
                    "Step #{}. Detected NaN values in {}".format(self.iter, k)
                )

        # save parameters and optimizer state
        pyro.get_param_store().save(self.path / "params")
        self.optim.save(self.path / "optimizer")

        # save parameters in matlab format
        keys = ["h_loc", "w_mean", "x_mean", "y_mean", "b_loc"]
        matlab = {k: pyro.param(f"d/{k}").data.cpu().numpy() for k in keys}
        matlab[
            "parametersDescription"
        ] = "Parameters for N x F x K spots. \
            N - target sites, F - frames, K - max number of spots in the image. \
            h_loc - mean intensity, w_mean - mean spot width, \
            x_mean - x position, y_mean - y position, \
            b_loc - background intensity."
        if self.classify:
            matlab["z_probs"] = self.z_probs.cpu().numpy()
            matlab["j_probs"] = self.j_probs.cpu().numpy()
            matlab["m_probs"] = self.m_probs.cpu().numpy()
            matlab["z_marginal"] = self.z_marginal.cpu().numpy()
            matlab[
                "probabilitiesDescription"
            ] = "Probabilities for N x F x K spots. \
                z_probs - on-target spot probability, \
                j_probs - off-target spot probability, \
                m_probs - spot probability (on-target + off-target), \
                z_marginal - total on-target spot probability (sum of z_probs)."
        matlab["aoilist"] = self.data.target.index.values
        matlab["aoilistDescription"] = "aoi numbering from aoiinfo"
        matlab["framelist"] = self.data.drift.index.values
        matlab["framelistDescription"] = "frame numbering from driftlist"
        savemat(self.path / "parameters.mat", matlab)

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

        if self.classify and self.data.labels is not None:
            mask = self.data.labels["z"] < 2
            pred_labels = self.z_map.cpu().numpy()[mask]
            true_labels = self.data.labels["z"][mask]

            metrics = {}
            with np.errstate(divide="ignore", invalid="ignore"):
                metrics["MCC"] = matthews_corrcoef(true_labels, pred_labels)
            metrics["Recall"] = recall_score(true_labels, pred_labels, zero_division=0)
            metrics["Precision"] = precision_score(
                true_labels, pred_labels, zero_division=0
            )

            neg, pos = {}, {}
            neg["TN"], neg["FP"], pos["FN"], pos["TP"] = confusion_matrix(
                true_labels, pred_labels
            ).ravel()

            self.writer.add_scalars("ACCURACY", metrics, self.iter)
            self.writer.add_scalars("NEGATIVES", neg, self.iter)
            self.writer.add_scalars("POSITIVES", pos, self.iter)
            for key, value in {**metrics, **pos, **neg}.items():
                global_params[key] = value

        if self._rolling is None:
            self._rolling = global_params.to_frame().T
        elif global_params.name not in self._rolling.index:
            self._rolling = self._rolling.append(global_params)
        if len(self._rolling) > 100:
            self._rolling = self._rolling.drop(self._rolling.index[0])
            conv_params = ["-ELBO", "proximity_loc", "gain_loc", "lamda_loc"]
            if all(
                self._rolling[p].std() / self._rolling[p].iloc[-50:].std() < 1.05
                for p in conv_params
            ):
                self._stop = True

        global_params.to_csv(self.path / "global_params.csv")
        self._rolling.to_csv(self.path / "rolling_params.csv")
        self.logger.info("Step #{}.".format(self.iter))

    def load_checkpoint(self, path=None):
        path = Path(path) if path else self.path
        global_params = pd.read_csv(
            path / "global_params.csv", squeeze=True, index_col=0
        )
        self._rolling = pd.read_csv(path / "rolling_params.csv", index_col=0)
        self.iter = int(global_params.name)
        self.optim.load(path / "optimizer")
        pyro.clear_param_store()
        pyro.get_param_store().load(path / "params", map_location=self.device)
        # self.predictions = np.load(os.path.join(path, "predictions.npy"))
        self.logger.info(
            "Step #{}. Loaded model params and optimizer state from {}".format(
                self.iter, path
            )
        )

    def load_parameters(self, path=None):
        path = Path(path) if path else self.path
        pyro.clear_param_store()
        pyro.get_param_store().load(path / "params", map_location=self.device)
        self._K = 2
        self._S = 1

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
            weights = self.data_loc(
                torch.ones(1),
                pyro.param("d/w_mode"),
                pyro.param("d/x_mode"),
                pyro.param("d/y_mode"),
                torch.arange(self.data.N),
            )
            signal = (self.data.data * weights).sum(dim=(-2, -1))
            noise = (
                self.offset_var + (signal - self.offset_mean) * pyro.param("gain")
            ).sqrt()
            result = (signal - pyro.param("d/b_loc") - self.offset_mean) / noise
            mask = self.z_probs > 0.5
            return result[mask]
