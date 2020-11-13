import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import pyro
import pyro.distributions as dist
from pyro import param
from pyro.infer import SVI
from pyro.infer import JitTraceEnum_ELBO, TraceEnum_ELBO
from pyro.ops.indexing import Vindex
from pyro.ops.stats import quantile
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import matthews_corrcoef, confusion_matrix, \
    recall_score, precision_score
import logging
from tapqir import __version__ as tapqir_version
from tqdm import tqdm
from tapqir.utils.dataset import load_data
from scipy.io import savemat


class GaussianSpot:
    r"""
    Calculates ideal shape of the 2D-Gaussian spot given spot parameters,
    target positions, and drift list.

        :math:`\dfrac{h_{knf}}{2 \pi w^2_{nfk}} \exp{\left ( -\dfrac{(i-x_{nfk})^2 + (j-y_{nfk})^2}{2w^2_{nfk}} \right)}`

    :param target: Target positions.
    :param drift: Frame drift list.
    """

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
            scale_tril=torch.eye(2) * width[..., None, None, None, None]
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

    @property
    def S(self):
        r"""
        Number of distinct molecular states for the binder molecules.
        """
        return self._S

    @property
    def K(self):
        r"""
        Maximum number of spots that can be present in a single image.
        """
        return self._K

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
        offset_max = quantile(self.data.offset.flatten(), 0.995).item()
        offset_min = quantile(self.data.offset.flatten(), 0.005).item()
        self.data.offset = torch.clamp(self.data.offset, offset_min, offset_max)
        self.offset_samples, self.offset_weights = torch.unique(self.data.offset, sorted=True, return_counts=True)
        self.offset_weights = self.offset_weights.float() / self.offset_weights.sum()
        self.offset_mean = torch.sum(self.offset_samples * self.offset_weights)
        self.offset_var = torch.sum(self.offset_samples ** 2 * self.offset_weights) - self.offset_mean ** 2

        # load control data
        if control:
            self.control = load_data(self.data_path, dtype="control", device=self.device)
            self.control_loc = GaussianSpot(
                self.control.target, self.control.drift,
                self.control.D)
        else:
            self.control = control

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
        except FileNotFoundError:
            pyro.clear_param_store()

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
        for i in tqdm(range(num_iter)):
            # with torch.autograd.detect_anomaly():
            # import pdb; pdb.set_trace()
            self.iter_loss = self.svi.step()
            if not self.iter % 100:
                self.save_checkpoint()
            self.iter += 1

    def log(self):
        self.path = os.path.join(
            self.data_path, "runs",
            "{}".format(self.name),
            "{}".format(tapqir_version.split("+")[0]),
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
        for k, v in pyro.get_param_store().items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                # import pdb; pdb.set_trace()
                raise ValueError("Step #{}. Detected NaN values in {}".format(self.iter, k))
        # if any([torch.isnan(v).any()
        #         for v in pyro.get_param_store().values()]):

        # save parameters and optimizer state
        pyro.get_param_store().save(os.path.join(self.path, "params"))
        self.optim.save(os.path.join(self.path, "optimizer"))

        # save parameters in matlab format
        keys = ["h_loc", "w_mean", "x_mean", "y_mean", "b_loc"]
        matlab = {k: param(f"d/{k}").data.cpu().numpy() for k in keys}
        matlab["parametersDescription"] = \
            "Parameters for N x F x K spots. \
            N - target sites, F - frames, K - max number of spots in the image. \
            h_loc - mean intensity, w_mean - mean spot width, \
            x_mean - x position, y_mean - y position, \
            b_loc - background intensity."
        if self.name == "spotdetection":
            matlab["z_probs"] = self.z_probs.cpu().numpy()
            matlab["j_probs"] = self.j_probs.cpu().numpy()
            matlab["m_probs"] = self.m_probs.cpu().numpy()
            matlab["z_marginal"] = self.z_marginal.cpu().numpy()
            matlab["probabilitiesDescription"] = \
                "Probabilities for N x F x K spots. \
                z_probs - on-target spot probability, \
                j_probs - off-target spot probability, \
                m_probs - spot probability (on-target + off-target), \
                z_marginal - total on-target spot probability (sum of z_probs)."
        matlab["aoilist"] = self.data.target.index.values
        matlab["aoilistDescription"] = "aoi numbering from aoiinfo"
        matlab["framelist"] = self.data.drift.index.values
        matlab["framelistDescription"] = "frame numbering from driftlist"
        savemat(os.path.join(self.path, "parameters.mat"), matlab)

        # save global paramters in csv file and for tensorboard
        global_params = pd.Series(data={"iter": self.iter})

        self.writer.add_scalar(
            "-ELBO", self.iter_loss, self.iter)
        global_params["-ELBO"] = self.iter_loss
        for name, val in pyro.get_param_store().items():
            if val.dim() == 0:
                self.writer.add_scalar(name, val.item(), self.iter)
                global_params[name] = val.item()
            elif val.dim() == 1 and len(val) <= self.S+1:
                scalars = {str(i): v.item() for i, v in enumerate(val)}
                self.writer.add_scalars(name, scalars, self.iter)
                for key, value in scalars.items():
                    global_params["{}_{}".format(name, key)] = value

        if self.data.labels is not None and self.name == "spotdetection":
            mask = self.data.labels["z"] < 2
            pred_labels = (self.z_marginal > 0.5).cpu().numpy()[mask]
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
                global_params[key] = value
            try:
                atten_labels = np.copy(self.data.labels)
                atten_labels["z"][
                    self.data.labels["spotpicker"] != (self.z_marginal > 0.5)] = 2
                atten_labels["spotpicker"] = 0
                np.save(os.path.join(self.path, "atten_labels.npy"),
                        atten_labels)
            except:
                pass

        global_params.to_csv(os.path.join(self.path, "global_params.csv"))
        self.logger.info("Step #{}.".format(self.iter))

    def load_checkpoint(self, path=None):
        if path is None:
            path = self.path
        global_params = pd.read_csv(os.path.join(path, "global_params.csv"), header=None, squeeze=True, index_col=0)
        self.iter = int(global_params["iter"])
        self.optim.load(os.path.join(path, "optimizer"))
        pyro.clear_param_store()
        pyro.get_param_store().load(
            os.path.join(path, "params"),
            map_location=self.device)
        # self.predictions = np.load(os.path.join(path, "predictions.npy"))
        self.logger.info(
            "Step #{}. Loaded model params and optimizer state from {}"
            .format(self.iter, path))

    def load_parameters(self, path=None):
        if path is None:
            path = self.path
        pyro.clear_param_store()
        pyro.get_param_store().load(
            os.path.join(path, "params"),
            map_location=self.device)
        self._K = 2 # param("d/h_loc").shape[-1]
        self._S = 1
        # self._S = len(param("probs_z")) - 1
        # self.predictions = np.load(os.path.join(path, "predictions.npy"))

    def snr(self):
        r"""
        Calculate the signal-to-noise ratio.

        Total signal:

            :math:`\mu_{knf} =  \sum_{ij} I_{nfij} \mathcal{N}(i, j \mid x_{knf}, y_{knf}, w_{knf})`

        Noise:

            :math:`\sigma^2_{knf} = \sigma^2_{\text{offset}} + \mu_{knf} \text{gain}`

        Signal-to-noise ratio:

            :math:`\text{SNR}_{knf} = \dfrac{\mu_{knf} - b_{nf} - \mu_{\text{offset}}}{\sigma_{knf}} \text{ for } \theta_{nf} = k`
        """
        with torch.no_grad():
            weights = self.data_loc(
                torch.ones(1),
                param("d/w_mode"),
                param("d/x_mode"),
                param("d/y_mode"),
                torch.arange(self.data.N))
            signal = (self.data.data * weights).sum(dim=(-2, -1))
            noise = (self.offset_var + (signal - self.offset_mean) * param("gain")).sqrt()
            result = (signal - param("d/b_loc") - self.offset_mean) / noise
            mask = self.z_probs > 0.5
            return result[mask]
