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
import torch.distributions.constraints as constraints
from torch.distributions.utils import probs_to_logits, logits_to_probs
from sklearn.metrics import matthews_corrcoef, confusion_matrix, \
    recall_score, precision_score
import logging
from cosmos import __version__ as cosmos_version
from tqdm import tqdm
from cosmos.utils.dataset import load_data
import itertools


class GaussianSpot(nn.Module):
    r"""
    Calculates ideal shape of the 2D-Gaussian spot given spot parameters,
    target positions, and drift list.

        :math:`\dfrac{h_{knf}}{2 \pi w^2_{nfk}} \exp{\left ( -\dfrac{(i-x_{nfk})^2 + (j-y_{nfk})^2}{2w^2_{nfk}} \right)}`

    :param target: AoI target positions.
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
            drift[["dx", "dy"]].values.reshape(-1, 1, 2)
            + target[["x", "y"]].values.reshape(-1, 1, 1, 2),
        ).float()

    # Ideal 2D gaussian spots
    def forward(self, height, width, x, y, ndx, fdx=None):
        r"""
        :param height: integrated spot intensity.
        :param width: width of the 2D-Gaussian spot.
        :param x: relative :math:`x`-axis position relative to the target.
        :param y: relative :math:`y`-axis position relative to the target.
        :param ndx: AoI indices.
        :param fdx: Frame indices.
        :return: Ideal shape 2D-Gaussian spot.
        :rtype: ~pyro.distributions.Categorical
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


class Model:
    r"""
    Base class for cosmos models.

    **Implementing New Models**:

    Derived models must implement the methods
    :meth:`model`
    :meth:`guide`
    """

    def __init__(self, S, K=2):
        super().__init__()
        self._K = K
        self._S = S
        self.batch_size = None
        # for plotting
        self.n = None
        self.frames = None

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

    @property
    def num_states(self):
        r"""
        Total number of states for the image model given by:

            :math:`2^K + S K 2^{K-1}`
        """
        return 2**self.K + self.S * self.K * 2**(self.K-1)

    @property
    def logits_j(self):
        result = torch.zeros(2, self.K+1, dtype=torch.float)
        result[0, :self.K] = dist.Poisson(param("rate_j")).log_prob(torch.arange(self.K).float())
        result[0, -1] =  torch.log1p(-result[0, :self.K].exp().sum())
        result[1, :self.K-1] = dist.Poisson(param("rate_j")).log_prob(torch.arange(self.K-1).float())
        result[1, -2] = torch.log1p(-result[0, :self.K-1].exp().sum())
        return result

    @property
    def logits_state(self):
        logits_z = Vindex(param("logits_z"))[self.state_to_z.sum(-1)]
        logits_j = Vindex(self.logits_j)[self.ontarget.sum(-1), self.state_to_j.sum(-1)]
        _, idx, counts = torch.unique(
            torch.stack((self.state_to_z.sum(-1), self.state_to_j.sum(-1)), -1),
            return_counts=True, return_inverse=True, dim=0)
        return logits_z + logits_j - torch.log(Vindex(counts)[idx].float())

    @property
    def state_to_z(self):
        result = torch.zeros(self.num_states, self.K, dtype=torch.long)
        for i in range(2**self.K, self.num_states):
            s, r = divmod(i - 2**self.K, self.K * 2**(self.K-1))
            k, t = divmod(r, 2**(self.K-1))
            result[i, k] = s+1
        return result

    @property
    def ontarget(self):
        return torch.clamp(self.state_to_z, min=0, max=1)

    @property
    def state_to_j(self):
        result = torch.zeros(self.num_states, self.K, dtype=torch.long)
        k_lst = torch.tensor(list(itertools.product([0, 1], repeat=self.K)), dtype=torch.long)
        km1_lst = torch.tensor(list(itertools.product([0, 1], repeat=self.K-1)), dtype=torch.long)
        kdx = torch.arange(self.K)
        result[:2**self.K] = k_lst
        for s in range(self.S):
            for k in range(self.K):
                km1dx = torch.cat([kdx[:k], kdx[k+1:]])
                result[2**self.K+(s*self.K+k)*2**(self.K-1):2**self.K+(s*self.K+k+1)*2**(self.K-1), km1dx] = km1_lst
        return result

    @property
    def state_to_m(self):
        return torch.clamp(self.state_to_z + self.state_to_j, min=0, max=1)

    @property
    def z_probs(self):
        r"""
        Probability of an on-target spot :math:`p(z_{knf})`.
        """
        return torch.einsum(
            "nfi,iks->nfks",
            logits_to_probs(param("d/logits_state").data),
            torch.eye(self.S+1)[self.state_to_z])

    @property
    def j_probs(self):
        r"""
        Probability of an off-target spot :math:`p(j_{knf})`.
        """
        return torch.einsum(
            "nfi,ikt->nfkt",
            logits_to_probs(param("d/logits_state").data),
            torch.eye(2)[self.state_to_j])

    @property
    def m_probs(self):
        r"""
        Probability of a spot :math:`p(m_{knf})`.
        """
        return torch.einsum(
            "nfi,ikt->nfkt",
            logits_to_probs(param("d/logits_state").data),
            torch.eye(2)[self.state_to_m])

    @property
    def z_marginal(self):
        return self.z_probs[..., 1:].sum(dim=(-2,-1))

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
                self.infer()
                self.save_checkpoint()
            self.iter += 1

    def log(self):
        self.path = os.path.join(
            self.data_path, "runs",
            "{}".format(self.name),
            "{}state".format(cosmos_version.split("+")[0]),
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
            raise ValueError("Step #{}. Detected NaN values in parameters".format(self.iter))

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
            pred_labels = (self.z_marginal > 0.5).cpu()[mask]
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
                    self.data.labels["spotpicker"] != (self.z_marginal > 0.5)] = 2
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
        params_last = pd.read_csv(os.path.join(path, "params_last.csv"), header=None, squeeze=True, index_col=0)
        self.iter = int(params_last["iter"])
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
