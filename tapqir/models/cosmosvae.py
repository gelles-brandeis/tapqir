# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

import math
from typing import Union

import torch
import torch.distributions.constraints as constraints
import torch.nn as nn
from pyro.ops.indexing import Vindex
from pyroapi import distributions as dist
from pyroapi import handlers, infer, pyro
from torch.distributions import transforms
from torch.distributions.utils import lazy_property
from torch.nn.functional import one_hot

from tapqir.distributions import KSMOGN, AffineBeta
from tapqir.distributions.util import expand_offtarget, probs_m, probs_theta
from tapqir.models.model import Model


class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(196, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fch = nn.Linear(hidden_dim, z_dim)
        self.fcx = nn.Linear(hidden_dim, z_dim)
        self.fcy = nn.Linear(hidden_dim, z_dim)
        self.fcb = nn.Linear(hidden_dim, 1)
        # setup the non-linearities
        self.activation = nn.ReLU()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        batch_shape = x.shape[:-2]
        shape = batch_shape + (196,)
        x = x.reshape(shape)
        # then compute the hidden units
        z = self.activation(self.fc1(x))
        z = self.activation(self.fc2(z))
        h_loc = torch.exp(self.fch(z))
        # x_loc = torch.exp(self.fcx(z))
        loc = -(14 + 1) / 2 + torch.finfo(x.dtype).eps
        scale = (14 + 1) - 2 * torch.finfo(x.dtype).eps
        x_loc = transforms.ComposeTransform(
            [transforms.SigmoidTransform(), transforms.AffineTransform(loc, scale)]
        )(self.fcx(z))
        y_loc = transforms.ComposeTransform(
            [transforms.SigmoidTransform(), transforms.AffineTransform(loc, scale)]
        )(self.fcy(z))
        # y_loc = torch.exp(self.fcy(z))
        # b_loc = torch.exp(self.fcb(z))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        #  z_loc = self.softplus(self.fc21(hidden))
        #  # z_loc = torch.exp(self.fc21(hidden))
        #  z_scale = torch.exp(self.fc22(hidden))
        return h_loc, x_loc, y_loc


# A general purpose module to construct networks that look like:
# [Linear (256 -> 1)]
# [Linear (256 -> 256), ReLU (), Linear (256 -> 1)]
# [Linear (256 -> 256), ReLU (), Linear (256 -> 1), ReLU ()]
# etc.
class MLP(nn.Module):
    def __init__(
        self, in_size, out_sizes, non_linear_layer, output_non_linearity=False
    ):
        super().__init__()
        assert len(out_sizes) >= 1
        layers = []
        in_sizes = [in_size] + out_sizes[0:-1]
        sizes = list(zip(in_sizes, out_sizes))
        for (i, o) in sizes[0:-1]:
            layers.append(nn.Linear(i, o))
            layers.append(non_linear_layer())
        layers.append(nn.Linear(sizes[-1][0], sizes[-1][1]))
        if output_non_linearity:
            layers.append(non_linear_layer())
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


# Takes the guide RNN hidden state to parameters of the guide
# distributions over z_where and z_pres.
class Predict(nn.Module):
    def __init__(self, input_size, h_sizes, non_linear_layer):
        super().__init__()
        #  self.z_pres_size = z_pres_size
        #  self.z_where_size = z_where_size
        # output_size = z_pres_size + 2 * z_where_size
        output_size = 2
        self.mlp = MLP(input_size, h_sizes + [output_size], non_linear_layer)

    def forward(self, h):
        out = self.mlp(h)
        # z_pres_p = torch.sigmoid(out[:, 0 : self.z_pres_size])
        x = out[:, 0]
        y = out[:, 1]
        #  z_where_loc = out[:, self.z_pres_size : self.z_pres_size + self.z_where_size]
        #  z_where_scale = softplus(out[:, (self.z_pres_size + self.z_where_size) :])
        loc = -(14 + 1) / 2 + torch.finfo(x.dtype).eps
        scale = (14 + 1) - 2 * torch.finfo(x.dtype).eps
        x_loc = transforms.ComposeTransform(
            [transforms.SigmoidTransform(), transforms.AffineTransform(loc, scale)]
        )(x)
        y_loc = transforms.ComposeTransform(
            [transforms.SigmoidTransform(), transforms.AffineTransform(loc, scale)]
        )(y)
        #  x_loc = torch.clamp(x, min=loc, max=loc+scale)
        #  y_loc = torch.clamp(y, min=loc, max=loc+scale)
        return x_loc, y_loc
        # return z_pres_p, z_where_loc, z_where_scale


class CosmosVAE(Model):
    r"""
    **Single-Color Time-Independent Colocalization Model**

    **Reference**:

    1. Ordabayev YA, Friedman LJ, Gelles J, Theobald DL.
       Bayesian machine learning analysis of single-molecule fluorescence colocalization images.
       eLife. 2022 March. doi: `10.7554/eLife.73860 <https://doi.org/10.7554/eLife.73860>`_.

    :param S: Number of distinct molecular states for the binder molecules.
    :param K: Maximum number of spots that can be present in a single image.
    :param channels: Number of color channels.
    :param device: Computation device (cpu or gpu).
    :param dtype: Floating point precision.
    :param use_pykeops: Use pykeops as backend to marginalize out offset.
    """

    name = "cosmosvae"

    def __init__(
        self,
        S: int = 1,
        K: int = 2,
        channels: Union[tuple, list] = (0,),
        device: str = "cpu",
        dtype: str = "double",
        use_pykeops: bool = True,
        background_mean_std: float = 1000,
        background_std_std: float = 100,
        lamda_rate: float = 1,
        height_std: float = 10000,
        width_min: float = 0.75,
        width_max: float = 2.25,
        proximity_rate: float = 1,
        gain_std: float = 50,
    ):
        super().__init__(S, K, channels, device, dtype)
        assert S == 1, "This is a single-state model!"
        assert len(self.channels) == 1, "Please specify exactly one color channel"
        self.cdx = torch.tensor(self.channels[0])
        self.full_name = f"{self.name}-channel{self.cdx}"
        self._global_params = ["gain", "proximity", "lamda", "pi"]
        self.use_pykeops = use_pykeops
        self.conv_params = ["-ELBO", "proximity_loc", "gain_loc", "lamda_loc"]
        # priors settings
        self.background_mean_std = background_mean_std
        self.background_std_std = background_std_std
        self.lamda_rate = lamda_rate
        self.height_std = height_std
        self.width_min = width_min
        self.width_max = width_max
        self.proximity_rate = proximity_rate
        self.gain_std = gain_std

        # AIR
        rnn_input_size = 14 * 14 + 2
        rnn_hidden_size = 256
        self.rnn = nn.LSTMCell(rnn_input_size, rnn_hidden_size)
        # self.encoder = Encoder(2, 400)
        non_linearity = "ReLU"
        nl = getattr(nn, non_linearity)
        predict_net = [200]
        self.predict = Predict(rnn_hidden_size, predict_net, nl)

        # Create parameters.
        self.h_init = nn.Parameter(torch.zeros(1, rnn_hidden_size))
        self.c_init = nn.Parameter(torch.zeros(1, rnn_hidden_size))
        self.x_init = nn.Parameter(torch.zeros(1, 1))
        self.y_init = nn.Parameter(torch.zeros(1, 1))

    def model(self):
        r"""
        **Generative Model**
        """
        # global parameters
        gain = pyro.sample("gain", dist.HalfNormal(self.gain_std))
        pi = pyro.sample("pi", dist.Dirichlet(torch.ones(self.S + 1) / (self.S + 1)))
        pi = expand_offtarget(pi)
        lamda = pyro.sample("lamda", dist.Exponential(self.lamda_rate))
        proximity = pyro.sample("proximity", dist.Exponential(self.proximity_rate))
        size = torch.stack(
            (
                torch.full_like(proximity, 2.0),
                (((self.data.P + 1) / (2 * proximity)) ** 2 - 1),
            ),
            dim=-1,
        )

        # spots
        spots = pyro.plate("spots", self.K)
        # aoi sites
        aois = pyro.plate(
            "aois",
            self.data.Nt,
            subsample=self.n,
            subsample_size=self.nbatch_size,
            dim=-2,
        )
        # time frames
        frames = pyro.plate(
            "frames",
            self.data.F,
            subsample=self.f,
            subsample_size=self.fbatch_size,
            dim=-1,
        )

        with aois as ndx:
            ndx = ndx[:, None]
            mask = Vindex(self.data.mask)[ndx].to(self.device)
            with handlers.mask(mask=mask):
                # background mean and std
                background_mean = pyro.sample(
                    "background_mean", dist.HalfNormal(self.background_mean_std)
                )
                background_std = pyro.sample(
                    "background_std", dist.HalfNormal(self.background_std_std)
                )
                with frames as fdx:
                    # fetch data
                    obs, target_locs, is_ontarget = self.data.fetch(ndx, fdx, self.cdx)
                    # sample background intensity
                    background = pyro.sample(
                        "background",
                        dist.Gamma(
                            (background_mean / background_std) ** 2,
                            background_mean / background_std**2,
                        ),
                    )

                    # sample hidden model state (1+S,)
                    z = pyro.sample(
                        "z",
                        dist.Categorical(Vindex(pi)[..., :, is_ontarget.long()]),
                        infer={"enumerate": "parallel"},
                    )
                    theta = pyro.sample(
                        "theta",
                        dist.Categorical(
                            Vindex(probs_theta(self.K, self.device))[
                                torch.clamp(z, min=0, max=1)
                            ]
                        ),
                        infer={"enumerate": "parallel"},
                    )
                    onehot_theta = one_hot(theta, num_classes=1 + self.K)

                    ms, heights, widths, xs, ys = [], [], [], [], []
                    for kdx in spots:
                        specific = onehot_theta[..., 1 + kdx]
                        # spot presence
                        m = pyro.sample(
                            f"m_{kdx}",
                            dist.Bernoulli(
                                Vindex(probs_m(lamda, self.K))[..., theta, kdx]
                            ),
                        )
                        with handlers.mask(mask=m > 0):
                            # sample spot variables
                            height = pyro.sample(
                                f"height_{kdx}",
                                dist.HalfNormal(self.height_std),
                            )
                            width = pyro.sample(
                                f"width_{kdx}",
                                AffineBeta(
                                    1.5,
                                    2,
                                    self.width_min,
                                    self.width_max,
                                ),
                            )
                            x = pyro.sample(
                                f"x_{kdx}",
                                AffineBeta(
                                    0,
                                    Vindex(size)[..., specific],
                                    -(self.data.P + 1) / 2,
                                    (self.data.P + 1) / 2,
                                ),
                            )
                            y = pyro.sample(
                                f"y_{kdx}",
                                AffineBeta(
                                    0,
                                    Vindex(size)[..., specific],
                                    -(self.data.P + 1) / 2,
                                    (self.data.P + 1) / 2,
                                ),
                            )

                        # append
                        # height = height * background
                        ms.append(m)
                        heights.append(height)
                        widths.append(width)
                        xs.append(x)
                        ys.append(y)

                    # observed data
                    pyro.sample(
                        "data",
                        KSMOGN(
                            torch.stack(heights, -1),
                            torch.stack(widths, -1),
                            torch.stack(xs, -1),
                            torch.stack(ys, -1),
                            target_locs,
                            background,
                            gain,
                            self.data.offset.samples,
                            self.data.offset.logits.to(self.dtype),
                            self.data.P,
                            torch.stack(torch.broadcast_tensors(*ms), -1),
                            self.use_pykeops,
                        ),
                        obs=obs,
                    )

    def guide(self):
        # register PyTorch module `encoder` with Pyro
        pyro.module("predict", self.predict)
        pyro.module("rnn", self.rnn)
        pyro.param("h_init", self.h_init)
        pyro.param("c_init", self.c_init)
        pyro.param("x_init", self.x_init)
        pyro.param("y_init", self.y_init)
        # global parameters
        pyro.sample(
            "gain",
            dist.Gamma(
                pyro.param("gain_loc") * pyro.param("gain_beta"),
                pyro.param("gain_beta"),
            ),
        )
        pyro.sample("pi", dist.Dirichlet(pyro.param("pi_mean") * pyro.param("pi_size")))
        pyro.sample(
            "lamda",
            dist.Gamma(
                pyro.param("lamda_loc") * pyro.param("lamda_beta"),
                pyro.param("lamda_beta"),
            ),
        )
        pyro.sample(
            "proximity",
            AffineBeta(
                pyro.param("proximity_loc"),
                pyro.param("proximity_size"),
                0,
                (self.data.P + 1) / math.sqrt(12),
            ),
        )

        # spots
        spots = pyro.plate("spots", self.K)
        # aoi sites
        aois = pyro.plate(
            "aois",
            self.data.Nt,
            subsample=self.n,
            subsample_size=self.nbatch_size,
            dim=-2,
        )
        # time frames
        frames = pyro.plate(
            "frames",
            self.data.F,
            subsample=self.f,
            subsample_size=self.fbatch_size,
            dim=-1,
        )

        with aois as ndx:
            ndx = ndx[:, None]
            mask = Vindex(self.data.mask)[ndx].to(self.device)
            with handlers.mask(mask=mask):
                pyro.sample(
                    "background_mean",
                    dist.Delta(Vindex(pyro.param("background_mean_loc"))[ndx, 0]),
                )
                pyro.sample(
                    "background_std",
                    dist.Delta(Vindex(pyro.param("background_std_loc"))[ndx, 0]),
                )
                with frames as fdx:
                    # sample background intensity
                    background = pyro.sample(
                        "background",
                        dist.Gamma(
                            Vindex(pyro.param("b_loc"))[ndx, fdx]
                            * Vindex(pyro.param("b_beta"))[ndx, fdx],
                            Vindex(pyro.param("b_beta"))[ndx, fdx],
                        ),
                    )

                    # fetch data
                    obs, _, __ = self.data.fetch(ndx, fdx, self.cdx)
                    # use the encoder to get the parameters used to define q(z|x)
                    #  h_loc, x_loc, y_loc = self.encoder(
                    #      (obs - self.data.offset.mean) / background[..., None, None] - 1
                    #  )
                    # data = (obs - self.data.offset.mean) / background[..., None, None] - 1
                    data = (obs - 90 - 150) / 150
                    #  batch_shape = data.shape[:-2]
                    #  shape = batch_shape + (196,)
                    shape = background.shape
                    #  nbatch_size = data.shape[0]
                    #  fbatch_size = data.shape[1]
                    data = data.expand(shape + (14, 14))
                    data = data.reshape(-1, 196)
                    n = torch.numel(background)

                    state = {
                        "h": self.h_init.expand(n, -1),
                        "c": self.c_init.expand(n, -1),
                        "x": self.x_init.expand(n, -1),
                        "y": self.y_init.expand(n, -1),
                    }
                    for kdx in spots:
                        # state = self.guide_step(kdx, ndx, fdx, state, data)
                        rnn_input = torch.cat((data, state["x"], state["y"]), -1)
                        h, c = self.rnn(rnn_input, (state["h"], state["c"]))
                        x_loc, y_loc = self.predict(h)
                        x_loc = x_loc.reshape(shape)
                        y_loc = y_loc.reshape(shape)
                        # sample spot presence m
                        m = pyro.sample(
                            f"m_{kdx}",
                            dist.Bernoulli(
                                Vindex(pyro.param("m_probs"))[kdx, ndx, fdx]
                            ),
                            infer={"enumerate": "parallel"},
                        )
                        with handlers.mask(mask=m > 0):
                            # sample spot variables
                            pyro.sample(
                                f"height_{kdx}",
                                dist.Gamma(
                                    # h_loc[..., kdx]
                                    Vindex(pyro.param("h_loc"))[kdx, ndx, fdx]
                                    * Vindex(pyro.param("h_beta"))[kdx, ndx, fdx],
                                    Vindex(pyro.param("h_beta"))[kdx, ndx, fdx],
                                ),
                                # dist.Delta(torch.tensor(3000.0)),
                            )
                            pyro.sample(
                                f"width_{kdx}",
                                #  AffineBeta(
                                #      Vindex(pyro.param("w_mean"))[kdx, ndx, fdx],
                                #      Vindex(pyro.param("w_size"))[kdx, ndx, fdx],
                                #      0.75,
                                #      2.25,
                                #  ),
                                dist.Delta(torch.tensor(1.4)),
                            )
                            x = pyro.sample(
                                f"x_{kdx}",
                                AffineBeta(
                                    x_loc,
                                    # Vindex(pyro.param("x_mean"))[kdx, ndx, fdx],
                                    Vindex(pyro.param("size"))[kdx, ndx, fdx],
                                    -(self.data.P + 1) / 2,
                                    (self.data.P + 1) / 2,
                                ),
                            )
                            y = pyro.sample(
                                f"y_{kdx}",
                                AffineBeta(
                                    y_loc,
                                    # y_loc[..., kdx],
                                    # Vindex(pyro.param("y_mean"))[kdx, ndx, fdx],
                                    Vindex(pyro.param("size"))[kdx, ndx, fdx],
                                    -(self.data.P + 1) / 2,
                                    (self.data.P + 1) / 2,
                                ),
                            )
                            state = {
                                "h": h,
                                "c": c,
                                "x": x.reshape(-1, 1),
                                "y": y.reshape(-1, 1),
                            }

    def init_parameters(self):
        """
        Initialize variational parameters.
        """
        device = self.device
        data = self.data

        pyro.param(
            "pi_mean",
            lambda: torch.ones(self.S + 1, device=device),
            constraint=constraints.simplex,
        )
        pyro.param(
            "pi_size",
            lambda: torch.tensor(2, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "m_probs",
            lambda: torch.full((self.K, data.Nt, data.F), 0.5, device=device),
            constraint=constraints.unit_interval,
        )

        self._init_parameters()

    def _init_parameters(self):
        """
        Parameters shared between different models.
        """
        device = self.device
        data = self.data

        pyro.param(
            "proximity_loc",
            lambda: torch.tensor(0.5, device=device),
            constraint=constraints.interval(
                0,
                (self.data.P + 1) / math.sqrt(12) - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            "proximity_size",
            lambda: torch.tensor(100, device=device),
            constraint=constraints.greater_than(2.0),
        )
        pyro.param(
            "lamda_loc",
            lambda: torch.tensor(0.5, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "lamda_beta",
            lambda: torch.tensor(100, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "gain_loc",
            lambda: torch.tensor(5, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "gain_beta",
            lambda: torch.tensor(100, device=device),
            constraint=constraints.positive,
        )

        pyro.param(
            "background_mean_loc",
            lambda: torch.full(
                (data.Nt, 1),
                data.median[self.cdx] - self.data.offset.mean,
                device=device,
            ),
            constraint=constraints.positive,
        )
        pyro.param(
            "background_std_loc",
            lambda: torch.ones(data.Nt, 1, device=device),
            constraint=constraints.positive,
        )

        pyro.param(
            "b_loc",
            lambda: torch.full(
                (data.Nt, data.F),
                data.median[self.cdx] - self.data.offset.mean,
                device=device,
            ),
            constraint=constraints.positive,
        )
        pyro.param(
            "b_beta",
            lambda: torch.ones(data.Nt, data.F, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "h_loc",
            lambda: torch.full((self.K, data.Nt, data.F), 2000, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "h_beta",
            lambda: torch.full((self.K, data.Nt, data.F), 0.001, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "w_mean",
            lambda: torch.full((self.K, data.Nt, data.F), 1.5, device=device),
            constraint=constraints.interval(
                0.75 + torch.finfo(self.dtype).eps,
                2.25 - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            "w_size",
            lambda: torch.full((self.K, data.Nt, data.F), 100, device=device),
            constraint=constraints.greater_than(2.0),
        )
        pyro.param(
            "x_mean",
            lambda: torch.zeros(self.K, data.Nt, data.F, device=device),
            constraint=constraints.interval(
                -(data.P + 1) / 2 + torch.finfo(self.dtype).eps,
                (data.P + 1) / 2 - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            "y_mean",
            lambda: torch.zeros(self.K, data.Nt, data.F, device=device),
            constraint=constraints.interval(
                -(data.P + 1) / 2 + torch.finfo(self.dtype).eps,
                (data.P + 1) / 2 - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            "size",
            lambda: torch.full((self.K, data.Nt, data.F), 200, device=device),
            constraint=constraints.greater_than(2.0),
        )

    def TraceELBO(self, jit=False):
        """
        A trace implementation of ELBO-based SVI that supports - exhaustive enumeration over
        discrete sample sites, and - local parallel sampling over any sample site in the guide.
        """
        return (infer.JitTraceEnum_ELBO if jit else infer.TraceEnum_ELBO)(
            max_plate_nesting=2, ignore_jit_warnings=True
        )

    @lazy_property
    def compute_probs(self) -> torch.Tensor:
        z_probs = torch.zeros(self.data.Nt, self.data.F)
        theta_probs = torch.zeros(self.K, self.data.Nt, self.data.F)
        nbatch_size = self.nbatch_size
        fbatch_size = self.fbatch_size
        N = sum(self.data.is_ontarget)
        for ndx in torch.split(torch.arange(N), nbatch_size):
            for fdx in torch.split(torch.arange(self.data.F), fbatch_size):
                self.n = ndx
                self.f = fdx
                self.nbatch_size = len(ndx)
                self.fbatch_size = len(fdx)
                with torch.no_grad(), pyro.plate(
                    "particles", size=50, dim=-3
                ), handlers.enum(first_available_dim=-4):
                    guide_tr = handlers.trace(self.guide).get_trace()
                    model_tr = handlers.trace(
                        handlers.replay(
                            handlers.block(self.model, hide=["data"]), trace=guide_tr
                        )
                    ).get_trace()
                model_tr.compute_log_prob()
                guide_tr.compute_log_prob()
                # 0 - theta
                # 1 - z
                # 2 - m_1
                # 3 - m_0
                # p(z, theta, phi)
                logp = 0
                for name in ["z", "theta", "m_0", "m_1", "x_0", "x_1", "y_0", "y_1"]:
                    logp = logp + model_tr.nodes[name]["unscaled_log_prob"]
                # p(z, theta | phi) = p(z, theta, phi) - p(z, theta, phi).sum(z, theta)
                logp = logp - logp.logsumexp((0, 1))
                expectation = (
                    guide_tr.nodes["m_0"]["unscaled_log_prob"]
                    + guide_tr.nodes["m_1"]["unscaled_log_prob"]
                    + logp
                )
                # average over m
                result = expectation.logsumexp((2, 3))
                # marginalize theta
                z_logits = result.logsumexp(0)
                z_probs[ndx[:, None], fdx] = z_logits[1].exp().mean(-3)
                # marginalize z
                theta_logits = result.logsumexp(1)
                theta_probs[:, ndx[:, None], fdx] = theta_logits[1:].exp().mean(-3)
        self.n = None
        self.f = None
        self.nbatch_size = nbatch_size
        self.fbatch_size = fbatch_size
        return z_probs, theta_probs

    @property
    def z_probs(self) -> torch.Tensor:
        r"""
        Probability of there being a target-specific spot :math:`p(z=1)`
        """
        return self.compute_probs[0]

    @property
    def theta_probs(self) -> torch.Tensor:
        r"""
        Posterior target-specific spot probability :math:`q(\theta = k)`.
        """
        return self.compute_probs[1]

    @property
    def m_probs(self) -> torch.Tensor:
        r"""
        Posterior spot presence probability :math:`q(m=1)`.
        """
        return pyro.param("m_probs").data

    @property
    def pspecific(self) -> torch.Tensor:
        r"""
        Probability of there being a target-specific spot :math:`p(\mathsf{specific})`
        """
        return self.z_probs

    @property
    def z_map(self) -> torch.Tensor:
        return self.z_probs > 0.5
