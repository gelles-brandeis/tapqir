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
from tapqir.models.cosmos import cosmos


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
        output_size = 3
        self.mlp = MLP(input_size, h_sizes + [output_size], non_linear_layer)

    def forward(self, h):
        out = self.mlp(h)
        # z_pres_p = torch.sigmoid(out[:, 0 : self.z_pres_size])
        w = out[:, 0]
        x = out[:, 1]
        y = out[:, 2]
        #  z_where_loc = out[:, self.z_pres_size : self.z_pres_size + self.z_where_size]
        #  z_where_scale = softplus(out[:, (self.z_pres_size + self.z_where_size) :])
        w_min = 0.75 + torch.finfo(x.dtype).eps
        w_scale = 1.5 - 2 * torch.finfo(x.dtype).eps
        w_mean = transforms.ComposeTransform(
            [transforms.SigmoidTransform(), transforms.AffineTransform(w_min, w_scale)]
        )(w)
        loc = -(14 + 1) / 2 + torch.finfo(x.dtype).eps
        scale = (14 + 1) - 2 * torch.finfo(x.dtype).eps
        x_loc = transforms.ComposeTransform(
            [transforms.SigmoidTransform(), transforms.AffineTransform(loc, scale)]
        )(x)
        y_loc = transforms.ComposeTransform(
            [transforms.SigmoidTransform(), transforms.AffineTransform(loc, scale)]
        )(y)
        return w_mean, x_loc, y_loc


class cosmosvae(cosmos):
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
        K: int = 2,
        Q: int = None,
        device: str = "cpu",
        dtype: str = "double",
        use_pykeops: bool = True,
        priors: dict = {
            "background_mean_std": 1000.0,
            "background_std_std": 100.0,
            "lamda_rate": 1.0,
            "height_std": 10000.0,
            "width_min": 0.75,
            "width_max": 2.25,
            "proximity_rate": 1.0,
            "gain_std": 50.0,
        },
    ):
        super().__init__(K=K, Q=Q, device=device, dtype=dtype, priors=priors)
        self._global_params = ["gain", "proximity", "lamda", "pi"]
        self.use_pykeops = use_pykeops
        self.conv_params = ["-ELBO", "proximity_loc", "gain_loc", "lamda_loc"]

        # AIR
        p = 3
        rnn_input_size = 14 * 14 + p
        rnn_hidden_size = 256
        self.rnn = nn.LSTMCell(rnn_input_size, rnn_hidden_size)
        non_linearity = "ReLU"
        nl = getattr(nn, non_linearity)
        predict_net = [200]
        self.predict = Predict(rnn_hidden_size, predict_net, nl)

        # Create parameters.
        self.h_init = nn.Parameter(torch.zeros(1, rnn_hidden_size))
        self.c_init = nn.Parameter(torch.zeros(1, rnn_hidden_size))
        self.width_init = nn.Parameter(torch.zeros(1, 1))
        self.x_init = nn.Parameter(torch.zeros(1, 1))
        self.y_init = nn.Parameter(torch.zeros(1, 1))

    def guide(self):
        # register PyTorch module `encoder` with Pyro
        pyro.module("predict", self.predict)
        pyro.module("rnn", self.rnn)
        pyro.param("h_init", self.h_init)
        pyro.param("c_init", self.c_init)
        pyro.param("width_init", self.width_init)
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
        pyro.sample(
            "pi",
            dist.Dirichlet(pyro.param("pi_mean") * pyro.param("pi_size")).to_event(1),
        )
        pyro.sample(
            "lamda",
            dist.Gamma(
                pyro.param("lamda_loc") * pyro.param("lamda_beta"),
                pyro.param("lamda_beta"),
            ).to_event(1),
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
            dim=-3,
        )
        # time frames
        frames = pyro.plate(
            "frames",
            self.data.F,
            subsample=self.f,
            subsample_size=self.fbatch_size,
            dim=-2,
        )
        # color channels
        channels = pyro.plate(
            "channels",
            self.data.C,
            dim=-1,
        )

        with channels as cdx, aois as ndx:
            ndx = ndx[:, None, None]
            mask = Vindex(self.data.mask)[ndx].to(self.device)
            with handlers.mask(mask=mask):
                pyro.sample(
                    "background_mean",
                    dist.Delta(Vindex(pyro.param("background_mean_loc"))[ndx, 0, cdx]),
                )
                pyro.sample(
                    "background_std",
                    dist.Delta(Vindex(pyro.param("background_std_loc"))[ndx, 0, cdx]),
                )
                with frames as fdx:
                    fdx = fdx[:, None]
                    # sample background intensity
                    background = pyro.sample(
                        "background",
                        dist.Gamma(
                            Vindex(pyro.param("b_loc"))[ndx, fdx, cdx]
                            * Vindex(pyro.param("b_beta"))[ndx, fdx, cdx],
                            Vindex(pyro.param("b_beta"))[ndx, fdx, cdx],
                        ),
                    )

                    # fetch data
                    obs, _, __ = self.data.fetch(ndx, fdx, cdx)
                    # data = (obs - self.data.offset.mean) / background[..., None, None] - 1
                    data = (obs - 90 - 150) / 150
                    shape = background.shape
                    data = data.expand(shape + (14, 14))
                    data = data.reshape(-1, 196)
                    n = torch.numel(background)

                    state = {
                        "h": self.h_init.expand(n, -1),
                        "c": self.c_init.expand(n, -1),
                        "w": self.width_init.expand(n, -1),
                        "x": self.x_init.expand(n, -1),
                        "y": self.y_init.expand(n, -1),
                    }
                    for kdx in spots:
                        # state = self.guide_step(kdx, ndx, fdx, state, data)
                        rnn_input = torch.cat(
                            (data, state["w"], state["x"], state["y"]), -1
                        )
                        h, c = self.rnn(rnn_input, (state["h"], state["c"]))
                        w_mean, x_loc, y_loc = self.predict(h)
                        w_mean = w_mean.reshape(shape)
                        x_loc = x_loc.reshape(shape)
                        y_loc = y_loc.reshape(shape)
                        # sample spot presence m
                        m = pyro.sample(
                            f"m_k{kdx}",
                            dist.Bernoulli(
                                Vindex(pyro.param("m_probs"))[kdx, ndx, fdx, cdx]
                            ),
                            infer={"enumerate": "parallel"},
                        )
                        with handlers.mask(mask=m > 0):
                            # sample spot variables
                            pyro.sample(
                                f"height_k{kdx}",
                                dist.Gamma(
                                    Vindex(pyro.param("h_loc"))[kdx, ndx, fdx, cdx]
                                    * Vindex(pyro.param("h_beta"))[kdx, ndx, fdx, cdx],
                                    Vindex(pyro.param("h_beta"))[kdx, ndx, fdx, cdx],
                                ),
                            )
                            w = pyro.sample(
                                f"width_k{kdx}",
                                AffineBeta(
                                    w_mean,
                                    Vindex(pyro.param("w_size"))[kdx, ndx, fdx, cdx],
                                    0.75,
                                    2.25,
                                ),
                            )
                            x = pyro.sample(
                                f"x_k{kdx}",
                                AffineBeta(
                                    x_loc,
                                    Vindex(pyro.param("size"))[kdx, ndx, fdx, cdx],
                                    -(self.data.P + 1) / 2,
                                    (self.data.P + 1) / 2,
                                ),
                            )
                            y = pyro.sample(
                                f"y_k{kdx}",
                                AffineBeta(
                                    y_loc,
                                    Vindex(pyro.param("size"))[kdx, ndx, fdx, cdx],
                                    -(self.data.P + 1) / 2,
                                    (self.data.P + 1) / 2,
                                ),
                            )
                            state = {
                                "h": h,
                                "c": c,
                                "w": w.reshape(-1, 1),
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
            lambda: torch.ones((self.Q, self.S + 1), device=device),
            constraint=constraints.simplex,
        )
        pyro.param(
            "pi_size",
            lambda: torch.full((self.Q, 1), 2, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "m_probs",
            lambda: torch.full((self.K, data.Nt, data.F, self.Q), 0.5, device=device),
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
            lambda: torch.full((self.Q,), 0.5, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "lamda_beta",
            lambda: torch.full((self.Q,), 100, device=device),
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
            lambda: (data.median.to(device) - data.offset.mean).expand(
                data.Nt, 1, data.C
            ),
            constraint=constraints.positive,
        )
        pyro.param(
            "background_std_loc",
            lambda: torch.ones(data.Nt, 1, data.C, device=device),
            constraint=constraints.positive,
        )

        pyro.param(
            "b_loc",
            lambda: (data.median.to(device) - self.data.offset.mean).expand(
                data.Nt, data.F, data.C
            ),
            constraint=constraints.positive,
        )
        pyro.param(
            "b_beta",
            lambda: torch.ones(data.Nt, data.F, data.C, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "h_loc",
            lambda: torch.full((self.K, data.Nt, data.F, self.Q), 2000, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "h_beta",
            lambda: torch.full((self.K, data.Nt, data.F, self.Q), 0.001, device=device),
            constraint=constraints.positive,
        )
        pyro.param(
            "w_size",
            lambda: torch.full((self.K, data.Nt, data.F, self.Q), 100, device=device),
            constraint=constraints.greater_than(2.0),
        )
        pyro.param(
            "size",
            lambda: torch.full((self.K, data.Nt, data.F, self.Q), 200, device=device),
            constraint=constraints.greater_than(2.0),
        )
