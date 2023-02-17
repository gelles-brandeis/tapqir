# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

import math

import torch
import torch.distributions.constraints as constraints
import torch.nn as nn
from pyro.nn import PyroModule, PyroParam, pyro_method
from pyro.ops.indexing import Vindex
from pyroapi import distributions as dist
from pyroapi import handlers, pyro
from torch.distributions import transforms

from tapqir.distributions import AffineBeta
from tapqir.models.cosmos import cosmos
from tapqir.utils.stats import torch_to_scipy_dist


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
        for i, o in sizes[0:-1]:
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
    def __init__(self, input_size, output_size, h_sizes, non_linear_layer):
        super().__init__()
        self.mlp = MLP(input_size, h_sizes + [output_size], non_linear_layer)

    def forward(self, h):
        out = self.mlp(h)
        return out


class cosmosvae(cosmos, PyroModule):
    r"""
    *EXPERIMENTAL*

    **Amortized Multi-Color Time-Independent Colocalization Model**

    :param K: Maximum number of spots that can be present in a single image.
    :param device: Computation device (cpu or gpu).
    :param dtype: Floating point precision.
    :param use_pykeops: Use pykeops as backend to marginalize out offset.
    :param priors: Dictionary of parameters of prior distributions.
    """

    name = "cosmosvae"

    def __init__(
        self,
        S: int = 1,
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

        # nn config
        nl = nn.ReLU
        predict_net = [128, 128]
        image_size = self.data.P**2

        # background
        self.predict_b = PyroModule[Predict](image_size, 1, predict_net, nl)

        # x and y
        rnn_input_size = image_size + 4
        rnn_hidden_size = 256
        self.rnn = PyroModule[nn.LSTMCell](rnn_input_size, rnn_hidden_size)
        self.predict_xy = PyroModule[Predict](rnn_hidden_size, 2, predict_net, nl)

        # rnn parameters
        self.h_init = PyroParam(torch.zeros(1, rnn_hidden_size))
        self.c_init = PyroParam(torch.zeros(1, rnn_hidden_size))
        self.height_init = PyroParam(torch.zeros(1, 1))
        self.width_init = PyroParam(torch.zeros(1, 1))
        self.x_init = PyroParam(torch.zeros(1, 1))
        self.y_init = PyroParam(torch.zeros(1, 1))

    @pyro_method
    def guide(self):
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
            mask = Vindex(self.data.mask.to(self.device))[ndx]
            with handlers.mask(mask=mask):
                background_mean = pyro.sample(
                    "background_mean",
                    dist.Delta(Vindex(pyro.param("background_mean_loc"))[ndx, 0, cdx]),
                )
                pyro.sample(
                    "background_std",
                    dist.Delta(Vindex(pyro.param("background_std_loc"))[ndx, 0, cdx]),
                )
                with frames as fdx:
                    fdx = fdx[:, None]
                    # fetch data
                    obs, target_locs, __ = self.data.fetch(ndx, fdx, cdx)
                    # sample background intensity
                    b_loc = self.get_b_loc(obs, background_mean)
                    background = pyro.sample(
                        "background",
                        dist.Gamma(
                            b_loc * Vindex(pyro.param("b_beta"))[ndx, fdx, cdx],
                            Vindex(pyro.param("b_beta"))[ndx, fdx, cdx],
                        ),
                    )

                    # initial rnn input
                    n = torch.numel(background)
                    state = {
                        "h": self.h_init.expand(n, -1),
                        "c": self.c_init.expand(n, -1),
                        "height": self.height_init.expand(n, -1),
                        "width": self.width_init.expand(n, -1),
                        "x": self.x_init.expand(n, -1),
                        "y": self.y_init.expand(n, -1),
                    }

                    for kdx in spots:
                        # sample spot presence m
                        x_mean, y_mean, state = self.get_xy(obs, b_loc, state)
                        m = pyro.sample(
                            f"m_k{kdx}",
                            dist.Bernoulli(
                                # m_probs[..., kdx]
                                Vindex(pyro.param("m_probs"))[kdx, ndx, fdx, cdx]
                            ),
                            infer={"enumerate": "parallel"},
                        )
                        with handlers.mask(mask=m > 0):
                            # sample spot variables
                            x = pyro.sample(
                                f"x_k{kdx}",
                                AffineBeta(
                                    x_mean,
                                    Vindex(pyro.param("size"))[kdx, ndx, fdx, cdx],
                                    -(self.data.P + 1) / 2,
                                    (self.data.P + 1) / 2,
                                ),
                            )
                            y = pyro.sample(
                                f"y_k{kdx}",
                                AffineBeta(
                                    y_mean,
                                    Vindex(pyro.param("size"))[kdx, ndx, fdx, cdx],
                                    -(self.data.P + 1) / 2,
                                    (self.data.P + 1) / 2,
                                ),
                            )
                            width = pyro.sample(
                                f"width_k{kdx}",
                                AffineBeta(
                                    Vindex(pyro.param("w_mean"))[kdx, ndx, fdx, cdx],
                                    Vindex(pyro.param("w_size"))[kdx, ndx, fdx, cdx],
                                    self.priors["width_min"],
                                    self.priors["width_max"],
                                ),
                            )
                            height = pyro.sample(
                                f"height_k{kdx}",
                                dist.Gamma(
                                    Vindex(pyro.param("h_loc"))[kdx, ndx, fdx, cdx]
                                    * Vindex(pyro.param("h_beta"))[kdx, ndx, fdx, cdx],
                                    Vindex(pyro.param("h_beta"))[kdx, ndx, fdx, cdx],
                                ),
                            )
                            # update state
                            state["height"] = ((height / b_loc).reshape(-1, 1),)
                            state["width"] = width.reshape(-1, 1)
                            state["x"] = x.reshape(-1, 1)
                            state["y"] = y.reshape(-1, 1)

    def get_background(self, obs, b):
        offset = self.data.offset.mean
        data = (obs - offset - b[..., None, None]) / b[..., None, None]
        shape = data.shape[:-2]
        data = data.reshape(-1, 196)
        out = self.predict_b(data)

        eps = torch.finfo(out.dtype).eps
        b_loc = torch.maximum(out[:, 0] + 1, torch.tensor(eps))
        b_loc = b_loc.reshape(shape)
        b_loc = b * b_loc
        return b_loc

    def get_xy(self, obs, b_loc, state):
        offset = self.data.offset.mean
        data = (obs - offset - b_loc[..., None, None]) / b_loc[..., None, None]
        shape = data.shape[:-2]
        data = data.reshape(-1, self.data.P**2)

        # rnn
        rnn_input = torch.cat(
            (
                data,
                state["height"],
                state["width"],
                state["x"],
                state["y"],
            ),
            -1,
        )
        h, c = self.rnn(rnn_input, (state["h"], state["c"]))

        # predict
        xy = self.predict_xy(h)
        eps = torch.finfo(xy.dtype).eps

        loc = -(14 + 1) / 2 + 3 * eps
        scale = (14 + 1) - 6 * eps
        x_mean = transforms.ComposeTransform(
            [transforms.SigmoidTransform(), transforms.AffineTransform(loc, scale)]
        )(xy[:, 0])
        y_mean = transforms.ComposeTransform(
            [transforms.SigmoidTransform(), transforms.AffineTransform(loc, scale)]
        )(xy[:, 1])
        x_mean = x_mean.reshape(shape)
        y_mean = y_mean.reshape(shape)
        state["h"] = h
        state["c"] = c
        return x_mean, y_mean, state

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
            lambda: (
                data.images.double().mean((-2, -1)).mean(-2, keepdim=True).to(device)
                - data.offset.mean
            ),
            constraint=constraints.positive,
        )
        pyro.param(
            "background_std_loc",
            lambda: torch.ones(data.Nt, 1, data.C, device=device),
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
            "w_mean",
            lambda: torch.full((self.K, data.Nt, data.F, self.Q), 1.5, device=device),
            constraint=constraints.interval(
                0.75 + torch.finfo(self.dtype).eps,
                2.25 - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            "w_size",
            lambda: torch.full((self.K, data.Nt, data.F, self.Q), 100, device=device),
            constraint=constraints.greater_than(2.0),
        )
        #  pyro.param(
        #      "x_mean",
        #      lambda: torch.zeros(self.K, data.Nt, data.F, self.Q, device=device),
        #      constraint=constraints.interval(
        #          -(data.P + 1) / 2 + torch.finfo(self.dtype).eps,
        #          (data.P + 1) / 2 - torch.finfo(self.dtype).eps,
        #      ),
        #  )
        #  pyro.param(
        #      "y_mean",
        #      lambda: torch.zeros(self.K, data.Nt, data.F, self.Q, device=device),
        #      constraint=constraints.interval(
        #          -(data.P + 1) / 2 + torch.finfo(self.dtype).eps,
        #          (data.P + 1) / 2 - torch.finfo(self.dtype).eps,
        #      ),
        #  )
        pyro.param(
            "size",
            lambda: torch.full((self.K, data.Nt, data.F, self.Q), 200, device=device),
            constraint=constraints.greater_than(2.0),
        )

    @torch.no_grad()
    def compute_params(self, CI):
        obs = self.data.images.cuda()
        b_locs, x_means, y_means = [], [], []
        for ndx in torch.split(torch.arange(len(obs)), 200):
            # background
            b_loc = self.get_background(
                obs[ndx], pyro.param("background_mean_loc")[ndx]
            )
            b_locs.append(b_loc)
            # xy
            n = torch.numel(b_loc)
            state = {
                "h": self.h_init.expand(n, -1),
                "c": self.c_init.expand(n, -1),
                "height": self.height_init.expand(n, -1),
                "width": self.width_init.expand(n, -1),
                "x": self.x_init.expand(n, -1),
                "y": self.y_init.expand(n, -1),
            }

            x_means_k, y_means_k = [], []
            for kdx in range(self.K):
                x_mean, y_mean, state = self.get_xy(obs[ndx], b_loc, state)
                # update state
                state["height"] = (pyro.param("h_loc")[kdx, ndx] / b_loc).reshape(-1, 1)
                state["width"] = pyro.param("w_mean")[kdx, ndx].reshape(-1, 1)
                state["x"] = pyro.param("x_mean")[kdx, ndx].reshape(-1, 1)
                state["y"] = pyro.param("y_mean")[kdx, ndx].reshape(-1, 1)
                x_means_k.append(x_mean)
                y_means_k.append(y_mean)
            x_means.append(torch.stack(x_means_k, 0))
            y_means.append(torch.stack(y_means_k, 0))
        b_loc = torch.cat(b_locs, 0)
        x_mean = torch.cat(x_means, 1)
        y_mean = torch.cat(y_means, 1)
        params = {}
        for param in self.ci_params:
            if param == "gain":
                fn = dist.Gamma(
                    pyro.param("gain_loc") * pyro.param("gain_beta"),
                    pyro.param("gain_beta"),
                )
            elif param == "alpha":
                fn = dist.Dirichlet(pyro.param("alpha_mean") * pyro.param("alpha_size"))
            elif param == "pi":
                fn = dist.Dirichlet(pyro.param("pi_mean") * pyro.param("pi_size"))
            elif param == "init":
                fn = dist.Dirichlet(pyro.param("init_mean") * pyro.param("init_size"))
            elif param == "trans":
                fn = dist.Dirichlet(pyro.param("trans_mean") * pyro.param("trans_size"))
            elif param == "lamda":
                fn = dist.Gamma(
                    pyro.param("lamda_loc") * pyro.param("lamda_beta"),
                    pyro.param("lamda_beta"),
                )
            elif param == "proximity":
                fn = AffineBeta(
                    pyro.param("proximity_loc"),
                    pyro.param("proximity_size"),
                    0,
                    (self.data.P + 1) / math.sqrt(12),
                )
            elif param == "background":
                fn = dist.Gamma(b_loc * pyro.param("b_beta"), pyro.param("b_beta"))
            elif param == "height":
                fn = dist.Gamma(
                    pyro.param("h_loc") * pyro.param("h_beta"),
                    pyro.param("h_beta"),
                )
            elif param == "width":
                fn = AffineBeta(
                    pyro.param("w_mean"),
                    pyro.param("w_size"),
                    self.priors["width_min"],
                    self.priors["width_max"],
                )
            elif param == "x":
                fn = AffineBeta(
                    x_mean,
                    pyro.param("size"),
                    -(self.data.P + 1) / 2,
                    (self.data.P + 1) / 2,
                )
            elif param == "y":
                fn = AffineBeta(
                    y_mean,
                    pyro.param("size"),
                    -(self.data.P + 1) / 2,
                    (self.data.P + 1) / 2,
                )
            scipy_dist = torch_to_scipy_dist(fn)
            LL, UL = scipy_dist.interval(alpha=CI)
            params[param] = {}
            params[param]["LL"] = torch.as_tensor(LL, device=torch.device("cpu"))
            params[param]["UL"] = torch.as_tensor(UL, device=torch.device("cpu"))
            params[param]["Mean"] = fn.mean.detach().cpu()

        params["m_probs"] = self.m_probs.cpu()
        params["z_probs"] = self.z_probs.cpu()
        params["theta_probs"] = self.theta_probs.cpu()
        params["z_map"] = self.z_map.data.cpu()
        params["p_specific"] = params["theta_probs"].sum(0)

        return params
