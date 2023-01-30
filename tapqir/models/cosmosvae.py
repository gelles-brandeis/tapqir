# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

import math

import torch
import torch.distributions.constraints as constraints
import torch.nn as nn
from pyro.ops.indexing import Vindex
from pyroapi import distributions as dist
from pyroapi import handlers, pyro
from torch.distributions import transforms

from tapqir.distributions import AffineBeta
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
    def __init__(self, input_size, output_size, h_sizes, non_linear_layer):
        super().__init__()
        self.mlp = MLP(input_size, h_sizes + [output_size], non_linear_layer)

    def forward(self, h):
        out = self.mlp(h)
        return out


class cosmosnn(cosmos):
    r"""
    *EXPERIMENTAL*

    **Amortized Multi-Color Time-Independent Colocalization Model**

    **Reference**:

    1. Ordabayev YA, Friedman LJ, Gelles J, Theobald DL.
       Bayesian machine learning analysis of single-molecule fluorescence colocalization images.
       eLife. 2022 March. doi: `10.7554/eLife.73860 <https://doi.org/10.7554/eLife.73860>`_.

    :param K: Maximum number of spots that can be present in a single image.
    :param device: Computation device (cpu or gpu).
    :param dtype: Floating point precision.
    :param use_pykeops: Use pykeops as backend to marginalize out offset.
    :param priors: Dictionary of parameters of prior distributions.
    """

    name = "cosmosnn"

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

        # AIR
        non_linearity = "ReLU"
        nl = getattr(nn, non_linearity)
        predict_net = [128, 128]
        input_size = 14 * 14
        self.predict_b = Predict(input_size, 1, predict_net, nl)
        self.predict_spots = Predict(input_size, 10, predict_net, nl)

        # Create parameters.

    def guide(self):
        # register PyTorch module `encoder` with Pyro
        pyro.module("predict_b", self.predict_b)
        pyro.module("predict_spots", self.predict_spots)
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
                    obs, _, __ = self.data.fetch(ndx, fdx, cdx)
                    b_loc = self.get_background(obs, background_mean)
                    # sample background intensity
                    background = pyro.sample(
                        "background",
                        dist.Gamma(
                            b_loc * Vindex(pyro.param("b_beta"))[ndx, fdx, cdx],
                            Vindex(pyro.param("b_beta"))[ndx, fdx, cdx],
                        ),
                    )
                    m_probs, h_loc, w_mean, x_mean, y_mean = self.get_spot_params(
                        obs, background
                    )

                    for kdx in spots:
                        # sample spot presence m
                        m = pyro.sample(
                            f"m_k{kdx}",
                            dist.Bernoulli(
                                m_probs[..., kdx]
                                # Vindex(pyro.param("m_probs"))[kdx, ndx, fdx, cdx]
                            ),
                            infer={"enumerate": "parallel"},
                        )
                        with handlers.mask(mask=m > 0):
                            # sample spot variables
                            height = pyro.sample(
                                f"height_k{kdx}",
                                dist.Gamma(
                                    h_loc[..., kdx]
                                    # Vindex(pyro.param("h_loc"))[kdx, ndx, fdx, cdx]
                                    * Vindex(pyro.param("h_beta"))[kdx, ndx, fdx, cdx],
                                    Vindex(pyro.param("h_beta"))[kdx, ndx, fdx, cdx],
                                ),
                            )
                            width = pyro.sample(
                                f"width_k{kdx}",
                                AffineBeta(
                                    w_mean[..., kdx],
                                    # Vindex(pyro.param("w_mean"))[kdx, ndx, fdx, cdx],
                                    Vindex(pyro.param("w_size"))[kdx, ndx, fdx, cdx],
                                    self.priors["width_min"],
                                    self.priors["width_max"],
                                ),
                            )
                            x = pyro.sample(
                                f"x_k{kdx}",
                                AffineBeta(
                                    x_mean[..., kdx],
                                    # Vindex(pyro.param("x_mean"))[kdx, ndx, fdx, cdx],
                                    Vindex(pyro.param("size"))[kdx, ndx, fdx, cdx],
                                    -(self.data.P + 1) / 2,
                                    (self.data.P + 1) / 2,
                                ),
                            )
                            y = pyro.sample(
                                f"y_k{kdx}",
                                AffineBeta(
                                    y_mean[..., kdx],
                                    # Vindex(pyro.param("y_mean"))[kdx, ndx, fdx, cdx],
                                    Vindex(pyro.param("size"))[kdx, ndx, fdx, cdx],
                                    -(self.data.P + 1) / 2,
                                    (self.data.P + 1) / 2,
                                ),
                            )

    @torch.no_grad()
    def get_background(self, obs, b):
        offset = self.data.offset.mean
        data = (obs - offset - b[..., None, None]) / b[..., None, None]
        shape = data.shape[:-2]
        data = data.reshape(-1, 196)
        out = self.predict_b(data)

        relu = nn.ReLU()
        b_loc = relu(out[:, 0] + 1)
        b_loc = b_loc.reshape(shape)
        b_loc = b * b_loc
        return b_loc

    @torch.no_grad()
    def get_spot_params(self, obs, b):
        offset = self.data.offset.mean
        data = (obs - offset - b[..., None, None]) / b[..., None, None]
        shape = data.shape[:-2]
        data = data.reshape(-1, 196)
        out = self.predict_spots(data)
        eps = torch.finfo(out.dtype).eps

        # spot probs
        m_probs = transforms.SigmoidTransform()(out[:, :2])
        # intensity
        h_loc = transforms.ExpTransform()(out[:, 2:4])
        # width
        w_min = 0.75 + eps
        w_scale = 1.5 - 2 * eps
        w_mean = transforms.ComposeTransform(
            [transforms.SigmoidTransform(), transforms.AffineTransform(w_min, w_scale)]
        )(out[:, 4:6])
        #  w_size = transforms.ComposeTransform(
        #      [transforms.ExpTransform(), transforms.AffineTransform(2, 1)]
        #  )(out[:, 3])
        # position
        loc = -(14 + 1) / 2 + eps
        scale = (14 + 1) - 2 * eps
        x_mean = transforms.ComposeTransform(
            [transforms.SigmoidTransform(), transforms.AffineTransform(loc, scale)]
        )(out[:, 6:8])
        y_mean = transforms.ComposeTransform(
            [transforms.SigmoidTransform(), transforms.AffineTransform(loc, scale)]
        )(out[:, 8:10])
        #  size = transforms.ComposeTransform(
        #      [transforms.ExpTransform(), transforms.AffineTransform(2, 1)]
        #  )(out[:, 6])

        # reshape
        m_probs = m_probs.reshape(shape + (2,))
        h_loc = h_loc.reshape(shape + (2,)) * b[..., None]
        w_mean = w_mean.reshape(shape + (2,))
        x_mean = x_mean.reshape(shape + (2,))
        y_mean = y_mean.reshape(shape + (2,))
        return m_probs, h_loc, w_mean, x_mean, y_mean

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
        #  pyro.param(
        #      "m_probs",
        #      lambda: torch.full((self.K, data.Nt, data.F, self.Q), 0.5, device=device),
        #      constraint=constraints.unit_interval,
        #  )

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
            "b_beta",
            lambda: torch.ones(data.Nt, data.F, data.C, device=device),
            constraint=constraints.positive,
        )
        #  pyro.param(
        #      "h_loc",
        #      lambda: torch.full((self.K, data.Nt, data.F, self.Q), 2000, device=device),
        #      constraint=constraints.positive,
        #  )
        pyro.param(
            "h_beta",
            lambda: torch.full((self.K, data.Nt, data.F, self.Q), 0.001, device=device),
            constraint=constraints.positive,
        )
        #  pyro.param(
        #      "w_mean",
        #      lambda: torch.full((self.K, data.Nt, data.F, self.Q), 1.5, device=device),
        #      constraint=constraints.interval(
        #          0.75 + torch.finfo(self.dtype).eps,
        #          2.25 - torch.finfo(self.dtype).eps,
        #      ),
        #  )
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
