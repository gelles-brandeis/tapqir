# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

import torch
import torch.distributions.constraints as constraints
from pyroapi import distributions as dist
from pyroapi import infer, pyro

from tapqir.distributions import AffineBeta, KSpotGammaNoise
from tapqir.models.model import Model


class MultiSpot(Model):
    """
    Base Model. Describes images as a superposition of multiple (K) spots.

    Model parameters:

    * gain - camera gain
    * background_mean - background intensity over an AOI
    * background_std - standard deviation of background intensity over an AOI
    * background - background intensity of an image
    * height - integrated spot intensity
    * width - spot width
    * x - spot x-axis center
    * y - spot y-axis center
    * offset - camera offset
    * data - observed images
    """

    name = "multispot"

    def __init__(self, S=1, K=2, device="cpu", dtype="double"):
        super().__init__(S, K, device, dtype)
        self.conv_params = ["-ELBO", "gain_loc"]
        self._global_params = ["gain"]

    def TraceELBO(self, jit=False):
        return (infer.JitTrace_ELBO if jit else infer.Trace_ELBO)(
            ignore_jit_warnings=True
        )

    @property
    def pspecific(self):
        return None

    def model(self):
        # global parameters
        self.gain = pyro.sample("gain", dist.HalfNormal(50)).squeeze()

        # ontarget data
        self.spot_model(self.data.ontarget, prefix="d")

        # offtarget control data
        if self.data.offtarget.images is not None:
            self.spot_model(self.data.offtarget, prefix="c")

    def guide(self):
        # global parameters
        pyro.sample(
            "gain",
            dist.Gamma(
                pyro.param("gain_loc") * pyro.param("gain_beta"),
                pyro.param("gain_beta"),
            ),
        )

        # ontarget data
        self.spot_guide(self.data.ontarget, prefix="d")

        # offtarget control data
        if self.data.offtarget.images is not None:
            self.spot_guide(self.data.offtarget, prefix="c")

    def spot_model(self, data, prefix):
        # spots
        spots = pyro.plate(f"{prefix}/spots", self.K)
        # aoi sites
        aois = pyro.plate(f"{prefix}/aois", data.N, dim=-2)
        # time frames
        frames = pyro.plate(f"{prefix}/frames", data.F, dim=-1)

        with aois as ndx:
            # background mean and std
            background_mean = pyro.sample(
                f"{prefix}/background_mean", dist.HalfNormal(1000)
            )
            background_std = pyro.sample(
                f"{prefix}/background_std", dist.HalfNormal(100)
            )
            with frames:
                # sample background intensity
                background = pyro.sample(
                    f"{prefix}/background",
                    dist.Gamma(
                        (background_mean / background_std) ** 2,
                        background_mean / background_std ** 2,
                    ),
                )

                heights, widths, xs, ys = [], [], [], []
                for kdx in spots:
                    # sample spot variables
                    height = pyro.sample(
                        f"{prefix}/height_{kdx}", dist.HalfNormal(10000)
                    )
                    width = pyro.sample(
                        f"{prefix}/width_{kdx}",
                        AffineBeta(
                            1.5,
                            2,
                            0.75,
                            2.25,
                        ),
                    )
                    x = pyro.sample(
                        f"{prefix}/x_{kdx}",
                        AffineBeta(
                            0,
                            2,
                            -(data.P + 1) / 2,
                            (data.P + 1) / 2,
                        ),
                    )
                    y = pyro.sample(
                        f"{prefix}/y_{kdx}",
                        AffineBeta(
                            0,
                            2,
                            -(data.P + 1) / 2,
                            (data.P + 1) / 2,
                        ),
                    )

                    # append
                    heights.append(height)
                    widths.append(width)
                    xs.append(x)
                    ys.append(y)

                # subtract offset
                odx = pyro.sample(
                    f"{prefix}/offset",
                    dist.Categorical(logits=self.data.offset.logits.to(self.dtype))
                    .expand([data.P, data.P])
                    .to_event(2),
                )
                offset = self.data.offset.samples[odx]
                # fetch data
                obs, target_locs = data.fetch(ndx)
                # observed data
                pyro.sample(
                    f"{prefix}/data",
                    KSpotGammaNoise(
                        torch.stack(heights, -1),
                        torch.stack(widths, -1),
                        torch.stack(xs, -1),
                        torch.stack(ys, -1),
                        target_locs,
                        background,
                        offset,
                        self.gain,
                        data.P,
                    ),
                    obs=obs,
                )

    def spot_guide(self, data, prefix):
        # spots
        spots = pyro.plate(f"{prefix}/spots", self.K)
        # aoi sites
        aois = pyro.plate(
            f"{prefix}/aois",
            data.N,
            subsample_size=self.batch_size,
            subsample=self.n,
            dim=-2,
        )
        # time frames
        frames = pyro.plate(f"{prefix}/frames", data.F, dim=-1)

        with aois as ndx:
            pyro.sample(
                f"{prefix}/background_mean",
                dist.Delta(pyro.param(f"{prefix}/background_mean_loc")[ndx]),
            )
            pyro.sample(
                f"{prefix}/background_std",
                dist.Delta(pyro.param(f"{prefix}/background_std_loc")[ndx]),
            )
            with frames:
                # sample background intensity
                pyro.sample(
                    f"{prefix}/background",
                    dist.Gamma(
                        pyro.param(f"{prefix}/b_loc")[ndx]
                        * pyro.param(f"{prefix}/b_beta")[ndx],
                        pyro.param(f"{prefix}/b_beta")[ndx],
                    ),
                )

                for kdx in spots:
                    # sample spot variables
                    pyro.sample(
                        f"{prefix}/height_{kdx}",
                        dist.Gamma(
                            pyro.param(f"{prefix}/h_loc")[kdx, ndx]
                            * pyro.param(f"{prefix}/h_beta")[kdx, ndx],
                            pyro.param(f"{prefix}/h_beta")[kdx, ndx],
                        ),
                    )
                    pyro.sample(
                        f"{prefix}/width_{kdx}",
                        AffineBeta(
                            pyro.param(f"{prefix}/w_mean")[kdx, ndx],
                            pyro.param(f"{prefix}/w_size")[kdx, ndx],
                            0.75,
                            2.25,
                        ),
                    )
                    pyro.sample(
                        f"{prefix}/x_{kdx}",
                        AffineBeta(
                            pyro.param(f"{prefix}/x_mean")[kdx, ndx],
                            pyro.param(f"{prefix}/size")[kdx, ndx],
                            -(data.P + 1) / 2,
                            (data.P + 1) / 2,
                        ),
                    )
                    pyro.sample(
                        f"{prefix}/y_{kdx}",
                        AffineBeta(
                            pyro.param(f"{prefix}/y_mean")[kdx, ndx],
                            pyro.param(f"{prefix}/size")[kdx, ndx],
                            -(data.P + 1) / 2,
                            (data.P + 1) / 2,
                        ),
                    )

                pyro.sample(
                    f"{prefix}/offset",
                    dist.Categorical(logits=self.data.offset.logits.to(self.dtype))
                    .expand([data.P, data.P])
                    .to_event(2),
                )

    def init_parameters(self):
        pyro.param("gain_loc", lambda: torch.tensor(5), constraint=constraints.positive)
        pyro.param(
            "gain_beta", lambda: torch.tensor(100), constraint=constraints.positive
        )

        self.spot_parameters(self.data.ontarget, prefix="d")

        if self.data.offtarget.images is not None:
            self.spot_parameters(self.data.offtarget, prefix="c")

    def spot_parameters(self, data, prefix):
        pyro.param(
            f"{prefix}/background_mean_loc",
            lambda: torch.full((data.N, 1), data.median - self.data.offset.mean),
            constraint=constraints.positive,
        )
        pyro.param(
            f"{prefix}/background_std_loc",
            lambda: torch.ones(data.N, 1),
            constraint=constraints.positive,
        )

        pyro.param(
            f"{prefix}/b_loc",
            lambda: torch.full((data.N, data.F), data.median - self.data.offset.mean),
            constraint=constraints.positive,
        )
        pyro.param(
            f"{prefix}/b_beta",
            lambda: torch.ones(data.N, data.F),
            constraint=constraints.positive,
        )
        pyro.param(
            f"{prefix}/h_loc",
            lambda: torch.full((self.K, data.N, data.F), 1000),
            constraint=constraints.positive,
        )
        pyro.param(
            f"{prefix}/h_beta",
            lambda: torch.ones(self.K, data.N, data.F),
            constraint=constraints.positive,
        )
        pyro.param(
            f"{prefix}/w_mean",
            lambda: torch.full((self.K, data.N, data.F), 1.5),
            constraint=constraints.interval(
                0.75 + torch.finfo(self.dtype).eps,
                2.25 - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            f"{prefix}/w_size",
            lambda: torch.full((self.K, data.N, data.F), 100.0),
            constraint=constraints.greater_than(2.0),
        )
        pyro.param(
            f"{prefix}/x_mean",
            lambda: torch.zeros(self.K, data.N, data.F),
            constraint=constraints.interval(
                -(data.P + 1) / 2 + torch.finfo(self.dtype).eps,
                (data.P + 1) / 2 - torch.finfo(self.dtype).eps,
            ),
        )
        pyro.param(
            f"{prefix}/y_mean",
            lambda: torch.zeros(self.K, data.N, data.F),
            constraint=constraints.interval(
                -(data.P + 1) / 2 + torch.finfo(self.dtype).eps,
                (data.P + 1) / 2 - torch.finfo(self.dtype).eps,
            ),
        )
        size = torch.full((self.K, data.N, data.F), ((data.P + 1) / (2 * 0.5)) ** 2 - 1)
        if self.K == 2:
            size[1] = 5.0
        elif self.K == 3:
            size[1] = 7.0
            size[2] = 3.0
        pyro.param(
            f"{prefix}/size", lambda: size, constraint=constraints.greater_than(2.0)
        )
