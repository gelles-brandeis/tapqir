# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

from pyro.contrib.autoname import scope
from pyro.ops.indexing import Vindex
from pyroapi import distributions as dist
from pyroapi import handlers, pyro

from tapqir.distributions import AffineBeta, FixedOffsetGamma
from tapqir.models import Cosmos


class FixedOffset(Cosmos):
    r"""
    for :math:`n=1` to :math:`N`:

        for :math:`n=1` to :math:`F`:

            :math:`b_{nf} \sim  \text{Gamma}(b_{nf}|\mu^b_n, \beta^b_n)`

            :math:`b_{nf} \sim  \text{Gamma}(b_{nf}|\mu^b_n, \beta^b_n)`
    """
    name = "fixedoffset"

    @handlers.block(hide=["width_mean", "width_size"])
    def model(self):
        # initialize model parameters
        # self.model_parameters()

        # test data
        with scope(prefix="d"):
            self.spot_model(self.data, self.data_loc, prefix="d")

        # control data
        if self.control:
            with scope(prefix="c"):
                self.spot_model(self.control, self.control_loc, prefix="c")

    def spot_model(self, data, data_loc, prefix):
        # target sites
        N_plate = pyro.plate("N_pyro.plate", data.N, dim=-2)
        # time frames
        F_plate = pyro.plate("F_pyro.plate", data.F, dim=-1)

        with N_plate as ndx, F_plate:
            # sample background intensity
            background = pyro.sample("background", dist.Gamma(150.0, 1.0))
            locs = background[..., None, None]

            # sample hidden model state (1+K*S,)
            if data.dtype == "test":
                theta = pyro.sample("theta", dist.Categorical(self.probs_theta))
            else:
                theta = 0

            for kdx in range(self.K):
                ontarget = Vindex(self.ontarget)[theta, kdx]
                # spot presence
                m = pyro.sample(
                    f"m_{kdx}", dist.Categorical(Vindex(self.probs_m)[theta, kdx])
                )
                with handlers.mask(mask=m > 0):
                    # sample spot variables
                    height = pyro.sample(
                        f"height_{kdx}",
                        dist.Gamma(
                            pyro.param("height_loc") / pyro.param("gain"),
                            1 / pyro.param("gain"),
                        ),
                    )
                    width = pyro.sample(
                        f"width_{kdx}",
                        AffineBeta(
                            pyro.param("width_mean"),
                            pyro.param("width_size"),
                            0.75,
                            2.25,
                        ),
                    )
                    x = pyro.sample(
                        f"x_{kdx}",
                        AffineBeta(
                            0, self.size[ontarget], -(data.D + 1) / 2, (data.D + 1) / 2
                        ),
                    )
                    y = pyro.sample(
                        f"y_{kdx}",
                        AffineBeta(
                            0, self.size[ontarget], -(data.D + 1) / 2, (data.D + 1) / 2
                        ),
                    )

                # calculate image shape w/o offset
                height = height.masked_fill(m == 0, 0.0)
                gaussian = data_loc(height, width, x, y, ndx)
                locs = locs + gaussian

            # observed data
            pyro.sample(
                "data",
                FixedOffsetGamma(
                    locs, pyro.param("gain"), pyro.param("offset")
                ).to_event(2),
            )
