from pyro import param, sample, plate, poutine
from pyro.distributions import Categorical, Gamma
from pyro.ops.indexing import Vindex
from pyro.contrib.autoname import scope

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

    @poutine.block(hide=["width_mean", "width_size"])
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
        N_plate = plate("N_plate", data.N, dim=-2)
        # time frames
        F_plate = plate("F_plate", data.F, dim=-1)

        with N_plate as ndx, F_plate:
            # sample background intensity
            background = sample("background", Gamma(150.0, 1.0))
            locs = background[..., None, None]

            # sample hidden model state (1+K*S,)
            if data.dtype == "test":
                theta = sample("theta", Categorical(self.probs_theta))
            else:
                theta = 0

            for kdx in range(self.K):
                ontarget = Vindex(self.ontarget)[theta, kdx]
                # spot presence
                m = sample(f"m_{kdx}", Categorical(Vindex(self.probs_m)[theta, kdx]))
                with poutine.mask(mask=m > 0):
                    # sample spot variables
                    height = sample(
                        f"height_{kdx}",
                        Gamma(param("height_loc") / param("gain"), 1 / param("gain")),
                    )
                    width = sample(
                        f"width_{kdx}",
                        AffineBeta(
                            param("width_mean"), param("width_size"), 0.75, 2.25
                        ),
                    )
                    x = sample(
                        f"x_{kdx}",
                        AffineBeta(
                            0, self.size[ontarget], -(data.D + 1) / 2, (data.D + 1) / 2
                        ),
                    )
                    y = sample(
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
            sample(
                "data",
                FixedOffsetGamma(locs, param("gain"), param("offset")).to_event(2),
            )
