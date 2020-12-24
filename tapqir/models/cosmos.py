import torch
import torch.distributions.constraints as constraints
from pyro import param, plate, poutine, sample
from pyro.contrib.autoname import scope
from pyro.distributions import Categorical, Gamma, HalfNormal, Poisson
from pyro.ops.indexing import Vindex
from torch.distributions.utils import lazy_property

from tapqir.distributions import AffineBeta, ConvolutedGamma
from tapqir.models import Model


class Cosmos(Model):
    r"""
    for :math:`n=1` to :math:`N`:

        for :math:`n=1` to :math:`F`:

            :math:`b_{nf} \sim  \text{Gamma}(b_{nf}|\mu^b_n, \beta^b_n)`

            :math:`b_{nf} \sim  \text{Gamma}(b_{nf}|\mu^b_n, \beta^b_n)`
    """
    name = "cosmos"

    def __init__(self, S, K):
        super().__init__(S, K)
        self.classify = False

    @lazy_property
    def num_states(self):
        r"""
        Total number of states for the image model given by:

            :math:`2 (1+SK) K`
        """
        return 2 * (1 + self.K * self.S) * self.K

    @property
    def probs_j(self):
        result = torch.zeros(2, self.K + 1, dtype=torch.float)
        result[0, : self.K] = (
            Poisson(param("rate_j")).log_prob(torch.arange(self.K).float()).exp()
        )
        result[0, -1] = 1 - result[0, : self.K].sum()
        result[1, : self.K - 1] = (
            Poisson(param("rate_j")).log_prob(torch.arange(self.K - 1).float()).exp()
        )
        result[1, -2] = 1 - result[0, : self.K - 1].sum()
        return result

    @property
    def probs_m(self):
        # this only works for K=2
        result = torch.zeros(1 + self.K * self.S, self.K, 2, dtype=torch.float)
        result[0, :, 0] = self.probs_j[0, 0] + self.probs_j[0, 1] / 2
        result[0, :, 1] = self.probs_j[0, 2] + self.probs_j[0, 1] / 2
        result[1, 0, 1] = 1
        result[1, 1, 0] = self.probs_j[1, 0]
        result[1, 1, 1] = self.probs_j[1, 1]
        result[2, 0, 0] = self.probs_j[1, 0]
        result[2, 0, 1] = self.probs_j[1, 1]
        result[2, 1, 1] = 1
        return result

    @property
    def probs_theta(self):
        result = torch.zeros(self.K * self.S + 1, dtype=torch.float)
        result[0] = param("probs_z")[0]
        for s in range(self.S):
            for k in range(self.K):
                result[self.K * s + k + 1] = param("probs_z")[s + 1] / self.K
        return result

    @lazy_property
    def theta_to_z(self):
        result = torch.zeros(self.K * self.S + 1, self.K, dtype=torch.long)
        for s in range(self.S):
            result[1 + s * self.K : 1 + (s + 1) * self.K] = torch.eye(self.K) * (s + 1)
        return result

    @lazy_property
    def ontarget(self):
        return torch.clamp(self.theta_to_z, min=0, max=1)

    @property
    def z_probs(self):
        r"""
        Probability of an on-target spot :math:`p(z_{knf})`.
        """
        return param("d/theta_probs").data[..., 1:].permute(2, 0, 1)

    @property
    def j_probs(self):
        r"""
        Probability of an off-target spot :math:`p(j_{knf})`.
        """
        return self.m_probs - self.z_probs

    @property
    def m_probs(self):
        r"""
        Probability of a spot :math:`p(m_{knf})`.
        """
        return torch.einsum(
            "sknf,nfs->knf",
            param("d/m_probs").data[..., 1],
            param("d/theta_probs").data,
        )

    @property
    def z_marginal(self):
        return self.z_probs.sum(-3)

    @property
    def z_map(self):
        return self.z_marginal > 0.5

    @property
    def inference_config_model(self):
        if self.classify:
            return {"hide_types": ["param"]}
        return {"hide": ["width_mean", "width_size", "height_scale"]}

    @property
    def inference_config_guide(self):
        if self.classify:
            return {
                "expose": ["d/theta_probs", "d/m_probs"],
                "expose_types": ["sample"],
            }
        return {"expose_types": ["sample", "param"]}

    def model(self):
        with poutine.block(**self.inference_config_model):
            # initialize model parameters
            self.model_parameters()

            # test data
            with scope(prefix="d"):
                self.spot_model(self.data, self.data_loc, prefix="d")

            # control data
            if self.control:
                with scope(prefix="c"):
                    self.spot_model(self.control, self.control_loc, prefix="c")

    def guide(self):
        with poutine.block(**self.inference_config_guide):
            # initialize guide parameters
            self.guide_parameters()

            # test data
            with scope(prefix="d"):
                self.spot_guide(self.data, prefix="d")

            # control data
            if self.control:
                with scope(prefix="c"):
                    self.spot_guide(self.control, prefix="c")

    def spot_model(self, data, data_loc, prefix):
        # target sites
        N_plate = plate("N_plate", data.N, dim=-2)
        # time frames
        F_plate = plate("F_plate", data.F, dim=-1)

        with N_plate as ndx, F_plate:
            # sample background intensity
            background = sample(
                "background",
                Gamma(
                    param(f"{prefix}/background_loc")[ndx]
                    * param(f"{prefix}/background_beta")[ndx],
                    param(f"{prefix}/background_beta")[ndx],
                ),
            )
            locs = background[..., None, None]

            # sample hidden model state (1+K*S,)
            if data.dtype == "test":
                if self.classify:
                    theta = sample("theta", Categorical(self.probs_theta))
                else:
                    theta = sample(
                        "theta",
                        Categorical(self.probs_theta),
                        infer={"enumerate": "parallel"},
                    )
            else:
                theta = 0

            for kdx in range(self.K):
                ontarget = Vindex(self.ontarget)[theta, kdx]
                # spot presence
                m = sample(f"m_{kdx}", Categorical(Vindex(self.probs_m)[theta, kdx]))
                with poutine.mask(mask=m > 0):
                    # sample spot variables
                    height = sample(f"height_{kdx}", HalfNormal(param("height_scale")))
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
                ConvolutedGamma(
                    locs / param("gain"),
                    1 / param("gain"),
                    self.data.offset_samples,
                    self.data.offset_logits,
                ).to_event(2),
                obs=data[ndx],
            )

    def spot_guide(self, data, prefix):
        # target sites
        N_plate = plate(
            "N_plate", data.N, subsample_size=self.batch_size, subsample=self.n, dim=-2
        )
        # time frames
        F_plate = plate("F_plate", data.F, dim=-1)

        with N_plate as ndx, F_plate as fdx:
            if prefix == "d":
                self.batch_idx = ndx.cpu()
            # sample background intensity
            sample(
                "background",
                Gamma(
                    param(f"{prefix}/b_loc")[ndx] * param(f"{prefix}/b_beta")[ndx],
                    param(f"{prefix}/b_beta")[ndx],
                ),
            )

            # sample hidden model state (3,1,1,1)
            if self.classify:
                if data.dtype == "test":
                    theta = sample(
                        "theta",
                        Categorical(param(f"{prefix}/theta_probs")[ndx]),
                        infer={"enumerate": "parallel"},
                    )
                else:
                    theta = 0

            for kdx in range(self.K):
                # spot presence
                if self.classify:
                    m_probs = Vindex(param(f"{prefix}/m_probs"))[
                        theta, kdx, ndx[:, None], fdx
                    ]
                else:
                    m_probs = Vindex(param(f"{prefix}/m_prob"))[kdx, ndx[:, None], fdx]
                m = sample(
                    f"m_{kdx}", Categorical(m_probs), infer={"enumerate": "parallel"}
                )
                with poutine.mask(mask=m > 0):
                    # sample spot variables
                    sample(
                        f"height_{kdx}",
                        Gamma(
                            param(f"{prefix}/h_loc")[kdx, ndx]
                            * param(f"{prefix}/h_beta")[kdx, ndx],
                            param(f"{prefix}/h_beta")[kdx, ndx],
                        ),
                    )
                    sample(
                        f"width_{kdx}",
                        AffineBeta(
                            param(f"{prefix}/w_mean")[kdx, ndx],
                            param(f"{prefix}/w_size")[kdx, ndx],
                            0.75,
                            2.25,
                        ),
                    )
                    sample(
                        f"x_{kdx}",
                        AffineBeta(
                            param(f"{prefix}/x_mean")[kdx, ndx],
                            param(f"{prefix}/size")[kdx, ndx],
                            -(data.D + 1) / 2,
                            (data.D + 1) / 2,
                        ),
                    )
                    sample(
                        f"y_{kdx}",
                        AffineBeta(
                            param(f"{prefix}/y_mean")[kdx, ndx],
                            param(f"{prefix}/size")[kdx, ndx],
                            -(data.D + 1) / 2,
                            (data.D + 1) / 2,
                        ),
                    )

    def model_parameters(self):
        param(
            "proximity", torch.tensor([0.5]), constraint=constraints.interval(0.01, 2.0)
        )
        self.size = torch.cat(
            (
                torch.tensor([2.0]),
                (((self.data.D + 1) / (2 * param("proximity"))) ** 2 - 1),
            ),
            dim=-1,
        )
        param("gain", torch.tensor(5.0), constraint=constraints.positive)
        param("probs_z", torch.ones(self.S + 1), constraint=constraints.simplex)
        param("rate_j", torch.tensor(0.5), constraint=constraints.positive)

        param(
            "d/background_loc",
            torch.ones(self.data.N, 1)
            * (self.data.data_median - self.data.offset_median),
            constraint=constraints.positive,
        )
        param(
            "d/background_beta",
            torch.ones(self.data.N, 1),
            constraint=constraints.positive,
        )

        if self.control:
            param(
                "c/background_loc",
                torch.ones(self.control.N, 1)
                * (self.data.data_median - self.data.offset_median),
                constraint=constraints.positive,
            )
            param(
                "c/background_beta",
                torch.ones(self.control.N, 1),
                constraint=constraints.positive,
            )

        param(
            "width_mean",
            torch.tensor([1.5]),
            constraint=constraints.interval(0.75, 2.25),
        )
        param("width_size", torch.tensor([2.0]), constraint=constraints.positive)
        param("height_scale", torch.tensor(10000.0), constraint=constraints.positive)

    def guide_parameters(self):
        self.spot_parameters(self.data, prefix="d")

        if self.control:
            self.spot_parameters(self.control, prefix="c")

    def spot_parameters(self, data, prefix):
        param(
            f"{prefix}/theta_probs",
            torch.ones(data.N, data.F, 1 + self.K * self.S),
            constraint=constraints.simplex,
        )
        m_probs = torch.ones(1 + self.K * self.S, self.K, data.N, data.F, 2)
        m_probs[1, 0, :, :, 0] = 0
        m_probs[2, 1, :, :, 0] = 0
        param(f"{prefix}/m_probs", m_probs, constraint=constraints.simplex)
        param(
            f"{prefix}/m_prob",
            torch.ones(self.K, data.N, data.F, 2),
            constraint=constraints.simplex,
        )
        param(
            f"{prefix}/b_loc",
            (self.data.data_median - self.data.offset_median).repeat(data.N, data.F),
            constraint=constraints.positive,
        )
        param(
            f"{prefix}/b_beta",
            torch.ones(data.N, data.F),
            constraint=constraints.positive,
        )
        param(
            f"{prefix}/h_loc",
            torch.full((self.K, data.N, data.F), 2000.0),
            # (self.data.noise * 2).repeat(self.K, data.N, data.F),
            constraint=constraints.positive,
        )
        param(
            f"{prefix}/h_beta",
            torch.ones(self.K, data.N, data.F) * 0.001,
            constraint=constraints.positive,
        )
        param(
            f"{prefix}/w_mean",
            torch.ones(self.K, data.N, data.F) * 1.5,
            constraint=constraints.interval(0.75, 2.25),
        )
        param(
            f"{prefix}/w_size",
            torch.ones(self.K, data.N, data.F) * 100.0,
            constraint=constraints.greater_than(2.0),
        )
        param(
            f"{prefix}/x_mean",
            torch.zeros(self.K, data.N, data.F),
            constraint=constraints.interval(-(data.D + 1) / 2, (data.D + 1) / 2),
        )
        param(
            f"{prefix}/y_mean",
            torch.zeros(self.K, data.N, data.F),
            constraint=constraints.interval(-(data.D + 1) / 2, (data.D + 1) / 2),
        )
        size = torch.ones(self.K, data.N, data.F) * 200.0
        if self.K == 2:
            size[1] = 7.0
        elif self.K == 3:
            size[1] = 7.0
            size[2] = 3.0
        param(f"{prefix}/size", size, constraint=constraints.greater_than(2.0))
