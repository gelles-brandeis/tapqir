# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import contextlib
from functools import reduce

import funsor
from pyro.contrib.funsor import to_data
from pyro.contrib.funsor.handlers import enum, plate, replay
from pyro.contrib.funsor.infer.elbo import ELBO
from pyro.contrib.funsor.infer.traceenum_elbo import terms_from_trace

from tapqir.handlers import trace

from .sum_product import compute_expectations


class TraceMarkovEnum_ELBO(ELBO):
    def differentiable_loss(self, model, guide, *args, **kwargs):

        # get batched, enumerated, to_funsor-ed traces from the guide and model
        with plate(
            size=self.num_particles
        ) if self.num_particles > 1 else contextlib.ExitStack(), enum(
            first_available_dim=(-self.max_plate_nesting - 1)
            if self.max_plate_nesting
            else None
        ):
            guide_tr = trace()(guide).get_trace(*args, **kwargs)
            model_tr = trace()(replay(model, trace=guide_tr)).get_trace(*args, **kwargs)

        # extract from traces all metadata that we will need to compute the elbo
        guide_terms = terms_from_trace(guide_tr)
        model_terms = terms_from_trace(model_tr)

        # build up a lazy expression for the elbo
        with funsor.terms.eager:
            # identify and contract out auxiliary variables in the model with partial_sum_product
            contracted_factors, uncontracted_factors = [], []
            for f in model_terms["log_factors"]:
                if model_terms["measure_vars"].intersection(f.inputs):
                    contracted_factors.append(f)
                else:
                    uncontracted_factors.append(f)
            # incorporate the effects of subsampling and handlers.scale through a common scale factor
            markov_dims = frozenset(
                {plate for plate, step in model_terms["plate_to_step"].items() if step}
            )
            contracted_costs = [
                model_terms["scale"] * f
                for f in funsor.sum_product.modified_partial_sum_product(
                    funsor.ops.logaddexp,
                    funsor.ops.add,
                    model_terms["log_measures"] + contracted_factors,
                    plate_to_step=model_terms["plate_to_step"],
                    eliminate=model_terms["measure_vars"] | markov_dims,
                )
            ]

            costs = contracted_costs + uncontracted_factors  # model costs: logp
            costs += [-f for f in guide_terms["log_factors"]]  # guide costs: -logq

            # finally, integrate out guide variables in the elbo and all plates
            if guide_terms["log_measures"]:
                markov_dims = frozenset(
                    {
                        plate
                        for plate, step in guide_terms["plate_to_step"].items()
                        if step
                    }
                )
                elbo_terms = compute_expectations(
                    guide_terms["log_measures"],
                    costs,
                    plate_to_step=guide_terms["plate_to_step"],
                    eliminate=(
                        guide_terms["plate_vars"]
                        | guide_terms["measure_vars"]
                        | markov_dims
                    ),
                )
            else:
                elbo_terms = costs
            elbo_terms = [term.reduce(funsor.ops.add) for term in elbo_terms]
            elbo = reduce(funsor.ops.add, elbo_terms)

        return -to_data(elbo)
