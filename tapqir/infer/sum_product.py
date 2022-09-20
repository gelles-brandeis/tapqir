# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict, defaultdict
from functools import reduce

import funsor
import funsor.ops as ops
from funsor.cnf import Contraction
from funsor.domains import Bint
from funsor.ops import AssociativeOp
from funsor.sum_product import _partition
from funsor.tensor import Tensor
from funsor.terms import Cat, Funsor, Number, Slice, Variable


def _scatter(src, res, subs):
    # inverse of advanced indexing
    # TODO check types of subs, in case some logic from eager_subs was accidentally left out?

    # use advanced indexing logic copied from Tensor.eager_subs:

    # materialize after checking for renaming case
    subs = OrderedDict((k, res.materialize(v)) for k, v in subs)

    # Compute result shapes.
    inputs = OrderedDict()
    for k, domain in res.inputs.items():
        inputs[k] = domain

    # Construct a dict with each input's positional dim,
    # counting from the right so as to support broadcasting.
    total_size = len(inputs) + len(res.output.shape)  # Assumes only scalar indices.
    new_dims = {}
    for k, domain in inputs.items():
        assert not domain.shape
        new_dims[k] = len(new_dims) - total_size

    # Use advanced indexing to construct a simultaneous substitution.
    index = []
    for k, domain in res.inputs.items():
        if k in subs:
            v = subs.get(k)
            if isinstance(v, Number):
                index.append(int(v.data))
            else:
                # Permute and expand v.data to end up at new_dims.
                assert isinstance(v, Tensor)
                v = v.align(tuple(k2 for k2 in inputs if k2 in v.inputs))
                assert isinstance(v, Tensor)
                v_shape = [1] * total_size
                for k2, size in zip(v.inputs, v.data.shape):
                    v_shape[new_dims[k2]] = size
                index.append(v.data.reshape(tuple(v_shape)))
        else:
            # Construct a [:] slice for this preserved input.
            offset_from_right = -1 - new_dims[k]
            index.append(
                ops.new_arange(res.data, domain.dtype).reshape(
                    (-1,) + (1,) * offset_from_right
                )
            )

    # Construct a [:] slice for the output.
    for i, size in enumerate(res.output.shape):
        offset_from_right = len(res.output.shape) - i - 1
        index.append(
            ops.new_arange(res.data, size).reshape((-1,) + (1,) * offset_from_right)
        )

    # the only difference from Tensor.eager_subs is here:
    # instead of indexing the rhs (lhs = rhs[index]), we index the lhs (lhs[index] = rhs)

    # unsqueeze to make broadcasting work
    src_inputs, src_data = src.inputs.copy(), src.data
    for k, v in res.inputs.items():
        if k not in src.inputs and isinstance(subs[k], Number):
            src_inputs[k] = Bint[1]
            src_data = src_data.unsqueeze(-1 - len(src.output.shape))
    src = Tensor(src_data, src_inputs, src.output.dtype).align(tuple(res.inputs.keys()))

    data = res.data
    data[tuple(index)] = src.data
    return Tensor(data, inputs, res.dtype)


def _contraction_identity(factor, step):
    """
    Helper function to create a Funsor with the same shape as ``factor``
    and log identity matrices corresponding to each pair of variables in ``step``.
    Contraction of the factor and _contraction_identity returns unchanged factor.
    """
    assert isinstance(factor, Funsor)
    assert isinstance(step, dict)
    inputs = factor.inputs.copy()
    result = Number(0.0)

    for prev, curr in step.items():
        step_inputs = OrderedDict()
        step_inputs[prev] = inputs.pop(prev)
        step_inputs[curr] = inputs.pop(curr)
        step_data = funsor.ops.new_eye(
            funsor.tensor.get_default_prototype(), (step_inputs[prev].size,)
        )
        result += Tensor(step_data.log(), step_inputs, factor.dtype)

    data = funsor.ops.new_zeros(funsor.tensor.get_default_prototype(), ()).expand(
        tuple(v.size for v in inputs.values())
    )
    result += Tensor(data, inputs, factor.dtype)

    return result


def _left_pad_right_crop(trans, time, step):
    """
    Helper function to pad ``trans`` factor with ``_contraction_identity`` of length 1
    from the left and crop the last time point from the right.
    """
    assert isinstance(trans, Funsor)
    assert isinstance(time, str)
    assert isinstance(step, dict)
    duration = trans.inputs[time].size
    pad = _contraction_identity(trans(**{time: Slice(time, 0, 1, 1, duration)}), step)
    trans_cropped_right = trans(**{time: Slice(time, 0, duration - 1, 1, duration)})
    result = Cat(time, (pad, trans_cropped_right))
    return result


def compute_expectations(
    factors, integrands, eliminate=frozenset(), plate_to_step=dict()
):
    """
    Compute expectation of integrand w.r.t. log factors.

    :param factors: List of log density funsors treated as measures.
    :type factors: tuple or list
    :param Funsor integrand: An integrand funsor.
    :param frozenset eliminate: A set of free variables to eliminate,
        including both sum variables and product variable.
    :param dict plate_to_step: A dict mapping markov dimensions to
        ``step`` collections that contain ordered sequences of Markov variable names
        (e.g., ``{"time": frozenset({("x_0", "x_prev", "x_curr")})}``).
        Plates are passed with an empty ``step``.
    :return: Expected value of integrand wrt log density factors.
    :rtype: Funsor
    """
    assert isinstance(factors, (tuple, list))
    assert all(isinstance(f, Funsor) for f in factors)
    assert isinstance(integrands, (tuple, list))
    assert all(isinstance(f, Funsor) for f in integrands)
    assert isinstance(eliminate, frozenset)
    assert isinstance(plate_to_step, dict)
    # process plate_to_step
    plate_to_step = plate_to_step.copy()
    prev_to_init = {}
    for key, step in plate_to_step.items():
        # map prev to init; works for any history > 0
        for chain in step:
            init, prev = chain[: len(chain) // 2], chain[len(chain) // 2 : -1]
            prev_to_init.update(zip(prev, init))
        # convert step to dict type required for MarkovProduct
        plate_to_step[key] = {chain[1]: chain[2] for chain in step}

    plates = frozenset(plate_to_step.keys())
    sum_vars = eliminate - plates
    prod_vars = eliminate.intersection(plates)
    markov_sum_vars = frozenset()
    for step in plate_to_step.values():
        markov_sum_vars |= frozenset(step.keys()) | frozenset(step.values())
    markov_sum_vars &= sum_vars
    markov_prod_vars = frozenset(
        k for k, v in plate_to_step.items() if v and k in eliminate
    )
    markov_sum_to_prod = defaultdict(set)
    for markov_prod in markov_prod_vars:
        for k, v in plate_to_step[markov_prod].items():
            markov_sum_to_prod[k].add(markov_prod)
            markov_sum_to_prod[v].add(markov_prod)

    var_to_ordinal = {}
    ordinal_to_factors = defaultdict(list)
    for f in factors:
        ordinal = plates.intersection(f.inputs)
        ordinal_to_factors[ordinal].append(f)
        for var in sum_vars.intersection(f.inputs):
            var_to_ordinal[var] = var_to_ordinal.get(var, ordinal) & ordinal

    ordinal_to_integrands = defaultdict(list)
    for integrand in integrands:
        ordinal = plates.intersection(integrand.inputs)
        ordinal_to_integrands[ordinal].append(integrand)

    ordinals = frozenset(ordinal_to_factors) | frozenset(ordinal_to_integrands)
    ordinal_to_factors.update(
        {ordinal: ordinal_to_factors.get(ordinal, []) for ordinal in ordinals}
    )

    ordinal_to_vars = defaultdict(set)
    for var, ordinal in var_to_ordinal.items():
        ordinal_to_vars[ordinal].add(var)

    results = []
    while ordinal_to_integrands:
        leaf = max(ordinal_to_integrands, key=len)
        leaf_factors = ordinal_to_factors.pop(leaf)
        leaf_integrands = ordinal_to_integrands.pop(leaf)
        leaf_reduce_vars = ordinal_to_vars[leaf]
        for (group_factors, group_vars) in _partition(
            leaf_factors + leaf_integrands, leaf_reduce_vars | markov_prod_vars
        ):
            # compute the expectation of integrand wrt group_vars
            # eliminate non markov vars
            group_integrands = frozenset(group_factors) & frozenset(leaf_integrands)
            group_factors = frozenset(group_factors) - frozenset(leaf_integrands)
            if group_integrands:
                integrand = reduce(ops.add, group_integrands)
            else:
                continue
            nonmarkov_vars = group_vars - markov_sum_vars - markov_prod_vars
            nonmarkov_factors = [
                f for f in group_factors if not nonmarkov_vars.isdisjoint(f.inputs)
            ]
            markov_factors = [
                f for f in group_factors if not nonmarkov_vars.intersection(f.inputs)
            ]
            if nonmarkov_factors:
                # compute expectation of integrand wrt nonmarkov vars
                log_measure = reduce(ops.add, nonmarkov_factors)
                integrand = funsor.Integrate(log_measure, integrand, nonmarkov_vars)
            # eliminate markov vars
            markov_vars = group_vars.intersection(markov_sum_vars)
            if markov_vars:
                markov_prod_var = [markov_sum_to_prod[var] for var in markov_vars]
                assert all(p == markov_prod_var[0] for p in markov_prod_var)
                if len(markov_prod_var[0]) != 1:
                    raise ValueError("intractable!")
                time = next(iter(markov_prod_var[0]))
                for v in sum_vars.intersection(f.inputs):
                    if time in var_to_ordinal[v] and var_to_ordinal[v] < leaf:
                        raise ValueError("intractable!")
                f = reduce(ops.add, markov_factors)
                time_var = Variable(time, f.inputs[time])
                group_step = {
                    k: v for (k, v) in plate_to_step[time].items() if v in markov_vars
                }
                # calculate forward (alpha) terms
                # FIXME: implement lazy funsor.adjoint._scatter function
                # alphas = naive_forward_terms(ops.logaddexp, ops.add, f, time_var, group_step)
                alphas = forward_terms(ops.logaddexp, ops.add, f, time_var, group_step)
                alphas = _left_pad_right_crop(alphas, time, group_step)
                alphas = alphas(**prev_to_init, **{v: k for k, v in group_step.items()})
                # compute expectation of integrand wrt markov vars
                log_measure = reduce(ops.add, [alphas, f])
                integrand = funsor.Integrate(log_measure, integrand, markov_vars)

            remaining_sum_vars = sum_vars.intersection(integrand.inputs)

            if not remaining_sum_vars:
                results.append(integrand.reduce(ops.add, leaf & prod_vars))
            else:
                new_plates = frozenset().union(
                    *(var_to_ordinal[v] for v in remaining_sum_vars)
                )
                if new_plates == leaf:
                    raise ValueError("intractable!")
                integrand = integrand.reduce(ops.add, leaf - new_plates)
                ordinal_to_integrands[new_plates].append(integrand)

    return results


def forward_terms(sum_op, prod_op, trans, time, step):
    """
    Similar to sequential_sum_product but also saves all
    forward terms
    """
    assert isinstance(sum_op, AssociativeOp)
    assert isinstance(prod_op, AssociativeOp)
    assert isinstance(trans, Funsor)
    assert isinstance(time, Variable)
    assert isinstance(step, dict)
    assert all(isinstance(k, str) for k in step.keys())
    assert all(isinstance(v, str) for v in step.values())
    if time.name in trans.inputs:
        assert time.output == trans.inputs[time.name]

    step = OrderedDict(sorted(step.items()))
    drop = tuple("_drop_{}".format(i) for i in range(len(step)))
    prev_to_drop = dict(zip(step.keys(), drop))
    curr_to_drop = dict(zip(step.values(), drop))
    drop = frozenset(Variable(v, trans.inputs[k]) for k, v in curr_to_drop.items())
    sum_terms = []

    # up sweep
    time, duration = time.name, time.output.size
    while duration > 1:
        even_duration = duration // 2 * 2
        x = trans(**{time: Slice(time, 0, even_duration, 2, duration)}, **curr_to_drop)
        y = trans(**{time: Slice(time, 1, even_duration, 2, duration)}, **prev_to_drop)
        contracted = Contraction(sum_op, prod_op, drop, x, y)

        if duration > even_duration:
            extra = trans(**{time: Slice(time, duration - 1, duration)})
            contracted = Cat(time, (contracted, extra))
        sum_terms.append(trans)
        trans = contracted
        duration = (duration + 1) // 2
    else:
        sum_terms.append(trans)

    # handle the root case
    sum_term = sum_terms.pop()
    left_term = _contraction_identity(sum_term, step)
    # down sweep
    while sum_terms:
        sum_term = sum_terms.pop()
        new_left_term = _contraction_identity(sum_term, step)
        duration = sum_term.inputs[time].size
        even_duration = duration // 2 * 2

        if duration > even_duration:
            slices = ((time, Slice(time, duration - 1, duration)),)
            # left terms
            extra_left_term = left_term(
                **{
                    time: Slice(
                        time,
                        even_duration // 2,
                        even_duration // 2 + 1,
                        1,
                        (duration + 1) // 2,
                    )
                }
            )
            left_term = left_term(
                **{time: Slice(time, 0, even_duration // 2, 1, (duration + 1) // 2)}
            )
            new_left_term = _scatter(extra_left_term, new_left_term, slices)

        # left terms
        left_sum = sum_term(
            **{time: Slice(time, 0, even_duration, 2, duration)}, **prev_to_drop
        )
        left_sum_and_term = Contraction(
            sum_op, prod_op, drop, left_sum, left_term(**curr_to_drop)
        )

        slices = ((time, Slice(time, 0, even_duration, 2, duration)),)
        new_left_term = _scatter(left_term, new_left_term, slices)
        slices = ((time, Slice(time, 1, even_duration, 2, duration)),)
        new_left_term = _scatter(left_sum_and_term, new_left_term, slices)

        left_term = new_left_term
    else:
        alphas = Contraction(
            sum_op, prod_op, drop, left_term(**curr_to_drop), sum_term(**prev_to_drop)
        )
    return alphas
