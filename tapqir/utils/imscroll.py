from functools import singledispatch

import numpy as np
import pandas as pd
import torch
from pyro.ops.stats import pi, resample


@singledispatch
def count_intervals(labels):
    r"""
    Count binding interval data.

    Co-localization and absent intervals are coded as -3
    and -2 respectively when they are the first (or only) interval in a record, 3 and 2
    when they are the last interval in a record and 1 and 0 elsewhere.

    Reference::

      @article{friedman2015multi,
        title={Multi-wavelength single-molecule fluorescence analysis of transcription mechanisms},
        author={Friedman, Larry J and Gelles, Jeff},
        journal={Methods},
        volume={86},
        pages={27--36},
        year={2015},
        publisher={Elsevier}
      }
    """
    raise NotImplementedError


@count_intervals.register(np.ndarray)
def _(labels):
    start_condition = (
        np.concatenate((~labels[:, 0:1], labels[:, :-1]), axis=1) != labels
    )
    start_aoi, start_frame = np.nonzero(start_condition)
    start_type = labels.astype("long")
    start_type[:, 0] = -start_type[:, 0] - 2
    start_type = start_type[start_aoi, start_frame]

    stop_condition = np.concatenate(
        (labels[:, :-1] != labels[:, 1:], np.ones_like(labels[:, 0:1])), axis=1
    )
    stop_aoi, stop_frame = np.nonzero(stop_condition)
    stop_type = labels.astype("long")
    stop_type[:, -1] += 2
    stop_type = stop_type[stop_aoi, stop_frame]

    assert all(start_aoi == stop_aoi)

    low_or_high = np.where(abs(start_type) > abs(stop_type), start_type, stop_type)

    result = pd.DataFrame(
        data={
            "aoi": start_aoi,
            "start_frame": start_frame,
            "stop_frame": stop_frame,
            "dwell_time": stop_frame + 1 - start_frame,
            "low_or_high": low_or_high,
        }
    )
    return result


@count_intervals.register(torch.Tensor)
def _(labels):
    start_condition = torch.cat((~labels[:, 0:1], labels[:, :-1]), dim=1) != labels
    start_aoi, start_frame = torch.nonzero(start_condition, as_tuple=True)
    start_type = labels.long()
    start_type[:, 0] = -start_type[:, 0] - 2
    start_type = start_type[start_aoi, start_frame]

    stop_condition = torch.cat(
        (labels[:, :-1] != labels[:, 1:], torch.ones_like(labels[:, 0:1])), dim=1
    )
    stop_aoi, stop_frame = torch.nonzero(stop_condition, as_tuple=True)
    stop_type = labels.long()
    stop_type[:, -1] += 2
    stop_type = stop_type[stop_aoi, stop_frame]

    assert all(start_aoi == stop_aoi)

    low_or_high = torch.where(abs(start_type) > abs(stop_type), start_type, stop_type)

    result = pd.DataFrame(
        data={
            "aoi": start_aoi,
            "start_frame": start_frame,
            "stop_frame": stop_frame,
            "dwell_time": stop_frame + 1 - start_frame,
            "low_or_high": low_or_high,
        }
    )
    return result


@singledispatch
def time_to_first_binding(labels):
    r"""
    Measure the time elapsed prior to the first binding.

    Time-to-first binding for a binary data:

        :math:`\mathrm{ttfb} =
        \sum_{f=1}^{F-1} f z_{n,f} \prod_{f^\prime=0}^{f-1} (1 - z_{n,f^\prime})
        + F \prod_{f^\prime=0}^{F-1} (1 - z_{n,f^\prime})`

    Expected value of the time-to-first binding:

        :math:`\mathbb{E}[\mathrm{ttfb}] =
        \sum_{f=1}^{F-1} f q(z_{n,f}=1) \prod_{f^\prime=f-1}^{f-1} q(z_{n,f^\prime}=0)
        + F \prod_{f^\prime=0}^{F-1} q(z_{n,f^\prime}=0)`

    Reference::

      @article{friedman2015multi,
        title={Multi-wavelength single-molecule fluorescence analysis of transcription mechanisms},
        author={Friedman, Larry J and Gelles, Jeff},
        journal={Methods},
        volume={86},
        pages={27--36},
        year={2015},
        publisher={Elsevier}
      }
    """
    raise NotImplementedError


@time_to_first_binding.register(np.ndarray)
def _(labels):
    labels = labels.astype("float")
    N, F = labels.shape
    frames = np.arange(1, F + 1)
    q1 = np.ones((N, F))
    q1[:, :-1] = labels[:, 1:]
    cumq0 = np.cumprod(1 - labels, axis=-1)
    ttfb = (frames * q1 * cumq0).sum(-1)
    return ttfb


@time_to_first_binding.register(torch.Tensor)
def _(labels):
    labels = labels.float()
    N, F = labels.shape
    frames = torch.arange(1, F + 1)
    q1 = torch.ones(N, F)
    q1[:, :-1] = labels[:, 1:]
    cumq0 = torch.cumprod(1 - labels, dim=-1)
    ttfb = (frames * q1 * cumq0).sum(-1)
    return ttfb


@singledispatch
def association_rate(labels):
    r"""
    Compute the on-rate from the binary data assuming a two-state HMM model.
    """
    raise NotImplementedError


@association_rate.register(np.ndarray)
def _(labels):
    binding_events = ((1 - labels[..., :-1]) * labels[..., 1:]).sum()
    off_states = (1 - labels[..., :-1]).sum()
    kon = binding_events / off_states
    return kon


@association_rate.register(torch.Tensor)
def _(labels):
    labels = labels.float()
    binding_events = ((1 - labels[..., :-1]) * labels[..., 1:]).sum()
    off_states = (1 - labels[..., :-1]).sum()
    kon = binding_events / off_states
    return kon


@singledispatch
def dissociation_rate(labels):
    r"""
    Compute the off-rate from the binary data assuming a two-state HMM model.
    """
    raise NotImplementedError


@dissociation_rate.register(np.ndarray)
def _(labels):
    dissociation_events = (labels[..., :-1] * (1 - labels[..., 1:])).sum()
    on_states = labels[..., :-1].sum()
    koff = dissociation_events / on_states
    return koff


@dissociation_rate.register(torch.Tensor)
def _(labels):
    labels = labels.float()
    dissociation_events = (labels[..., :-1] * (1 - labels[..., 1:])).sum()
    on_states = labels[..., :-1].sum()
    koff = dissociation_events / on_states
    return koff


@singledispatch
def bootstrap(samples, estimator, repetitions=1000, probs=0.68):
    raise NotImplementedError


@bootstrap.register(np.ndarray)
def _(samples, estimator, repetitions=1000, probs=0.68):
    estimand = np.zeros((repetitions,))
    for i in range(repetitions):
        bootstrap_values = np.random.choice(samples, size=len(samples), replace=True)
        estimand[i] = estimator(bootstrap_values)
    return np.quantile(estimand, (1 - probs) / 2), np.quantile(
        estimand, (1 + probs) / 2
    )


@bootstrap.register(torch.Tensor)
def _(samples, estimator, repetitions=1000, probs=0.68):
    estimand = torch.zeros(repetitions)
    for i in range(repetitions):
        bootstrap_values = resample(samples, num_samples=len(samples), replacement=True)
        estimand[i] = estimator(bootstrap_values)
    return pi(estimand, probs)
