from functools import singledispatch

import numpy as np
import pandas as pd
import torch


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
