import numpy as np
import pytest
import torch

from tapqir.utils.imscroll import count_intervals


@pytest.mark.parametrize(
    "labels,expected",
    [
        (
            np.array([[False, False, True], [True, False, True]]),
            np.array([[0, 2, -2], [0, 1, 3], [1, 1, -3], [1, 1, 0], [1, 1, 3]]),
        ),
        (
            np.array([[False, True, False], [True, True, False]]),
            np.array([[0, 1, -2], [0, 1, 1], [0, 1, 2], [1, 2, -3], [1, 1, 2]]),
        ),
        (
            torch.tensor([[False, False, True], [True, False, True]]),
            np.array([[0, 2, -2], [0, 1, 3], [1, 1, -3], [1, 1, 0], [1, 1, 3]]),
        ),
        (
            torch.tensor([[False, True, False], [True, True, False]]),
            np.array([[0, 1, -2], [0, 1, 1], [0, 1, 2], [1, 2, -3], [1, 1, 2]]),
        ),
    ],
)
def test_count_intervals(labels, expected):
    result = count_intervals(labels)
    actual = result[["aoi", "dwell_time", "low_or_high"]].values
    assert (actual == expected).all()
