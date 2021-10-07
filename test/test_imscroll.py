# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from tapqir.utils.imscroll import (
    association_rate,
    count_intervals,
    dissociation_rate,
    time_to_first_binding,
)


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


@pytest.mark.parametrize(
    "labels,expected",
    [
        (
            np.array(
                [
                    [False, False, False],
                    [False, False, True],
                    [False, True, True],
                    [True, False, True],
                ]
            ),
            np.array([3.0, 2.0, 1.0, 0.0]),
        ),
        (
            torch.tensor(
                [
                    [False, False, False],
                    [False, False, True],
                    [False, True, True],
                    [True, False, True],
                ]
            ),
            torch.tensor([3.0, 2.0, 1.0, 0.0]),
        ),
    ],
)
def test_time_to_first_binding(labels, expected):
    actual = time_to_first_binding(labels)
    assert (actual == expected).all()


@pytest.mark.parametrize(
    "labels,expected",
    [
        (
            np.array(
                [
                    [False, False, False, True, True],
                    [False, True, True, False, True],
                ]
            ),
            3 / 5,
        ),
        (
            np.array(
                [
                    [True, False, False, False, False],
                    [False, True, True, False, False],
                ]
            ),
            1 / 5,
        ),
    ],
)
def test_association_rate(labels, expected):
    actual = association_rate(labels)
    assert actual == expected


@pytest.mark.parametrize(
    "labels,expected",
    [
        (
            np.array(
                [
                    [False, False, False, True, True],
                    [False, True, True, False, True],
                ]
            ),
            1 / 3,
        ),
        (
            np.array(
                [
                    [True, False, False, False, False],
                    [False, True, True, False, False],
                ]
            ),
            2 / 3,
        ),
    ],
)
def test_dissociation_rate(labels, expected):
    actual = dissociation_rate(labels)
    assert actual == expected
