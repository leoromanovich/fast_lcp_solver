import numpy as np
import pytest


@pytest.fixture
def M():
    return np.array([
        [4,  2,  0,  3],
        [-1, 4, -3, -6],
        [1, -1,  1,  1],
        [0,  1,  0,  5]
    ])


@pytest.fixture
def q():
    return np.array([-1, 2, -1, -1]).reshape(-1, 1)
