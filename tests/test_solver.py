import numpy as np
from solve_lcp.solver import solveLCP


def test_solver(M, q):
    assert np.allclose(solveLCP(M, q).flatten(), np.array([0, 1, 2, 0]))
