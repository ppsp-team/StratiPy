import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))
import pytest

from stratipy import filtering_diffusion
import scipy.sparse as sp
import numpy as np


def sparse_random_matrix(n, d, unify_max_val):
    M = sp.rand(n, n, density=d)
    if unify_max_val:
        M.data[:] = 1
    return M


def test_propagation():
    n = 100
    M = sparse_random_matrix(n, 0.2, unify_max_val=True)
    adj = np.random.randint(0,2,size=(n,n))
    adj = sp.csr_matrix((adj + adj.T)/2)
    # assert for array equality before/after propagation with alpha = 0
    M1 = M.todense()
    M2 = (filtering_diffusion.propagation(M, adj, alpha=0, tol=10e-6)).todense()
    np.testing.assert_array_equal(M1, M2)


def test_compare_ij_ji():
    n=100
    ppi = sparse_random_matrix(n, 0.5, unify_max_val=False)
    ppi_min, ppi_max = filtering_diffusion.compare_ij_ji(ppi, out_min=True, out_max=True)
    ppi_max.todense()[ppi_max.todense()==0] = 1
    # all values of ppi_max should be greater than or equal to ppi_min's values
    comp_min_max = (ppi_max.todense() >= ppi_min.todense())
    all_true_matrix = np.ones((n, n), dtype=bool)
    np.testing.assert_array_equal(comp_min_max, all_true_matrix)
