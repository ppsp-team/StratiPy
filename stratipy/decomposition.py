import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp
from sklearn.decomposition import NMF
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot
import warnings
import time

# NOTE some variable names changed:
# sklearnComp -> gene_comp, sklearnStrat -> patient_strat
# sklearnCompDiff -> gene_comp_diff, sklearnStratDiff -> patient_strat_diff
# sklearnCompQDiff -> gene_comp_mean_qn, sklearnStratQDiff -> patient_strat_mean_qn
# sklearnCompQDiffMed -> gene_comp_median_qn, sklearnStratQDiffMed -> patient_strat_median_qn


def nmf_new(mut_final, mut_diff, mut_mean_qn, mut_median_qn, n_components,
            init='nndsvdar', random_state=0):
    # Numerical solver to use: ‘pg’ is a Projected Gradient solver (deprecated).
    # ‘cd’ is a Coordinate Descent solver (recommended).
    model = NMF(n_components=n_components, init=init,
                random_state=random_state)
    # TODO en boucle
    model.fit(mut_final)
    gene_comp = model.components_.copy()
    patient_strat = np.argmax(model.fit_transform(mut_final), axis=1).copy()
    # fit_transform more efficient than calling fit followed by transform

    model.fit(mut_diff)
    gene_comp_diff = model.components_.copy()
    patient_strat_diff = np.argmax(
        model.fit_transform(mut_diff), axis=1).copy()

    model.fit(mut_mean_qn)
    gene_comp_mean_qn = model.components_.copy()
    patient_strat_mean_qn = np.argmax(
        model.fit_transform(mut_mean_qn), axis=1).copy()

    model.fit(mut_median_qn)
    gene_comp_median_qn = model.components_.copy()
    patient_strat_median_qn = np.argmax(
        model.fit_transform(mut_median_qn), axis=1).copy()

    return (gene_comp, patient_strat,
            gene_comp_diff, patient_strat_diff,
            gene_comp_mean_qn, patient_strat_mean_qn,
            gene_comp_median_qn, patient_strat_median_qn)


def nmf_old(mut_final, mut_diff, mut_mean_qn, mut_median_qn, n_components,
            init='nndsvdar', random_state=0):
    # fit followed by transform
    model = NMF(n_components=n_components, init=init,
                random_state=random_state)
    # TODO en boucle
    model.fit(mut_final)
    gene_comp = model.components_.copy()
    patient_strat = np.argmax(model.transform(mut_final), axis=1).copy()

    model.fit(mut_diff)
    gene_comp_diff = model.components_.copy()
    patient_strat_diff = np.argmax(
        model.transform(mut_diff), axis=1).copy()

    model.fit(mut_mean_qn)
    gene_comp_mean_qn = model.components_.copy()
    patient_strat_mean_qn = np.argmax(
        model.transform(mut_mean_qn), axis=1).copy()

    model.fit(mut_median_qn)
    gene_comp_median_qn = model.components_.copy()
    patient_strat_median_qn = np.argmax(
        model.transform(mut_median_qn), axis=1).copy()

    return (gene_comp, patient_strat,
            gene_comp_diff, patient_strat_diff,
            gene_comp_mean_qn, patient_strat_mean_qn,
            gene_comp_median_qn, patient_strat_median_qn)


# Reuse scikit-learn functions

def check_non_negative(X, whom):
    X = X.data if sp.issparse(X) else X
    if (X < 0).any():
        raise ValueError("Negative values in data passed to %s" % whom)


def _sparseness(x):
    """Hoyer's measure of sparsity for a vector"""
    sqrt_n = np.sqrt(len(x))
    return (sqrt_n - LA.norm(x, 1) / LA.norm(x)) / (sqrt_n - 1)


def safe_vstack(Xs):
    if any(sp.issparse(X) for X in Xs):
        return sp.vstack(Xs)
    else:
        return np.vstack(Xs)


def NBS_init(X, n_components, init=None):
        n_samples, n_features = X.shape
        if init is None:
            if n_components < n_features:
                init = 'nndsvd'
            else:
                init = 'random'

        if init == 'nndsvd':
            W, H = _initialize_nmf(X, n_components)
        elif init == "random":
            rng = check_random_state(random_state)
            W = rng.randn(n_samples, n_components)
            # we do not write np.abs(W, out=W) to stay compatible with
            # numpy 1.5 and earlier where the 'out' keyword is not
            # supported as a kwarg on ufuncs
            np.abs(W, W)
            H = rng.randn(n_components, n_features)
            np.abs(H, H)
        else:
            raise ValueError(
                'Invalid init parameter: got %r instead of one of %r' %
                (init, (None, 'nndsvd', 'nndsvda', 'nndsvdar', 'random')))
        return W, H


def _initialize_nmf(X, n_components, variant=None, eps=1e-6,
                    random_state=None):
    """NNDSVD algorithm for NMF initialization.

    Computes a good initial guess for the non-negative
    rank k matrix approximation for X: X = WH

    Parameters
    ----------

    X : array, [n_samples, n_features]
        The data matrix to be decomposed.

    n_components : array, [n_components, n_features]
        The number of components desired in the approximation.

    variant : None | 'a' | 'ar'
        The variant of the NNDSVD algorithm.
        Accepts None, 'a', 'ar'
        None: leaves the zero entries as zero
        'a': Fills the zero entries with the average of X
        'ar': Fills the zero entries with standard normal random variates.
        Default: None

    eps: float
        Truncate all values less then this in output to zero.

    random_state : numpy.RandomState | int, optional
        The generator used to fill in the zeros, when using variant='ar'
        Default: numpy.random

    Returns
    -------

    (W, H) :
        Initial guesses for solving X ~= WH such that
        the number of columns in W is n_components.

    Remarks
    -------

    This implements the algorithm described in
    C. Boutsidis, E. Gallopoulos: SVD based
    initialization: A head start for nonnegative
    matrix factorization - Pattern Recognition, 2008

    http://tinyurl.com/nndsvd
    """
    check_non_negative(X, "NMF initialization")
    if variant not in (None, 'a', 'ar'):
        raise ValueError("Invalid variant name")

    U, S, V = randomized_svd(X, n_components)
    W, H = np.zeros(U.shape), np.zeros(V.shape)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = LA.norm(x_p), LA.norm(y_p)
        x_n_nrm, y_n_nrm = LA.norm(x_n), LA.norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    if variant == "a":
        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
    elif variant == "ar":
        random_state = check_random_state(random_state)
        avg = X.mean()
        W[W == 0] = abs(avg * random_state.randn(len(W[W == 0])) / 100)
        H[H == 0] = abs(avg * random_state.randn(len(H[H == 0])) / 100)

    return W, H


def gnmf(X, A, lambd=0, n_components=None, tol=1e-3, max_iter=100,
         verbose=False):

        X = check_array(X)
        check_non_negative(X, "NMF.fit")
        n_samples, n_features = X.shape

        if not n_components:
            n_components = min(n_samples, n_features)
        else:
            n_components = n_components

        W, H = NBS_init(X, n_components, init='nndsvd')

        list_reconstruction_err_ = []
        reconstruction_err_ = LA.norm(X - np.dot(W, H))
        list_reconstruction_err_.append(reconstruction_err_)

        eps = np.spacing(1)  # 2.2204460492503131e-16
        Lp = np.matrix(np.diag(np.asarray(A).sum(axis=0)))  # degree matrix
        Lm = A

        start = time.time()
        for n_iter in range(1, max_iter + 1):

            if verbose:
                print("Iteration ={:4d} / {:d} - - - - — Error = {:.2f} - - - - — Tolerance = {:f}".format(n_iter, max_iter, reconstruction_err_, tol))

            h1 = lambd*np.dot(H, Lm)+np.dot(W.T, (X+eps)/(np.dot(W, H)+eps))
            h2 = lambd*np.dot(H, Lp)+np.dot(W.T, np.ones(X.shape))
            H = np.multiply(H, (h1+eps)/(h2+eps))
            H[H <= 0] = eps
            H[np.isnan(H)] = eps

            w1 = np.dot((X+eps)/(np.dot(W, H)+eps), H.T)
            w2 = np.dot(np.ones(X.shape), H.T)
            W = np.multiply(W, (w1+eps)/(w2+eps))
            W[W <= 0] = eps
            W[np.isnan(W)] = eps

            if reconstruction_err_ > LA.norm(X - np.dot(W, H)):
                H = (1-eps)*H + eps*np.random.normal(
                    0, 1, (n_components, n_features))**2
                W = (1-eps)*W + eps*np.random.normal(
                    0, 1, (n_samples, n_components))**2
            reconstruction_err_ = LA.norm(X - np.dot(W, H))
            list_reconstruction_err_.append(reconstruction_err_)

            if reconstruction_err_ < tol:
                warnings.warn("Tolerance error reached during fit")
                break

            if np.isnan(W).any() or np.isnan(H).any():
                warnings.warn("NaN values at " + str(n_iter)+" Error="+str(
                    reconstruction_err_))
                break

            if n_iter == max_iter:
                warnings.warn("Iteration limit reached during fit")

        end = time.time()
        print("----- ", n_components, "components : %.2f sec -----"
              % (end-start))
        return (np.squeeze(np.asarray(W)), np.squeeze(np.asarray(H)),
                list_reconstruction_err_)
