#!/usr/bin/env python
# coding: utf-8
import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp
from scipy.io import loadmat, savemat
import itertools
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import randomized_svd
from scipy.spatial.distance import pdist, squareform
import warnings
import time
import datetime
import os
import glob
import collections

# NOTE some variable names changed:
# patientsNum -> n_patients
# genesNum -> n_genes
# patientsSelected -> patients_boot
# genesSelected -> genes_boot
# subselectionFiltered -> mut_boot
# subPPI -> ppi_boot
# permutationsNum -> n_permutations
# runBootstrap -> run_bootstrap
# subselectionDiffused -> mut_diff_boot
# subselectionQDiffused -> mut_mean_qn_boot
# subselectionQDiffusedMed -> mut_median_qn_boot
#    ->, ->, ->, ->, ->,


# Reuse scikit-learn functions
def check_non_negative(X, whom):
    X = X.data if sp.issparse(X) else X
    if (X < 0).any():
        raise ValueError("Negative values in data passed to %s" % whom)


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
    # dtype modification
    W, H = np.zeros(U.shape, dtype=np.float32), np.zeros(V.shape,
                                                         dtype=np.float32)
    # print('NMF initialization : W', type(W), W.dtype, W.shape)
    # print('NMF initialization : H', type(H), H.dtype, H.shape)
    # NMF initialization : W <class 'numpy.ndarray'> float32 (228, 2)
    # NMF initialization : H <class 'numpy.ndarray'> float32 (2, 9786)


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

    # print('NMF initialization - final : W', type(W), W.dtype, W.shape)
    # print('NMF initialization - final : H', type(H), H.dtype, H.shape)
    # NMF initialization - final : W <class 'numpy.ndarray'> float32 (228, 2)
    # NMF initialization - final : H <class 'numpy.ndarray'> float32 (2, 9786)

    return W, H


def gnmf(X, A, lambd=0, n_components=None, tol_nmf=1e-3, max_iter=100,
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

        for n_iter in range(1, max_iter + 1):

            if verbose:
                print("Iteration ={:4d} / {:d} - - - - — Error = {:.2f} - - - - — Tolerance = {:f}".format(n_iter, max_iter, reconstruction_err_, tol_nmf))

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

            if reconstruction_err_ < tol_nmf:
                warnings.warn("Tolerance error reached during fit")
                break

            if np.isnan(W).any() or np.isnan(H).any():
                warnings.warn("NaN values at " + str(n_iter)+" Error="+str(
                    reconstruction_err_))
                break

            if n_iter == max_iter:
                warnings.warn("Iteration limit reached during fit")

        return (np.squeeze(np.asarray(W)), np.squeeze(np.asarray(H)),
                list_reconstruction_err_)


def clustering_std_for_each_bootstrap(M):
    list_std = []
    for col in range(M.shape[1]):
        # for each colum (each bootstrap permutation), nan values removed
        one_col = M[:, col][~np.isnan(M[:, col])]
        # count of patient/gene number for each cluster
        occurences = [v for v in collections.Counter(one_col).values()]
        # standard deviation of occurences
        std = np.std(np.array(occurences))
        list_std.append(std)
    return np.array(list_std)


def bootstrap(result_folder, mut_type, mut_propag, ppi_final,
              influence_weight, simplification,
              alpha, tol, keep_singletons, ngh_max, min_mutation, max_mutation,
              n_components, n_permutations,
              run_bootstrap=False, lambd=1, tol_nmf=1e-3):

    boot_directory = result_folder+'bootstrap/'
    boot_mut_type_directory = boot_directory + mut_type + '/'

    if lambd > 0:
        boot_factorization_directory = boot_mut_type_directory + 'gnmf/'
    else:
        boot_factorization_directory = boot_mut_type_directory + 'nmf/'

    os.makedirs(boot_factorization_directory, exist_ok=True)
    boot_file = (boot_factorization_directory +
                 'bootstrap_weight={}_simp={}_alpha={}_tol={}_singletons={}_ngh={}_minMut={}_maxMut={}_comp={}_permut={}_lambd={}_tolNMF={}.mat'
                 .format(influence_weight, simplification, alpha, tol,
                         keep_singletons, ngh_max, min_mutation,
                         max_mutation, n_components, n_permutations, lambd,
                         tol_nmf))
    existance_same_param = os.path.exists(boot_file)
    # TODO overwrite condition
    if existance_same_param:
        bootstrap_data = loadmat(boot_file)
        genes_clustering = bootstrap_data['genes_clustering']
        patients_clustering = bootstrap_data['patients_clustering']
        print('***** Same parameters file of bootstrap already exists ***** {}'
              .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    else:
        if run_bootstrap:
            start = time.time()
            n_patients, n_genes = mut_propag.shape
            genes_clustering = np.zeros([n_genes, n_permutations],
                                        dtype=np.float32)*np.nan
            patients_clustering = np.zeros([n_patients, n_permutations],
                                           dtype=np.float32)*np.nan

            ppi_final = ppi_final.todense()

            EVERY_N = 100
            for perm in range(n_permutations):
                if (perm % EVERY_N) == 0:
                    print('Bootstrap : {} / {} permutations ----- {}'
                          .format(perm, n_permutations, datetime.datetime.now()
                                  .strftime("%Y-%m-%d %H:%M:%S")))

                # only 80% of patients and genes
                patients_boot = np.random.permutation(n_patients)[
                    0:int(n_patients*0.8)]
                genes_boot = np.random.permutation(n_genes)[0:int(n_genes*0.8)]
                mut_boot = mut_propag[patients_boot, :][:, genes_boot]
                ppi_boot = ppi_final[genes_boot, :][:, genes_boot]

                W, H, list_reconstruction_err_ = gnmf(mut_boot,
                                                      ppi_boot, lambd,
                                                      n_components, tol_nmf)
                if n_components > 1:
                    genes_clustering[genes_boot, perm] = np.argmax(H, axis=0)
                    patients_clustering[patients_boot, perm] = np.argmax(W, axis=1)
                else:
                    genes_clustering[genes_boot, perm] = H
                    patients_clustering[patients_boot, perm] = W

            # clustering std for each permutation of bootstrap
            genes_clustering_std = clustering_std_for_each_bootstrap(genes_clustering)
            patients_clustering_std = clustering_std_for_each_bootstrap(patients_clustering)

            savemat(boot_file, {'genes_clustering': genes_clustering,
                                'patients_clustering': patients_clustering,
                                'genes_clustering_std': genes_clustering_std,
                                'patients_clustering_std': patients_clustering_std},
                    do_compression=True)
            end = time.time()
            print("---------- Bootstrap = {} ---------- {}"
                  .format(datetime.timedelta(seconds=end-start),
                          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        else:
            newest_file = max(glob.iglob(
                boot_factorization_directory + '*.mat'), key=os.path.getctime)
            bootstrap_data = loadmat(newest_file)
            genes_clustering = bootstrap_data['genes_clustering']
            patients_clustering = bootstrap_data['patients_clustering']
    print('genes_clustering (bootstrap)', type(genes_clustering), genes_clustering.dtype)
    print('patients_clustering (bootstrap)', type(patients_clustering), patients_clustering.dtype)
    return genes_clustering, patients_clustering


def concensus_clustering_simple(mat):
    n_obj = mat.shape[0]
    distance = np.ones([n_obj, n_obj], dtype=np.float32)
    EVERY_N = 1000
    for obj1 in range(n_obj):
        if (obj1 % EVERY_N) == 0:
            print('consensus clustering : {} / {} objects ----- {}'
                  .format(obj1, n_obj, datetime.datetime.now()
                          .strftime("%Y-%m-%d %H:%M:%S")))

        for obj2 in range(obj1+1, n_obj):
            I = (np.isnan(mat[[obj1, obj2]]).sum(axis=0) == 0).sum()
            M = (mat[obj1, ] == mat[obj2, ]).sum()
            M.astype(np.float32)
            distance[obj1, obj2] = float(M)/I
            distance[obj2, obj1] = float(M)/I
    return distance  # return np ndarray


def consensus_clustering(result_folder, genes_clustering, patients_clustering,
                         influence_weight, simplification,
                         mut_type, alpha, tol, keep_singletons, ngh_max,
                         min_mutation, max_mutation,
                         n_components, n_permutations,
                         run_consensus=False, lambd=1, tol_nmf=1e-3):
        # TODO overwrite condition
    consensus_directory = result_folder+'consensus_clustering/'
    consensus_mut_type_directory = consensus_directory + mut_type + '/'

    if lambd > 0:
        consensus_factorization_directory = (
            consensus_mut_type_directory + 'gnmf/')
    else:
        consensus_factorization_directory = (
            consensus_mut_type_directory + 'nmf/')

    os.makedirs(consensus_factorization_directory, exist_ok=True)

    consensus_file = (consensus_factorization_directory +
                      'consensus_weight={}_simp={}_alpha={}_tol={}_singletons={}_ngh={}_minMut={}_maxMut={}_comp={}_permut={}_lambd={}_tolNMF={}.mat'
                      .format(influence_weight, simplification, alpha, tol,
                              keep_singletons, ngh_max,
                              min_mutation, max_mutation,
                              n_components, n_permutations, lambd, tol_nmf))
    existance_same_param = os.path.exists(consensus_file)

    if existance_same_param:
        consensus_data = loadmat(consensus_file)
        distance_genes = consensus_data['distance_genes']
        distance_patients = consensus_data['distance_patients']
        print('***** Same parameters file of consensus clustering already exists ***** {}'
              .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    else:
        if run_consensus:
            start = time.time()
            distance_genes = concensus_clustering_simple(genes_clustering)
            end = time.time()
            print("---------- distance_GENES = {} ---------- {}"
                  .format(datetime.timedelta(seconds=end-start),
                          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            start = time.time()
            distance_patients = concensus_clustering_simple(patients_clustering)
            end = time.time()
            print("---------- distance_PATIENTS = {} ---------- {}"
                  .format(datetime.timedelta(seconds=end-start),
                          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


            start = time.time()
            savemat(consensus_file, {'distance_genes': distance_genes,
                                     'distance_patients': distance_patients},
                    do_compression=True)
            end = time.time()
            print("---------- Save time = {} ---------- {}"
                  .format(datetime.timedelta(seconds=end-start),
                          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        else:
            #TODO condition wich no file exists
            newest_file = max(glob.iglob(
                consensus_factorization_directory + '*.mat'), key=os.path.getctime)
            consensus_data = loadmat(newest_file)
            distance_genes = consensus_data['distance_genes']
            distance_patients = consensus_data['distance_patients']

    return distance_genes, distance_patients
