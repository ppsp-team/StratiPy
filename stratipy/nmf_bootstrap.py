import sys
import os
from stratipy import load_data
import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp
from scipy.io import loadmat, savemat
import itertools
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import NMF
from scipy.spatial.distance import pdist, squareform
import warnings
import time
import datetime
import glob
import collections
from tqdm import tqdm, trange


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


def nmf(X, n_components, tol_nmf, max_iter=100):
    # return W,H as numpy array even X as sparce matrix
    check_non_negative(X, "NMF initialization")
    n_samples, n_features = X.shape

    if not n_components:
        n_components = min(n_samples, n_features)
    else:
        n_components = n_components

    model = NMF(n_components, tol=tol_nmf, max_iter=max_iter, init='nndsvd')
    W = model.fit_transform(X)
    H = model.components_

    return W, H


def full_bootstrap(mut_type, mut_propag, n_permutations, lambd, n_components,
                   tol_nmf, compute_gene_clustering, **kwargs):
    n_individuals, n_genes = mut_propag.shape
    genes_clustering = np.zeros([n_genes, n_permutations],
                                dtype=np.float32)*np.nan
    individuals_clustering = np.zeros([n_individuals, n_permutations],
                                   dtype=np.float32)*np.nan
    # GNMF
    if (mut_type != 'raw' and lambd != 0):
        ppi_final = kwargs['network'].todense()

    tqdm_bar = trange(n_permutations, desc='Bootstrap')
    for perm in tqdm_bar:
        # only 80% of individuals and genes
        individuals_boot = np.random.permutation(n_individuals)[
            0:int(n_individuals*0.8)]
        genes_boot = np.random.permutation(n_genes)[0:int(n_genes*0.8)]
        mut_boot = mut_propag[individuals_boot, :][:, genes_boot]

        # GNMF
        if (mut_type != 'raw' and lambd != 0):
            ppi_boot = ppi_final[genes_boot, :][:, genes_boot]
            W, H, list_reconstruction_err_ = gnmf(mut_boot,
                                                  ppi_boot, lambd,
                                                  n_components, tol_nmf)
        # NMF
        else:
            W, H = nmf(mut_boot, n_components, tol_nmf)

        if n_components > 1:
            if compute_gene_clustering:
                genes_clustering[genes_boot, perm] = np.argmax(H, axis=0)
            individuals_clustering[individuals_boot, perm] = np.argmax(W, axis=1)
        else:
            if compute_gene_clustering:
                genes_clustering[genes_boot, perm] = H
            patients_clustering[patients_boot, perm] = W

    return genes_clustering, patients_clustering


def sub_bootstrap(boot_factorization_directory, boot_filename, sub_perm,
                  mut_propag, ppi_final, lambd, n_components, tol_nmf):
    slurm_arr_task_id = int(sys.argv[3])
    print("Sub-bootstrap / Slurm array task ID:", slurm_arr_task_id)
    sub_boot_file = (boot_factorization_directory + 'sub{}_'
                     .format(slurm_arr_task_id) + boot_filename)
    existance_same_param = os.path.exists(sub_boot_file)

    if existance_same_param:
        print(' **** Same parameters file of Sub-bootstrap already exists')
    else:
        n_patients, n_genes = mut_propag.shape
        genes_clustering = np.zeros([n_genes, sub_perm], dtype=np.float32)*np.nan
        individuals_clustering = np.zeros([n_individuals, sub_perm],
                                          dtype=np.float32)*np.nan
        # GNMF
        if (mut_type != 'raw' and lambd != 0):
            ppi_final = kwargs['network'].todense()

        tqdm_bar = trange(sub_perm, desc='Sub-permutations of Bootstrap')
        for sub in tqdm_bar:
            individuals_boot = np.random.permutation(n_individuals)[
                0:int(n_individuals*0.8)]
            genes_boot = np.random.permutation(n_genes)[0:int(n_genes*0.8)]
            mut_boot = mut_propag[individuals_boot, :][:, genes_boot]

            # GNMF
            if (mut_type != 'raw' and lambd != 0):
                ppi_boot = ppi_final[genes_boot, :][:, genes_boot]
                # NMF or GNMF
                W, H, list_reconstruction_err_ = gnmf(mut_boot, ppi_boot, lambd,
                                                      n_components, tol_nmf)
            # NMF
            else:
                W, H = nmf(mut_boot, n_components, tol_nmf)

            genes_clustering[genes_boot, sub] = np.argmax(H, axis=0)
            individuals_clustering[individuals_boot, sub] = np.argmax(W, axis=1)

        savemat(sub_boot_file, {'genes_clustering': genes_clustering,
                                'individuals_clustering': individuals_clustering},
                do_compression=True)


def clustering_std_for_each_bootstrap(M):
    list_std = []
    for col in range(M.shape[1]):
        # for each column (each bootstrap permutation), nan values removed
        one_col = M[:, col][~np.isnan(M[:, col])]
        # count of patient/gene number for each cluster
        occurences = [v for v in collections.Counter(one_col).values()]
        # standard deviation of occurences
        std = np.std(np.array(occurences))
        list_std.append(std)
    return np.array(list_std)


def bootstrap_file(result_folder, mut_type, influence_weight, simplification,
                   alpha, tol, keep_singletons, ngh_max, min_mutation,
                   max_mutation, n_components, n_permutations, lambd, tol_nmf):
    boot_directory = result_folder+'bootstrap/'
    boot_mut_type_directory = boot_directory + mut_type + '/'

    if lambd > 0:
        boot_factorization_directory = boot_mut_type_directory + 'gnmf/'
    else:
        boot_factorization_directory = boot_mut_type_directory + 'nmf/'

    os.makedirs(boot_factorization_directory, exist_ok=True)
    boot_filename = ('bootstrap_weight={}_simp={}_alpha={}_tol={}_singletons={}_ngh={}_minMut={}_maxMut={}_comp={}_permut={}_lambd={}_tolNMF={}.mat'
                     .format(influence_weight, simplification, alpha, tol,
                         keep_singletons, ngh_max, min_mutation,
                         max_mutation, n_components, n_permutations, lambd,
                         tol_nmf))
    boot_file = boot_factorization_directory + boot_filename

    return boot_factorization_directory, boot_filename, boot_file


def get_mutation_profile_ppi(data_folder, ssc_mutation_data, ssc_subgroups,
                             gene_data, result_folder, mut_type, alpha, tol,
                             lambd, influence_weight, simplification,
                             keep_singletons, ngh_max):
    if mut_type == 'raw':
        mut_propag, mp_gene, mp_indiv = (
            load_data.load_specific_SSC_mutation_profile(
                data_folder, ssc_mutation_data, ssc_subgroups, gene_data))
    else:
        final_influence_mutation_directory = result_folder + 'final_influence/'
        final_influence_mutation_file = (
            final_influence_mutation_directory +
            'final_influence_mutation_profile_{}_alpha={}_tol={}.mat'.format(
                mut_type, alpha, tol))
        final_influence_data = loadmat(final_influence_mutation_file)
        mut_propag = final_influence_data['mut_propag']

        if lambd != 0:
            ppi_final_file = (
                final_influence_mutation_directory +
                'PPI_final_weight={}_simp={}_alpha={}_tol={}_singletons={}_ngh={}.mat'
                .format(influence_weight, simplification, alpha, tol, keep_singletons,
                        ngh_max))
            ppi_final_data = loadmat(ppi_final_file)
            ppi_final = ppi_final_data['ppi_final']

            return mut_propag, ppi_final
    return mut_propag


def bootstrap(data_folder, ssc_mutation_data, ssc_subgroups, gene_data,
              result_folder, mut_type, influence_weight,
              simplification, alpha, tol, keep_singletons, ngh_max,
              min_mutation, max_mutation, n_components, n_permutations,
              run_bootstrap, lambd, tol_nmf,
              compute_gene_clustering, sub_perm):
    boot_factorization_directory, boot_filename, boot_file = bootstrap_file(
        result_folder, mut_type, influence_weight, simplification, alpha, tol,
        keep_singletons, ngh_max, min_mutation, max_mutation, n_components,
        n_permutations, lambd, tol_nmf)
    existance_same_param = os.path.exists(boot_file)
    # TODO overwrite condition
    if existance_same_param:
        bootstrap_data = loadmat(boot_file)
        genes_clustering = bootstrap_data['genes_clustering']
        individuals_clustering = bootstrap_data['individuals_clustering']
        print(' **** Same parameters file of bootstrap already exists')

    else:
        if run_bootstrap == 'full':
            start = time.time()

            if (mut_type != 'raw' and lambd != 0):
                print('   === GNMF full bootstrap')
                mut_propag, ppi_final = get_mutation_profile_ppi(
                    data_folder, ssc_mutation_data, ssc_subgroups, gene_data,
                    result_folder, mut_type, alpha, tol, lambd,
                    influence_weight, simplification, keep_singletons, ngh_max)
                genes_clustering, individuals_clustering = full_bootstrap(
                    mut_type, mut_propag, n_permutations, lambd, n_components, tol_nmf,
                    compute_gene_clustering, network=ppi_final)
            else:
                print('   === NMF full bootstrap')
                mut_propag = get_mutation_profile_ppi(
                    data_folder, ssc_mutation_data, ssc_subgroups, gene_data,
                    result_folder, mut_type, alpha, tol, lambd,
                    influence_weight, simplification, keep_singletons, ngh_max)
                genes_clustering, individuals_clustering = full_bootstrap(
                    mut_type, mut_propag, n_permutations, lambd, n_components, tol_nmf,
                    compute_gene_clustering)

            end = time.time()
            print("---------- Bootstrap = {} ---------- {}"
                  .format(datetime.timedelta(seconds=end-start),
                          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            if compute_gene_clustering:
                genes_clustering_std = clustering_std_for_each_bootstrap(
                    genes_clustering)
            else:
                genes_clustering_std = float('NaN')
            individuals_clustering_std = clustering_std_for_each_bootstrap(
                individuals_clustering)

            savemat(boot_file, {'genes_clustering': genes_clustering,
                                'individuals_clustering': individuals_clustering,
                                'genes_clustering_std': genes_clustering_std,
                                'individuals_clustering_std': individuals_clustering_std},
                    do_compression=True)

            return genes_clustering, individuals_clustering

        elif run_bootstrap == 'split':
            start = time.time()
            if (mut_type != 'raw' and lambd != 0):
                print('   === GNMF split bootstrap')
                mut_propag, ppi_final = get_mutation_profile_ppi(
                    data_folder, ssc_mutation_data, ssc_subgroups, gene_data,
                    result_folder, mut_type, alpha, tol, lambd,
                    influence_weight, simplification, keep_singletons, ngh_max)
                sub_bootstrap(
                    mut_type, boot_factorization_directory, boot_filename, sub_perm,
                    mut_propag, lambd, n_components, tol_nmf, network=ppi_final)
            else:
                print('   === NMF split bootstrap')
                mut_propag = get_mutation_profile_ppi(
                    data_folder, ssc_mutation_data, ssc_subgroups, gene_data,
                    result_folder, mut_type, alpha, tol, lambd,
                    influence_weight, simplification, keep_singletons, ngh_max)
                sub_bootstrap(
                    mut_type, boot_factorization_directory, boot_filename, sub_perm,
                    mut_propag, lambd, n_components, tol_nmf)
            end = time.time()
            print("---------- Sub-Bootstrap = {} ---------- {}"
                  .format(datetime.timedelta(seconds=end-start),
                          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        # else:
        #     newest_file = max(glob.iglob(
        #         boot_factorization_directory + '*.mat'), key=os.path.getctime)
        #     bootstrap_data = loadmat(newest_file)
        #     genes_clustering = bootstrap_data['genes_clustering']
        #     individuals_clustering = bootstrap_data['individuals_clustering']
        #     return genes_clustering, individuals_clustering
