#!/usr/bin/env python
# coding: utf-8
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm
from scipy.io import loadmat, savemat
from nbs import Ppi, Patient
from subprocess import call
# import h5py
import os
import glob
import time
import datetime

# NOTE mutationProfileDiffusion -> propagation
# mutationProfile -> M, PPIAdjacencyMatrix -> adj, dataFolder -> data_folder
# PPI_influence_min -> ppi_influence_min, PPI_influence_max-> ppi_influence_max
# PPI_influence()-> calcul_ppi_influence(), PPI_influence -> ppi_influence
# influenceDistance->influence_distance
# influenceMat -> ppi_influence, PPIneighboorsMax -> ngh_max,
# bestInfluencers -> best_influencers
# filteredGenes -> deg0, keepSingletons -> keep_singletons
# mutationsMin -> min_mutation, mutationsMax -> mutationsMax
# newnet -> ppi_ngh, netFinal -> ppi_final, mutFinal -> mut_final
# filteredPatients -> filtered_patients


def propagation(M, adj, alpha=0.7, tol=10e-6):  # TODO equation, M, alpha
    """Network propagation iterative process

    Iterative algorithm for apply propagation using random walk on a network:
        Initialize::
            X1 = M

        Repeat::
            X2 = alpha * X1.A + (1-alpha) * M
            X1 = X2

        Until::
            norm(X2-X1) < tol

        Where::
            A : degree-normalized adjacency matrix

    Parameters
    ----------
    M : sparse matrix
        Data matrix to be diffused.

    adj : sparse matrix
        Adjacency matrice.

    alpha : float, default: 0.7
        Diffusion/propagation factor with 0 <= alpha <= 1.
        For alpha = 0 : no diffusion.
        For alpha = 1 :

    tol : float, default: 10e-6
        Convergence threshold.

    Returns
    -------
    X2 : sparse matrix
        Smoothed matrix.
    """
    print(' ==== propagation ==== ')
    n = adj.shape[0]
    adj = adj+sp.eye(n)

    d = sp.dia_matrix((np.array(adj.sum(axis=0))**-1, [0]), shape=(n,  n))
    A = adj.dot(d)

    X1 = M
    X2 = alpha * X1.dot(A) + (1-alpha) * M
    i = 0
    while norm(X2-X1) > tol:
        X1 = X2
        X2 = alpha * X1.dot(A) + (1-alpha) * M
        i += 1
        print('Propagation iteration = {}  ----- {}'.format(
            i, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    return X2


def compare_ij_ji(ppi, out_min=True, out_max=True):
    """Helper function for calcul_ppi_influence

    In most cases the influence (propagation) is not symmetric. We have to
    compare weight (a_ij) and (a_ji) for all pairs in order to obtain symmetric
    matrix/matrices. 2 choices available: minimum or maximum weight.
        a = min [(a_ij),(a_ji)]
        a = max [(a_ij),(a_ji)]
    Minimum weight is chosen to avoid Hubs phenomenon.

    Parameters
    ----------
    ppi : sparse matrix
        Matrice to apply comparison.

    out_min, out_max : boolean, default: True
        Minimum and/or maximum weight is chosen.

    Returns
    -------
    ppi_min, ppi_max : sparse matrix
        Symmertric matrix with minimum and/or maximum weight.
    """
    # TODO matrice type of ppi
    ppi = ppi.tolil()  # need "lil_matrix" for reshape
    # transpose to compare ppi(ij) and ppi(ji)
    ppi_transp = sp.lil_matrix.transpose(ppi)
    # reshape to 1D matrix
    ppi_1d = ppi.reshape((1, ppi.shape[0]**2))
    ppi_1d_transp = ppi_transp.reshape((1, ppi.shape[0]**2))

    # reshapeto original size matrix after comparison (min/max)
    if out_min and out_max:
        ppi_min = (sp.coo_matrix.tolil(
            sp.coo_matrix.min(sp.vstack([ppi_1d, ppi_1d_transp]), axis=0))
                   ).reshape((ppi.shape[0], ppi.shape[0]))
        ppi_max = (sp.coo_matrix.tolil(
            sp.coo_matrix.max(sp.vstack([ppi_1d, ppi_1d_transp]), axis=0))
                   ).reshape((ppi.shape[0], ppi.shape[0]))
        return ppi_min, ppi_max

    elif out_min:
        ppi_min = (sp.coo_matrix.tolil(
            sp.coo_matrix.min(sp.vstack([ppi_1d, ppi_1d_transp]), axis=0))
                   ).reshape((ppi.shape[0], ppi.shape[0]))
        return ppi_min

    elif out_max:
        ppi_max = (sp.coo_matrix.tolil(
            sp.coo_matrix.max(sp.vstack([ppi_1d, ppi_1d_transp]), axis=0))
                   ).reshape((ppi.shape[0], ppi.shape[0]))
        return ppi_max
    else:
        print('You have to choice Min or Max')  # TODO change error message


def calcul_ppi_influence(M, adj, data_folder,
                         compute=False, overwrite=False, alpha=0.7, tol=10e-6):
    # TODO for 'maximum weight'
    """Compute network influence score

    Network propagation iterative process is applied on PPI. (1) The  network
    influence distance matrix and (2) influence matrices based on minimum /
    maximum weight are saved as MATLAB-style files (.mat).
        - (1) : 'influence_distance_alpha={}_tol={}.mat'
                in 'influence_distance' directory
        - (2) : 'ppi_influence_alpha={}_tol={}.mat'
                in 'ppi_influence' directory
    Where {} are parameter values. The directories will be automatically
    created if not exist.

    If compute=False, the latest data of directory will be taken into
    account:
        - latest data with same parameters (alpha and tol)
        - if not exist, latest data of directory but with differents parameters

    Parameters
    ----------
    M : sparse matrix
        Data matrix to be diffused.

    adj : sparse matrix
        Adjacency matrice.

    data_folder : str
        Path to create a new directory for save new files. If you want to creat
        in current directory, enter '/directory_name'. Absolute path is also
        supported.

    compute : boolean, default: False
        If True, new network influence score will be computed.
        If False, the latest network influence score  will be taken into
        account.

    overwrite : boolean, default: False
        If True, new network influence score will be computed even if the file
        which same parameters already exists in the directory.

    alpha : float, default: 0.7
        Diffusion (propagation) factor with 0 <= alpha <= 1.
        For alpha = 0 : no diffusion.
        For alpha = 1 :

    tol : float, default: 10e-6
        Convergence threshold.

    Returns
    -------
    ppi_influence : sparse matrix
        Smoothed PPI influence matrices based on minimum / maximum weight.
    """
    ppi_influence_directory = data_folder+'ppi_influence/'
    ppi_influence_file = ppi_influence_directory + 'ppi_influence_alpha={}_tol={}.mat'.format(alpha, tol)
    existance_same_param = os.path.exists(ppi_influence_file)
    # TODO overwrite condition
    if existance_same_param:
        influence_data = loadmat(ppi_influence_file)
        ppi_influence_min = influence_data['ppi_influence_min']
        # ppi_influence_max = influence_data['ppi_influence_max']
        # alpha = influence_data['alpha'][0][0]
        print('***** Same parameters file of PPI influence already exists ***** {}'
              .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    else:
        if compute:
            start = time.time()
            influence = propagation(M, adj, alpha, tol)

            influence_distance_directory = data_folder+'influence_distance/'
            os.makedirs(influence_distance_directory, exist_ok=True)  # NOTE For Python ≥ 3.2
            influence_distance_file = (
                influence_distance_directory + 'influence_distance_alpha={}_tol={}.mat'.format(alpha, tol))

            # save influence distance before simplification with parameters' values in filename
            savemat(influence_distance_file,
                    {'influence': influence, 'alpha': alpha})  # do_compression=True

            # simplification: multiply by PPI adjacency matrix
            influence = influence.multiply(sp.lil_matrix(adj))  # TODO test without
            # -> influence as csr_matrix

            ppi_influence_min, ppi_influence_max = compare_ij_ji(
                influence, out_min=True, out_max=True)

            os.makedirs(ppi_influence_directory, exist_ok=True)# ppi_influence_file = ppi_influence_directory + 'ppi_influence_alpha={}_tol={}.mat'.format(alpha, tol)

            savemat(ppi_influence_file,
                    {'ppi_influence_min': ppi_influence_min,
                     'ppi_influence_max': ppi_influence_max,
                     'alpha': alpha}, do_compression=True)

            end = time.time()
            print("---------- Influence distance = {} ---------- {}"
                  .format(datetime.timedelta(seconds=end-start),
                          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        else:
            newest_file = max(glob.iglob(ppi_influence_directory + '*.mat'), key=os.path.getctime)
            influence_data = loadmat(newest_file)
            ppi_influence_min = influence_data['ppi_influence_min']
            # TODO print 'different parameters '

    return ppi_influence_min


def best_neighboors(ppi_filt, ppi_influence, ngh_max):
    """Helper function for filter_ppi_patients

    Keeps only the connections with the best influencers.

    Parameters
    ----------
    ppi_filt : sparse matrix
        Filtration from ppi_total : only genes in PPI are considered.

    ppi_influence : sparse matrix
        Smoothed PPI influence matrices based on minimum or maximum weight.

    ngh_max : int
        Number of best influencers in PPI.

    Returns
    -------
    ppi_ngh : sparse matrix
        PPI with only best influencers.
    """
    ppi_influence = ppi_influence.todense()
    ppi_filt = ppi_filt.todense()
    ppi_ngh = np.zeros(ppi_filt.shape)
    for i in range(ppi_filt.shape[0]):
        best_influencers = np.argpartition(
            -ppi_influence[i, :], ngh_max)[:, :ngh_max]
        ppi_ngh[i, best_influencers] = ppi_filt[i, best_influencers]
    ppi_ngh = np.max(np.dstack((ppi_ngh, ppi_ngh.T)), axis=2)
    # too stringent if np.min
    return sp.csc_matrix(ppi_ngh)


def filter_ppi_patients(ppi_total, mut_total, ppi_filt, ppi_influence, ngh_max,
                        keep_singletons=False,
                        min_mutation=10, max_mutation=2000):
    """Keeping only the connections with the best influencers and Filtering some
    patients based on mutation number

    'the 11 most influential neighbors of each gene in the network as
    determined by network influence distance were used'
    'Only mutation data generated using the Illumina GAIIx platform were
    retained for subsequent analy- sis, and patients with fewer than 10
    mutations were discarded.'

    Parameters
    ----------
    ppi_total : sparse matrix
        Built from all sparse sub-matrices (AA, ... , CC).

    mut_total : sparse matrix
        Patients' mutation profiles of all genes (rows: patients,
        columns: genes of AA, BB and CC).

    ppi_filt : sparse matrix
        Filtration from ppi_total : only genes in PPI are considered.

    ppi_influence : sparse matrix
        Smoothed PPI influence matrices based on minimum or maximum weight.

    ngh_max : int
        Number of best influencers in PPI.

    keep_singletons : boolean, default: False
        If True, proteins not annotated in PPI (genes founded only in patients'
        mutation profiles) will be also considered.
        If False, only annotated proteins in PPI will be considered.

    min_mutation, max_mutation : int
        Numbers of lowest mutations and highest mutations per patient.

    Returns
    -------
    ppi_final, mut_final : sparse matrix
        PPI and mutation profiles after filtering.
    """
    ppi_ngh = best_neighboors(ppi_filt, ppi_influence, ngh_max)
    deg0 = Ppi(ppi_total).deg == 0  # True if protein degree = 0

    if keep_singletons:
        ppi_final = sp.bmat([
            [ppi_ngh, sp.csc_matrix((ppi_ngh.shape[0], sum(deg0)))],
            [sp.csc_matrix((sum(deg0), ppi_ngh.shape[0])),
             sp.csc_matrix((sum(deg0), sum(deg0)))]
            ])  # -> COO matrix
        # mut_final=sp.bmat([[mut_total[:,deg0==False],mut_total[:,deg0==True]]])
        mut_final = mut_total
    else:
        ppi_final = ppi_ngh
        mut_final = mut_total[:, Ppi(ppi_total).deg > 0]

    # filtered_patients = np.array([k < min_mutation or k > max_mutation for k in Patient(mut_final).mut_per_patient])
    # mut_final = mut_final[filtered_patients == False, :]

    # to avoid worse comparison '== False'
    mut_final = mut_final[np.array([min_mutation < k < max_mutation for k in
                                    Patient(mut_final).mut_per_patient])]

    print("Removing %i patients with less than %i or more than %i mutations" %
          (mut_total.shape[0]-mut_final.shape[0], min_mutation, max_mutation))
    print("New adjacency matrix:", ppi_final.shape)
    print("New mutation profile matrix:", mut_final.shape)

    return ppi_final, mut_final


def quantile_norm_mean(anarray):
    """Helper function for propagation_profile

    Forces the observations/variables to have identical intensity distribution.

    Parameters
    ----------
    ppi_filt : sparse matrix
        Filtration from ppi_total : only genes in PPI are considered.

    ppi_influence : sparse matrix
        Smoothed PPI influence matrices based on minimum or maximum weight.

    ngh_max : int
        Number of best influencers in PPI.

    Returns
    -------
    ppi_ngh : sparse matrix
        PPI with only best influencers.
    """
    A = np.squeeze(np.asarray(anarray.T))
    AA = np.zeros_like(A)
    I = np.argsort(A, axis=0)
    AA[I, np.arange(A.shape[1])] = np.mean(A[I, np.arange(A.shape[1])],
                                           axis=1)[:, np.newaxis]
    return AA.T


def quantile_norm_median(anarray):
    A = np.squeeze(np.asarray(anarray.T))
    AA = np.zeros_like(A)
    I = np.argsort(A, axis=0)
    AA[I, np.arange(A.shape[1])] = np.median(A[I, np.arange(A.shape[1])],
                                             axis=1)[:, np.newaxis]
    return AA.T


def propagation_profile(mut_raw, adj, alpha, tol, qn):
        #  TODO error messages
        start = time.time()
        if alpha > 0:
            # TODO verification of same parameter file
            mut_propag = propagation(mut_raw, adj, alpha, tol).todense()
            mut_propag[np.isnan(mut_propag)] = 0
            if qn == 'mean':
                mut_type = 'mean_qn'
                mut_propag = quantile_norm_mean(mut_propag)
            elif qn == 'median':
                mut_type = 'median_qn'
                mut_propag = quantile_norm_median(mut_propag)
            else:
                mut_type = 'diff'

            end = time.time()
            print("---------- Propagation on {} mutation profile = {} ---------- {}"
                  .format(mut_type,
                          datetime.timedelta(seconds=end-start),
                          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            return mut_type, mut_propag

        else:
            mut_type = 'raw'
            mut_raw = mut_raw.todense()

            end = time.time()
            print("---------- Propagation on {} mutation profile = {} ---------- {}"
                  .format(mut_type,
                          datetime.timedelta(seconds=end-start),
                          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            return mut_type, mut_raw
