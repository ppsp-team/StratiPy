import sys
import os
sys.path.append(os.path.abspath('../../stratipy_cluster'))
from stratipy import nmf_bootstrap
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
from tqdm import tqdm, trange


def consensus_filename(result_folder, influence_weight, simplification, mut_type,
                   alpha, tol, keep_singletons, ngh_max, min_mutation,
                   max_mutation, n_components, n_permutations, lambd, tol_nmf):
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
    return consensus_file


def merge_sub_bootstrap(boot_factorization_directory, boot_filename, boot_file,
                        n_permutations, compute_gene_clustering):
    subfiles_list = glob.glob(boot_factorization_directory + 'sub*_' +
                              boot_filename)
    genes_clustering = []
    patients_clustering = []
    for file_path in subfiles_list:
        genes_clustering.append(loadmat(file_path)['genes_clustering'])
        patients_clustering.append(loadmat(file_path)['patients_clustering'])
    genes_clustering = np.hstack(genes_clustering)
    patients_clustering = np.hstack(patients_clustering)

    if (genes_clustering.shape[1] != n_permutations or
        patients_clustering.shape[1] != n_permutations):
        print('Bootstrap permutation error:  {} were requested but {} (gene clustering) and {} (individual clustering) obtained'
              .format(n_permutations, genes_clustering.shape[1],
                      patients_clustering.shape[1]))

    # clustering std for each permutation of bootstrap
    if compute_gene_clustering:
        genes_clustering_std = nmf_bootstrap.clustering_std_for_each_bootstrap(
            genes_clustering)
    else:
        genes_clustering_std = float('NaN')
    patients_clustering_std = nmf_bootstrap.clustering_std_for_each_bootstrap(
        patients_clustering)

    savemat(boot_file, {'genes_clustering': genes_clustering,
                        'patients_clustering': patients_clustering,
                        'genes_clustering_std': genes_clustering_std,
                        'patients_clustering_std': patients_clustering_std},
            do_compression=True)
    print(" ==== Sub-bootstrap data merged")

    return genes_clustering, patients_clustering


def simple_consensus(mat):
    n_obj = mat.shape[0]
    distance = np.ones([n_obj, n_obj], dtype=np.float32)

    # tqdm_bar = trange(n_obj, desc='Consensus clustering')
    # for obj1 in tqdm_bar:
    for obj1 in range(n_obj):
        for obj2 in range(obj1+1, n_obj):
            I = (np.isnan(mat[[obj1, obj2]]).sum(axis=0) == 0).sum()
            M = (mat[obj1, ] == mat[obj2, ]).sum()
            M.astype(np.float32)
            distance[obj1, obj2] = float(M)/I
            distance[obj2, obj1] = float(M)/I
    return distance  # return np ndarray


def run_save_consensus(consensus_file, genes_clustering, patients_clustering,
                       compute_gene_clustering=False):
    start = time.time()
    if compute_gene_clustering:
        distance_genes = simple_consensus(genes_clustering)
    else:
        distance_genes = float('NaN')
    end = time.time()
    print(" ==== distance_Genes = {} ---------- {}"
          .format(datetime.timedelta(seconds=end-start),
                  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    start = time.time()
    distance_patients = simple_consensus(patients_clustering)
    end = time.time()
    print(" ==== distance_patients = {} ---------- {}"
          .format(datetime.timedelta(seconds=end-start),
                  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    savemat(consensus_file, {'distance_genes': distance_genes,
                             'distance_patients': distance_patients},
            do_compression=True)

    return distance_genes, distance_patients


def consensus_from_full_bootstrap(consensus_file, boot_file, run_consensus,
                                  compute_gene_clustering=False):
    # TODO overwrite condition
    existance_same_param = os.path.exists(consensus_file)

    if existance_same_param:
        consensus_data = loadmat(consensus_file)
        distance_genes = consensus_data['distance_genes']
        distance_patients = consensus_data['distance_patients']
        print(' **** Same parameters file of consensus clustering already exists')
    else:
        if run_consensus:
            boot_data = loadmat(boot_file)
            genes_clustering = boot_data['genes_clustering']
            patients_clustering = boot_data['patients_clustering']

            distance_genes, distance_patients = run_save_consensus(
                consensus_file, genes_clustering, patients_clustering,
                compute_gene_clustering)
        else:
            #TODO condition wich no file exists
            newest_file = max(glob.iglob(
                consensus_factorization_directory + '*.mat'), key=os.path.getctime)
            consensus_data = loadmat(newest_file)
            distance_genes = consensus_data['distance_genes']
            distance_patients = consensus_data['distance_patients']

    return distance_genes, distance_patients


def consensus_from_sub_bootstrap(consensus_file, run_consensus,
                                 boot_factorization_directory, boot_filename,
                                 boot_file, n_permutations,
                                 compute_gene_clustering=False):
    # TODO overwrite condition
    existance_same_param = os.path.exists(consensus_file)

    if existance_same_param:
        consensus_data = loadmat(consensus_file)
        distance_genes = consensus_data['distance_genes']
        distance_patients = consensus_data['distance_patients']
        print(' **** Same parameters file of consensus clustering already exists')
    else:
        if run_consensus:
            existance_merged_boot = os.path.exists(boot_file)
            if existance_merged_boot:
                boot_data = loadmat(boot_file)
                genes_clustering = boot_data['genes_clustering']
                patients_clustering = boot_data['patients_clustering']

            else:
                genes_clustering, patients_clustering = merge_sub_bootstrap(
                    boot_factorization_directory, boot_filename, boot_file,
                    n_permutations, compute_gene_clustering)

            distance_genes, distance_patients = run_save_consensus(
                consensus_file, genes_clustering, patients_clustering,
                compute_gene_clustering)
        else:
            #TODO condition wich no file exists
            newest_file = max(glob.iglob(
                consensus_factorization_directory + '*.mat'), key=os.path.getctime)
            consensus_data = loadmat(newest_file)
            distance_genes = consensus_data['distance_genes']
            distance_patients = consensus_data['distance_patients']

        # remove all sub-bootstrap files
        subfiles_list = glob.glob(boot_factorization_directory + 'sub*_' +
                                  boot_filename)
        for file_path in subfiles_list:
            os.remove(file_path)
        print(" ....... Removed sub-bootstrap files: ", len(subfiles_list))

    return distance_genes, distance_patients


def sub_consensus(result_folder, mut_type, influence_weight,
                  simplification, alpha, tol, keep_singletons, ngh_max,
                  min_mutation, max_mutation, n_components, n_permutations,
                  lambd, tol_nmf, compute_gene_clustering, run_consensus):
    consensus_file = consensus_filename(
        result_folder, influence_weight, simplification, mut_type, alpha, tol,
        keep_singletons, ngh_max, min_mutation, max_mutation, n_components,
        n_permutations, lambd, tol_nmf)

    (boot_factorization_directory, boot_filename,
     boot_file) = nmf_bootstrap.bootstrap_file(
         result_folder, mut_type, influence_weight, simplification, alpha, tol,
         keep_singletons, ngh_max, min_mutation, max_mutation, n_components,
         n_permutations, lambd, tol_nmf)

    distance_genes, distance_patients = consensus_from_sub_bootstrap(
        consensus_file, run_consensus, boot_factorization_directory,
        boot_filename, boot_file, n_permutations, compute_gene_clustering)

    # distance_genes, distance_patients = consensus_from_full_bootstrap(
    #     consensus_file, boot_file, run_consensus, compute_gene_clustering)

    return distance_genes, distance_patients
