#!/usr/bin/env python
# coding: utf-8
import sys
import os
import importlib  # NOTE for python >= Python3.4
import scipy.sparse as sp
import numpy as np
import time
import datetime
from sklearn.model_selection import ParameterGrid
from scipy.io import loadmat, savemat
from memory_profiler import profile
# if "from memory_profiler import profile", timestamps will not be recorded
sys.path.append(os.path.abspath('../../stratipy'))
from stratipy import load_data, formatting_data, filtering_diffusion, clustering, hierarchical_clustering

# TODO PPI type param
param_grid = {'data_folder': ['../data/'],
              'patient_data': ['TCGA_UCEC'],
            #   'patient_data': ['Faroe'],
              'ppi_data': ['STRING'],
              'influence_weight': ['min'],
              'simplification': [True],
              'compute': [True],
              'overwrite': [False],
            #   'alpha': [0, 0.3, 0.5, 0.7, 1],
            #   'alpha': [0.7, 0.8, 0.9],
              'alpha': [0.7],
              'tol': [10e-3],
              'ngh_max': [11],
              'keep_singletons': [False],
            #   'min_mutation': [10],
              'min_mutation': [10],
              'max_mutation': [2000],
            #   'qn': [None, 'mean', 'median'],
              'qn': ['median'],
              'n_components': [2],
            #   'n_components': range(2, 10),
            #   'n_permutations': [1000],
              'n_permutations': [100],
              'run_bootstrap': [True],
              'run_consensus': [True],
            #   'lambd': [0, 1, 200],
              'lambd': [1],
              'tol_nmf': [1e-3],
              'linkage_method': ['ward']
            #   'linkage_method': ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']
              }

# 'lambd': range(0, 2)

# NOTE sys.stdout.flush()

@profile
def all_functions(params):

    if alpha == 0 and qn is not None:
        print('############ PASS ############')
        pass

    else:
        result_folder = data_folder + 'result_' + patient_data + '_' + ppi_data + '/'
        print(result_folder)
        print("alpha =", alpha)
        print("QN =", qn)
        print("k =", n_components)
        print("lambda =", lambd)
        print("PPI network =", ppi_data)

        # ------------ load_data.py ------------
        print("------------ load_data.py ------------")
        if patient_data == 'TCGA_UCEC':
            (patient_id, mutation_profile, gene_id_patient,
             gene_symbol_profile) = load_data.load_TCGA_UCEC_patient_data(
                 data_folder)

        elif patient_data == 'Faroe':
            mutation_profile, gene_id_patient = load_data.load_Faroe_Islands_data(
                data_folder)

        if ppi_data == 'STRING':
            gene_id_ppi, network = load_data.load_PPI_String(
                data_folder, ppi_data)

        else:
            gene_id_ppi, network = load_data.load_PPI_Y2H_or_APID(
                data_folder, ppi_data)

        # ------------ formatting_data.py ------------
        print("------------ formatting_data.py ------------")
        (network, mutation_profile,
         idx_ppi, idx_mut, idx_ppi_only, idx_mut_only) = (
            formatting_data.classify_gene_index(
                network, mutation_profile, gene_id_ppi, gene_id_patient))

        (ppi_total, mut_total, ppi_filt, mut_filt) = (
            formatting_data.all_genes_in_submatrices(
                network, idx_ppi, idx_mut, idx_ppi_only, idx_mut_only,
                mutation_profile))

        # ------------ filtering_diffusion.py ------------
        print("------------ filtering_diffusion.py ------------")
        # ppi_influence = (
        #     filtering_diffusion.calcul_ppi_influence(
        #         sp.eye(ppi_filt.shape[0]), ppi_filt,
        #         result_folder, compute, overwrite, alpha, tol))

        final_influence = (
            filtering_diffusion.calcul_final_influence(
                sp.eye(ppi_filt.shape[0], dtype=np.float32), ppi_filt,
                result_folder, influence_weight, simplification,
                compute, overwrite, alpha, tol))

        ppi_final, mut_final = filtering_diffusion.filter_ppi_patients(
            ppi_total, mut_total, ppi_filt, final_influence, ngh_max,
            keep_singletons, min_mutation, max_mutation)

        mut_type, mut_propag = filtering_diffusion.propagation_profile(
            mut_final, ppi_filt, alpha, tol, qn)

        # ------------ clustering.py ------------
        print("------------ clustering.py ------------")
        genes_clustering, patients_clustering = (clustering.bootstrap(
            result_folder, mut_type, mut_propag, ppi_final,
            influence_weight, simplification,
            alpha, tol, keep_singletons, ngh_max, min_mutation, max_mutation,
            n_components, n_permutations,
            run_bootstrap, lambd, tol_nmf))

        distance_genes, distance_patients = clustering.consensus_clustering(
            result_folder, genes_clustering, patients_clustering,
            influence_weight, simplification, mut_type,
            alpha, tol, keep_singletons, ngh_max, min_mutation, max_mutation,
            n_components, n_permutations, run_consensus, lambd, tol_nmf)

        # ------------ hierarchical_clustering.py ------------
        print("------------ hierarchical_clustering.py ------------")
        # if alpha > 0:
        #     if qn == 'mean':
        #         mut_type = 'mean_qn'
        #     elif qn == 'median':
        #         mut_type = 'median_qn'
        #     else:
        #         mut_type = 'diff'
        # else:
        #     mut_type = 'raw'
        # print("mutation type =", mut_type)
        #
        # consensus_directory = result_folder+'consensus_clustering/'
        # consensus_mut_type_directory = consensus_directory + mut_type + '/'
        #
        # hierarchical_directory = result_folder+'hierarchical_clustering/'
        # os.makedirs(hierarchical_directory, exist_ok=True)
        # hierarchical_mut_type_directory = hierarchical_directory + mut_type + '/'
        # os.makedirs(hierarchical_mut_type_directory, exist_ok=True)
        #
        # if lambd > 0:
        #     consensus_factorization_directory = (consensus_mut_type_directory + 'gnmf/')
        #     hierarchical_factorization_directory = (hierarchical_mut_type_directory + 'gnmf/')
        #
        # else:
        #     consensus_factorization_directory = (consensus_mut_type_directory + 'nmf/')
        #     hierarchical_factorization_directory = (hierarchical_mut_type_directory + 'nmf/')
        # os.makedirs(hierarchical_factorization_directory, exist_ok=True)
        #
        # consensus_file = (consensus_factorization_directory +
        #                   'consensus_alpha={}_tol={}_singletons={}_ngh={}_minMut={}_maxMut={}_comp={}_permut={}_lambd={}_tolNMF={}.mat'
        #                   .format(alpha, tol, keep_singletons, ngh_max,
        #                           min_mutation, max_mutation,
        #                           n_components, n_permutations, lambd, tol_nmf))
        #
        # consensus_data = loadmat(consensus_file)
        # distance_patients = consensus_data['distance_patients']
        #
        # hierarchical_clustering.distance_matrix(
        #     hierarchical_factorization_directory, distance_patients, ppi_data,
        #     mut_type,
        #     alpha, tol,  keep_singletons, ngh_max, min_mutation, max_mutation,
        #     n_components, n_permutations, lambd, tol_nmf, linkage_method)
        hierarchical_clustering.distance_patients_from_consensus_file(
            result_folder, distance_patients, ppi_data, mut_type,
            influence_weight, simplification, alpha, tol,  keep_singletons,
            ngh_max, min_mutation, max_mutation, n_components, n_permutations,
            lambd, tol_nmf, linkage_method)


if (sys.version_info < (3, 2)):
    raise "Must be using Python ≥ 3.2"

else:
    start_all = time.time()
    for params in list(ParameterGrid(param_grid)):
        start = time.time()
        print(params)
        for i in params.keys():
            exec("%s = %s" % (i, 'params[i]'))
        all_functions(params)
        end = time.time()
        print('---------- ONE STEP = {} ---------- {}'
              .format(datetime.timedelta(seconds=end-start),
                      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    end_all = time.time()
    print('---------- ALL = {} ---------- {}'
          .format(datetime.timedelta(seconds = end_all - start_all),
                  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
