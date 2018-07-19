#!/usr/bin/env python
# coding: utf-8
import sys
import os
sys.path.append(os.path.abspath('../../stratipy'))
from stratipy import load_data, formatting_data, filtering_diffusion, clustering, hierarchical_clustering
import importlib  # NOTE for python >= Python3.4
import scipy.sparse as sp
import numpy as np
import time
from datetime import datetime
from sklearn.model_selection import ParameterGrid
from scipy.io import loadmat, savemat
# from memory_profiler import profile
# if "from memory_profiler import profile", timestamps will not be recorded

# importlib.reload(load_data)
# print(dir(load_data))

# TODO PPI type param
param_grid = {'data_folder': ['../data/'],
              'patient_data': ['SSC'],
            #   'patient_data': ['Faroe'],
              # 'ssc_mutation_data': ['LoF', 'missense'],
              'ssc_mutation_data': ['LoF_mis30'],
              'ssc_subgroups': ['SSC1', 'SSC2'],
              # 'ssc_subgroups': ['SSC', 'SSC1', 'SSC2'],
              'gene_data': ['all', 'pli', 'sfari', 'brain1SD', 'brain2SD'],
              # 'gene_data': ['all'],
              'ppi_data': ['APID'], #'Hofree_STRING'
              'influence_weight': ['min'],
              'simplification': [True],
              'compute': [True],
              'overwrite': [False],
            #   'alpha': [0, 0.3, 0.5, 0.7, 1],
            #   'alpha': [0.7, 0.8, 0.9],
              'alpha': [0, 0.7],
              'tol': [10e-3],
              'ngh_max': [11],
              'keep_singletons': [False],
            #   'min_mutation': [10],
              'min_mutation': [0],
              'max_mutation': [2000],
            #   'qn': [None, 'mean', 'median'],
              'qn': [None, 'median'],
              'n_components': [2],
            #   'n_components': range(2, 10),
            #   'n_permutations': [1000],
              'n_permutations': [100],
              'run_bootstrap': [True],
              'run_consensus': [True],
            #   'lambd': [0, 1, 200],
              'lambd': [0],
              'tol_nmf': [1e-3],
              'compute_gene_clustering': [False],
              'linkage_method': ['average']
            #   'linkage_method': ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']
              }

# 'lambd': range(0, 2)

# NOTE sys.stdout.flush()

# @profile
def all_functions(params):

    if alpha == 0 and qn is not None:
        print('############ PASS ############')
        pass

    else:
        if patient_data == 'SSC':
            result_folder = (data_folder + 'result_' + ssc_mutation_data + '_' +
                                 ssc_subgroups + '_' + gene_data + '_' +  ppi_data + '/')
        else:
            result_folder = (data_folder + 'result_' + patient_data + '_' +
                             ppi_data + '/')
        print(result_folder, flush=True)
        print("alpha =", alpha, flush=True)
        print("QN =", qn, flush=True)
        print("k =", n_components, flush=True)
        print("lambda =", lambd, flush=True)
        print("PPI network =", ppi_data, flush=True)

        # ------------ load_data.py ------------
        print("------------ load_data.py ------------ {}"
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        if patient_data == 'TCGA_UCEC':
            (patient_id, mutation_profile, gene_id_patient,
             gene_symbol_profile) = load_data.load_TCGA_UCEC_patient_data(
                 data_folder)

        elif patient_data == 'Faroe':
            mutation_profile, gene_id_patient = (
                load_data.load_Faroe_Islands_data(data_folder))

        elif patient_data == 'SSC':
            mutation_profile, gene_id_patient, patient_id = (
                load_data.load_specific_SSC_mutation_profile(
                    data_folder, ssc_mutation_data, ssc_subgroups, gene_data))

        if ppi_data == 'Hofree_STRING':
            gene_id_ppi, network = load_data.load_Hofree_PPI_String(
                data_folder, ppi_data)

        else:
            gene_id_ppi, network = load_data.load_PPI_network(
                data_folder, ppi_data)

        # ------------ formatting_data.py ------------
        print("------------ formatting_data.py ------------ {}"
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        (network, mutation_profile,
         idx_ppi, idx_mut, idx_ppi_only, idx_mut_only) = (
            formatting_data.classify_gene_index(
                network, mutation_profile, gene_id_ppi, gene_id_patient))

        (ppi_total, mut_total, ppi_filt, mut_filt) = (
            formatting_data.all_genes_in_submatrices(
                network, idx_ppi, idx_mut, idx_ppi_only, idx_mut_only,
                mutation_profile))

        # ------------ filtering_diffusion.py ------------
        print("------------ filtering_diffusion.py ------------ {}"
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
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
            mut_final, ppi_filt, result_folder, alpha, tol, qn)

        # ------------ clustering.py ------------
        print("------------ clustering.py ------------ {}"
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        genes_clustering, patients_clustering = (clustering.bootstrap(
            result_folder, mut_type, mut_propag, ppi_final,
            influence_weight, simplification,
            alpha, tol, keep_singletons, ngh_max, min_mutation, max_mutation,
            n_components, n_permutations,
            run_bootstrap, lambd, tol_nmf, compute_gene_clustering))

        distance_genes, distance_patients = clustering.consensus_clustering(
            result_folder, genes_clustering, patients_clustering,
            influence_weight, simplification, mut_type,
            alpha, tol, keep_singletons, ngh_max, min_mutation, max_mutation,
            n_components, n_permutations, run_consensus, lambd, tol_nmf,
            compute_gene_clustering)

        # ------------ hierarchical_clustering.py ------------
        print("------------ hierarchical_clustering.py ------------ {}"
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

        hierarchical_clustering.distance_patients_from_consensus_file(
            result_folder, distance_patients, ppi_data, mut_type,
            influence_weight, simplification, alpha, tol,  keep_singletons,
            ngh_max, min_mutation, max_mutation, n_components, n_permutations,
            lambd, tol_nmf, linkage_method, patient_data, data_folder, ssc_subgroups, ssc_mutation_data, gene_data)

        (total_cluster_list, probands_cluster_list, siblings_cluster_list,
                male_cluster_list, female_cluster_list, iq_cluster_list,
                distCEU_list, mutation_nb_cluster_list,
                text_file) = hierarchical_clustering.get_lists_from_clusters(
                    data_folder, patient_data, ssc_mutation_data,
                    ssc_subgroups, ppi_data, gene_data, result_folder,
                    mut_type, influence_weight, simplification, alpha, tol,
                    keep_singletons, ngh_max, min_mutation, max_mutation,
                    n_components, n_permutations, lambd, tol_nmf,
                    linkage_method)

        hierarchical_clustering.bio_statistics(
            n_components, total_cluster_list, probands_cluster_list,
            siblings_cluster_list, male_cluster_list, female_cluster_list,
            iq_cluster_list, distCEU_list, mutation_nb_cluster_list, text_file)


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
          .format(datetime.timedelta(seconds=end_all - start_all),
                  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
