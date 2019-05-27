#!/usr/bin/env python
# coding: utf-8
import sys
import os
# sys.path.append(os.path.abspath('../../stratipy'))
# from stratipy import (load_data, formatting_data, filtering_diffusion,
#                       nmf_bootstrap, consensus_clustering,
#                       hierarchical_clustering, biostat, biostat_go,
#                       biostat_plot, parameters)
import importlib  # NOTE for python >= Python3.4
import scipy.sparse as sp
import numpy as np
import time
import datetime
from sklearn.model_selection import ParameterGrid
from scipy.io import loadmat, savemat
from tqdm import tqdm

# from memory_profiler import profile
# if "from memory_profiler import profile", timestamps will not be recorded

# importlib.reload(load_data)
# print(dir(load_data))

# TODO PPI type param
param_grid = {'data_folder': ['../data/'],
              'patient_data': ['SSC'],
              'ssc_mutation_data': ['MAF1_LoF_mis15'],
              'ssc_subgroups': ['SSC'],
              # 'ssc_subgroups': ['SSC1', 'SSC2'],
              # 'ssc_subgroups': ['SSC', 'SSC1', 'SSC2'],
              # 'gene_data': ['all', 'pli', 'sfari', 'brain1SD', 'brain2SD'],
              'gene_data': ['all', 'BrainSfariPli'],
              'ppi_data': ['APID', 'STRING'],
              # 'ppi_data': ['STRING'],
              'influence_weight': ['min'],
              'simplification': [True],
              'compute': [True],
              'overwrite': [False],
              # 'alpha': [0.7],
              'tol': [10e-3],
              'ngh_max': [11],
              'keep_singletons': [False],
              'min_mutation': [0],
              # 'max_mutation': [20000],
              'max_mutation': [2000],
              'mut_type': ['raw', 'propagated', 'mean_qn', 'median_qn'],
              'n_components': [2],
              # 'n_components': range(2, 21),
            #   'n_permutations': [1000],
              'n_permutations': [300],
              'sub_perm': [1],
              'run_bootstrap': ['split'],
              # 'run_bootstrap': ['full', 'split', None],
              'run_consensus': [True],
              # 'lambd': [0],
              'lambd': [0],
              # 'lambd': [0, 200],
              'tol_nmf': [1e-3],
              'compute_gene_clustering': [True],
              'linkage_method': ['average'],
              'p_val_threshold': [0.05]
              }
# 'lambd': range(0, 2)

# NOTE sys.stdout.flush()


def preprocessing(data_folder, patient_data, ssc_mutation_data, ssc_subgroups, gene_data,
                  ppi_data, result_folder, influence_weight, simplification,
                  compute, overwrite, alpha, tol, ngh_max,
                  keep_singletons, min_mutation, max_mutation, mut_type):
    print("------------ load_data.py ------------ {}"
          .format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
          flush=True)
    import load_data

    if patient_data == 'TCGA_UCEC':
        (patient_id, mutation_profile, gene_id_patient, gene_symbol_profile) = (
            load_data.load_TCGA_UCEC_patient_data(data_folder))
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

    print("------------ formatting_data.py ------------ {}"
          .format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
          flush=True)
    import formatting_data

    idx_ppi, idx_ppi_only, ppi_total, mut_total, ppi_filt = (
        formatting_data.formatting(
            network, mutation_profile, gene_id_ppi, gene_id_patient))

    print("------------ filtering_diffusion.py ------------ {}"
          .format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
          flush=True)
    import filtering_diffusion

    ppi_final, mut_propag = (
        filtering_diffusion.filtering(
            ppi_filt, result_folder, influence_weight, simplification, compute,
            overwrite, alpha, tol, ppi_total, mut_total, ngh_max,
            keep_singletons, min_mutation, max_mutation, mut_type))

    return gene_id_ppi, idx_ppi, idx_ppi_only


# @profile
def all_functions(params):
    if patient_data == 'SSC':
        if mut_type == 'raw':
            alpha = 0
            result_folder = (
            data_folder + 'result_' + ssc_mutation_data + '_' +
            ssc_subgroups + '_' + gene_data +  '/' + mut_type + '/')
        else:
            result_folder = (
                data_folder + 'result_' + ssc_mutation_data + '_' +
                ssc_subgroups + '_' + gene_data + '_' + ppi_data + '/')
        # result_folder = (
        #     '/Volumes/Abu3/min/201809_sfari_without_category6_NaN/result_' + ssc_mutation_data + '_' +
        #     ssc_subgroups + '_' + gene_data + '_' + ppi_data + '/')
    else:
        result_folder = (data_folder + 'result_' + patient_data + '_' +
                         ppi_data + '/')
        if mut_type == 'raw':
            alpha = 0

    print(result_folder, flush=True)
    print("mutation type =", mut_type, flush=True)
    print("alpha =", alpha, flush=True)
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

    print("\n------------ biostat.py ------------ {}"
          .format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
          flush=True)
    import biostat

    gene_id_ppi, idx_ppi, idx_ppi_only = preprocessing(
        data_folder, patient_data, ssc_mutation_data, ssc_subgroups, gene_data,
        ppi_data, result_folder, influence_weight, simplification, compute,
        overwrite, alpha, tol, ngh_max, keep_singletons, min_mutation,
        max_mutation, mut_type)

    biostat.biostat_analysis(
        data_folder, result_folder, patient_data, ssc_mutation_data,
        ssc_subgroups, ppi_data, gene_data, mut_type, influence_weight,
        simplification, alpha, tol, keep_singletons, ngh_max, min_mutation,
        max_mutation, n_components, n_permutations, lambd, tol_nmf,
        linkage_method, p_val_threshold, gene_id_ppi, idx_ppi, idx_ppi_only)

    biostat_go.biostat_go_enrichment(
        alpha, result_folder, mut_type, patient_data, data_folder, ssc_mutation_data,
        ssc_subgroups, gene_data, ppi_data, lambd, n_components, ngh_max, n_permutations)

    # print("\n------------ biostat_plot.py ------------ {}"
    #       .format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
    #       flush=True)
    # # no need SSC1/SSC2, no need k
    # import biostat_plot
    # biostat_plot.load_plot_biostat_individuals(
    #     result_folder, data_folder, ssc_mutation_data,
    #     gene_data, patient_data, ppi_data, mut_type, lambd, influence_weight,
    #     simplification, alpha, tol, keep_singletons, ngh_max, min_mutation,
    #     max_mutation, n_components, n_permutations, tol_nmf, linkage_method)

if (sys.version_info < (3, 2)):
    raise "Must be using Python ≥ 3.2"

else:
    # start_all = time.time()
    # tqdm_bar = trange(list(ParameterGrid(param_grid)))
    for params in tqdm(list(ParameterGrid(param_grid))):
        # start = time.time()
        # print(params)
        for i in params.keys():
            exec("%s = %s" % (i, 'params[i]'))
        all_functions(params)
        # end = time.time()
        # print('---------- ONE STEP = {} ---------- {}'
        #       .format(datetime.timedelta(seconds=end-start),
        #               datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    # end_all = time.time()
    # print('---------- ALL = {} ---------- {}'
    #       .format(datetime.timedelta(seconds=end_all - start_all),
    #               datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
