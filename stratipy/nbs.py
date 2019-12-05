#!/usr/bin/env python
# coding: utf-8
import sys
import os
sys.path.append(os.path.abspath('../../stratipy'))
from stratipy import (load_data, formatting_data, filtering_diffusion,
                      nmf_bootstrap, consensus_clustering,
                      hierarchical_clustering, biostat, biostat_go,
                      biostat_plot, parameters)
import importlib  # NOTE for python >= Python3.4
import scipy.sparse as sp
import numpy as np
import time
import datetime
from scipy.io import loadmat, savemat


def initiation(mut_type, patient_data, data_folder, ssc_mutation_data,
               ssc_subgroups, gene_data, ppi_data, lambd, n_components):
    if patient_data == 'SSC':
        if mut_type == 'raw':
            alpha = 0
            ppi_data = 'noPPI'

        else:
            if ppi_data == 'APID':
                alpha = 0.6

            elif ppi_data == 'STRING':
                alpha = 0.581

        result_folder = (
            data_folder + 'result_' + ssc_mutation_data + '_' +
            ssc_subgroups + '_' + gene_data + '_' + ppi_data + '/')
        # result_folder = (
        #     data_folder + '/Volumes/Abu3/min/201812_MAF50_alpha0.7/result_' + ssc_mutation_data + '_' +
        #     ssc_subgroups + '_' + gene_data + '_' + ppi_data + '/')
    else:
        result_folder = (data_folder + 'result_' + patient_data + '_' +
                         ppi_data + '/')
        if mut_type == 'raw':
            alpha = 0

    print(result_folder, flush=True)
    print("\nIndividuals =", ssc_subgroups, flush=True)
    print("mutation type =", mut_type, flush=True)
    print("alpha =", alpha, flush=True)
    print("lambda =", lambd, flush=True)
    print("k =", n_components, flush=True)

    return alpha, result_folder, ppi_data


def preprocess_noRaw(ppi_data, mut_type, ssc_subgroups, data_folder, patient_data, ssc_mutation_data,
               gene_data, influence_weight, simplification, compute, overwrite, tol, ngh_max,
               keep_singletons, min_mutation, max_mutation, result_folder, alpha):
    print("------------ load_data.py ------------ {}"
          .format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
          flush=True)
    if patient_data == 'TCGA_UCEC':
        (individual_id, mutation_profile, gene_id, gene_symbol_profile) = (
            load_data.load_TCGA_UCEC_patient_data(data_folder))
    elif patient_data == 'Faroe':
        mutation_profile, gene_id = (
            load_data.load_Faroe_Islands_data(data_folder))
    elif patient_data == 'SSC':
        mutation_profile, gene_id, individual_id = (
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
    idx_ppi, idx_ppi_only, ppi_total, mut_total, ppi_filt = (
        formatting_data.formatting(
            network, mutation_profile, gene_id_ppi, gene_id))

    # EntrezGene ID to int
    entrez_ppi = [int(i) for i in gene_id_ppi]
    # EntrezGene indexes in PPI after formatting
    idx_filtred = idx_ppi + idx_ppi_only

    print("------------ filtering_diffusion.py ------------ {}"
          .format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
          flush=True)
    ppi_final, mut_propag = (
        filtering_diffusion.filtering(
            ppi_filt, result_folder, influence_weight, simplification, compute,
            overwrite, alpha, tol, ppi_total, mut_total, ngh_max,
            keep_singletons, min_mutation, max_mutation, mut_type))

    return gene_id, individual_id, entrez_ppi, idx_filtred, mut_propag


def preprocessing(ppi_data, mut_type, ssc_subgroups, data_folder, patient_data,
                  ssc_mutation_data, gene_data, influence_weight,
                  simplification, compute, overwrite, tol, ngh_max,
                  keep_singletons, min_mutation, max_mutation, result_folder, alpha, return_val=False):
    if mut_type == 'raw':
        mutation_profile, mp_gene, mp_indiv = (
            load_data.load_specific_SSC_mutation_profile(
                data_folder, ssc_mutation_data, ssc_subgroups, gene_data))
    else:
        gene_id, mp_indiv, entrez_ppi, idx_filtred, mutation_profile = preprocess_noRaw(
            ppi_data, mut_type, ssc_subgroups, data_folder, patient_data, ssc_mutation_data,
                           gene_data, influence_weight, simplification, compute, overwrite, tol, ngh_max,
                           keep_singletons, min_mutation, max_mutation, result_folder, alpha)
        # Entrez Gene ID in filtered/formatted mutation profile
        mp_gene = [entrez_ppi[i] for i in idx_filtred]

    if return_val:
        return mutation_profile, mp_indiv, mp_gene, entrez_ppi, idx_filtred


def parallel_bootstrap(result_folder, mut_type, influence_weight,
                       simplification, alpha, tol, keep_singletons,
                       ngh_max, min_mutation, max_mutation, n_components,
                       n_permutations, run_bootstrap, lambd, tol_nmf,
                       compute_gene_clustering, sub_perm, data_folder, ssc_mutation_data, ssc_subgroups, gene_data):
    # if mut_type == 'raw':
    #     mut_propag, mp_gene, mp_indiv = (
    #         load_data.load_specific_SSC_mutation_profile(
    #             data_folder, ssc_mutation_data, ssc_subgroups, gene_data))
    #     # fake
    #     ppi_final_file = '../data/result_MAF1_LoF_mis15_SSC_all_allGenes_APID/final_influence/PPI_final_weight=min_simp=True_alpha=0.7_tol=0.01_singletons=False_ngh=11.mat'
    #
    # else:
    #     final_influence_mutation_directory = result_folder + 'final_influence/'
    #     final_influence_mutation_file = (
    #         final_influence_mutation_directory +
    #         'final_influence_mutation_profile_{}_alpha={}_tol={}.mat'.format(
    #             mut_type, alpha, tol))
    #     final_influence_data = loadmat(final_influence_mutation_file)
    #     mut_propag = final_influence_data['mut_propag']
    #
    #     ppi_final_file = (
    #         final_influence_mutation_directory +
    #         'PPI_final_weight={}_simp={}_alpha={}_tol={}_singletons={}_ngh={}.mat'
    #         .format(influence_weight, simplification, alpha, tol, keep_singletons,
    #                 ngh_max))
    # ppi_final_data = loadmat(ppi_final_file)
    # ppi_final = ppi_final_data['ppi_final']
    #
    # nmf_bootstrap.bootstrap(
    #     result_folder, mut_type, mut_propag, ppi_final,
    #     influence_weight, simplification, alpha, tol, keep_singletons,
    #     ngh_max, min_mutation, max_mutation, n_components,
    #     n_permutations, run_bootstrap, lambd, tol_nmf,
    #     compute_gene_clustering, sub_perm)

    nmf_bootstrap.bootstrap(data_folder, ssc_mutation_data, ssc_subgroups, gene_data,
                  result_folder, mut_type, influence_weight,
                  simplification, alpha, tol, keep_singletons, ngh_max,
                  min_mutation, max_mutation, n_components, n_permutations,
                  run_bootstrap, lambd, tol_nmf,
                  compute_gene_clustering, sub_perm)


def post_bootstrap(result_folder, mut_type, influence_weight, simplification,
                   alpha, tol, keep_singletons, ngh_max, min_mutation,
                   max_mutation, n_components, n_permutations, lambd, tol_nmf,
                   compute_gene_clustering, run_consensus, run_bootstrap,
                   linkage_method,
                   ppi_data, patient_data, data_folder, ssc_subgroups,
                   ssc_mutation_data, gene_data, p_val_threshold, compute,
                   overwrite):
    # print("------------ consensus_clustering.py ------------ {}"
    #       .format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
    #       flush=True)
    # distance_genes, distance_patients = (
    #     consensus_clustering.sub_consensus(
    #         result_folder, mut_type, influence_weight, simplification, alpha,
    #         tol, keep_singletons, ngh_max, min_mutation, max_mutation,
    #         n_components, n_permutations, lambd, tol_nmf,
    #         compute_gene_clustering, run_consensus))
    #
    # print("------------ hierarchical_clustering.py ------------ {}"
    #       .format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
    #       flush=True)
    # hierarchical_clustering.hierarchical(
    #     result_folder, distance_genes, distance_patients, ppi_data, mut_type,
    #     influence_weight, simplification, alpha, tol, keep_singletons, ngh_max,
    #     min_mutation, max_mutation, n_components, n_permutations, lambd,
    #     tol_nmf, linkage_method, patient_data, data_folder, ssc_subgroups,
    #     ssc_mutation_data, gene_data)

    print("\n------------ biostat.py ------------ {}"
          .format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
          flush=True)
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

    # biostat_go.biostat_go_enrichment(
    #     alpha, result_folder, mut_type, patient_data, data_folder, ssc_mutation_data,
    #     ssc_subgroups, gene_data, ppi_data, lambd, n_components, ngh_max, n_permutations)

    print("\n------------ biostat_plot.py ------------ {}"
          .format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
          flush=True)
    # no need SSC1/SSC2, no need k
    # biostat_plot.load_plot_biostat_individuals(
    #     result_folder, data_folder, ssc_mutation_data,
    #     gene_data, patient_data, ppi_data, mut_type, lambd, influence_weight,
    #     simplification, alpha, tol, keep_singletons, ngh_max, min_mutation,
    #     max_mutation, n_components, n_permutations, tol_nmf, linkage_method)
###############################################################################
###############################################################################
###############################################################################
###############################################################################


def all_functions(i, step):
    (data_folder, patient_data, ssc_mutation_data, ssc_subgroups,
     gene_data, ppi_data, influence_weight, simplification, compute,
     overwrite, alpha, tol, ngh_max, keep_singletons, min_mutation,
     max_mutation, mut_type, n_components, n_permutations, sub_perm, sub_perm,
     run_bootstrap, run_consensus, lambd, tol_nmf, compute_gene_clustering,
     linkage_method, p_val_threshold) = parameters.get_params(i)

    alpha, result_folder = initiation(
        mut_type, alpha, patient_data, data_folder, ssc_mutation_data,
        ssc_subgroups, gene_data, ppi_data, lambd, n_components)

    if step == "preprocessing":
        print('\n############ preprocessing step ############', flush=True)
        preprocessing(data_folder, patient_data, ssc_mutation_data, ssc_subgroups,
                      gene_data, ppi_data, result_folder, influence_weight,
                      simplification, compute, overwrite, alpha, tol,
                      ngh_max, keep_singletons, min_mutation, max_mutation,
                      mut_type)

    if step == "parallel_bootstrap":
        print('\n############ parallel_bootstrap step ############',
              flush=True)
        parallel_bootstrap(result_folder, mut_type, influence_weight,
                           simplification, alpha, tol, keep_singletons,
                           ngh_max, min_mutation, max_mutation,
                           n_components, n_permutations, run_bootstrap,
                           lambd, tol_nmf, compute_gene_clustering,
                           sub_perm)

    if step == "clustering":
        print('\n############ clustering step ############', flush=True)
        post_bootstrap(result_folder, mut_type, influence_weight,
                       simplification, alpha, tol, keep_singletons,
                       ngh_max, min_mutation, max_mutation, n_components,
                       n_permutations, lambd, tol_nmf,
                       compute_gene_clustering, run_consensus,
                       linkage_method, ppi_data, patient_data, data_folder,
                       ssc_subgroups, ssc_mutation_data, gene_data,
                       p_val_threshold, compute, overwrite)




        # if step == "all":
        #
        #
        #
        #
        #     consensus_file = consensus_clustering.consensus_file(
        #         result_folder, influence_weight, simplification, mut_type, alpha, tol,
        #         keep_singletons, ngh_max, min_mutation, max_mutation, n_components,
        #         n_permutations, lambd, tol_nmf)
        #
        #     # genes_clustering, patients_clustering directely from full bootstrap
        #     distance_genes, distance_patients = (
        #         consensus_clustering.consensus_from_full_bootstrap(
        #             consensus_file, genes_clustering, patients_clustering,
        #             run_consensus, compute_gene_clustering))


i = int(sys.argv[1])-1
step = sys.argv[2]
all_functions(i, step)
