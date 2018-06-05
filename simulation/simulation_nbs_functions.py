#!/usr/bin/env python
# coding: utf-8
import sys
import os
import scipy.sparse as sp
from scipy.io import loadmat, savemat
import numpy as np
import time
import datetime
import networkx as nx
sys.path.append(os.path.dirname(os.path.abspath('.')))
from stratipy import filtering_diffusion, clustering, hierarchical_clustering
# from simulation_functions import generate_network, addBetweenPathwaysConnection, generate_all_mutation_profile, plot_network_patient, save_dataset
import simulation_functions


# def all_functions(output_folder, ppi_data, patientsNum, mutationProb,
#                   marker_shapes, pathwaysNum, genesNum, connNeighboors,
#                   connProbability, connBetweenPathways, position,
#                  influence_weight, simplification,
#                  compute, overwrite, alpha, tol, ngh_max, keep_singletons, min_mutation,
#                  max_mutation, qn, n_components, n_permutations, run_bootstrap, run_consensus,
#                  lambd, tol_nmf, linkage_method):


# pathwaysNum = 6
# genesNum = 12
# connProbability = 0.4
# connNeighboors = 4
# connBetweenPathways = 2
# marker_shapes = ['o', 'p', '<', 'd', 's', '*']
#
# output_folder = output/'
# influence_weight = 'min'
# simplification = True
# tol = 10e-3
# keep_singletons = False
# min_mutation = 0
# max_mutation = 100
# n_components = 6
# n_permutations = 1000
# tol_nmf = 1e-3
# linkage_method = 'average'
# phenotype_idx = phenotype_idx
#
#
# alpha = 0
# ngh_max = 5 #10
# qn = None #"mean", "median"
# lambd = 0 # 200

#
def all_functions(output_folder, ppi_data, ppi_filt, mut_final,
                  influence_weight, simplification, compute, overwrite, alpha,
                  tol, ngh_max, keep_singletons, min_mutation, max_mutation, qn,
                  n_components, n_permutations, run_bootstrap, run_consensus,
                  lambd, tol_nmf, linkage_method):

    if alpha == 0 and qn is not None:
        print('############ PASS ############')
        pass

    else:
        start = time.time()
        result_folder = output_folder + 'nbs/'
        os.makedirs(result_folder, exist_ok=True)
        print('\n######################## Starting StratiPy ########################')
        print("\nGraph regulator factor (lambda) =", lambd)
        print("Permutation number of bootstrap =", n_permutations)
        print("alpha =", alpha)
        print("QN =", qn)
        print("k =", n_components)


        print("\n------------ filtering_diffusion.py ------------ {}"
              .format(datetime.datetime.now()
                      .strftime("%Y-%m-%d %H:%M:%S")))

        final_influence = (
            filtering_diffusion.calcul_final_influence(
                sp.eye(ppi_filt.shape[0], dtype=np.float32), ppi_filt,
                result_folder, influence_weight, simplification,
                compute, overwrite, alpha, tol))

        # ppi_final, mut_final = filtering_diffusion.filter_ppi_patients(
        #     ppi_total, mut_total, ppi_filt, final_influence, ngh_max,
        #     keep_singletons, min_mutation, max_mutation)

        mut_type, mut_propag = filtering_diffusion.propagation_profile(
            mut_final, ppi_filt, result_folder, alpha, tol, qn)


        return mut_type, mut_propag

        #
        # from sparse to networkx
        # final_ppi = nx.from_scipy_sparse_matrix(
        #     final_influence, create_using=nx.watts_strogatz_graph(
        #         genesNum, connNeighboors, connProbability))
        #
        #
        # simulation_functions.plot_network_patient(
        #     mut_type, alpha, tol, PPI, position, np.array(mut_propag), patientsNum,
        #     phenotypes, marker_shapes, output_folder)
        #
        #
        # print("\n------------ clustering.py ------------ {}"
        #       .format(datetime.datetime.now()
        #               .strftime("%Y-%m-%d %H:%M:%S")))
        # ppi_final = ppi_filt
        # genes_clustering, patients_clustering = (clustering.bootstrap(
        #     result_folder, mut_type, mut_propag, ppi_final,
        #     influence_weight, simplification,
        #     alpha, tol, keep_singletons, ngh_max, min_mutation,
        #     max_mutation, n_components, n_permutations, run_bootstrap,
        #     lambd, tol_nmf))
        #
        # distance_genes, distance_patients = clustering.consensus_clustering(
        #     result_folder, genes_clustering, patients_clustering,
        #     influence_weight, simplification, mut_type, alpha, tol,
        #     keep_singletons, ngh_max, min_mutation, max_mutation,
        #     n_components, n_permutations, run_consensus, lambd, tol_nmf)
        #
        #
        # print("\n------------ hierarchical_clustering.py ------------ {}"
        #       .format(datetime.datetime.now()
        #               .strftime("%Y-%m-%d %H:%M:%S")))
        # hierarchical_clustering.distance_patients_from_consensus_file(
        #     result_folder, distance_patients, ppi_data, mut_type,
        #     influence_weight, simplification, alpha, tol,  keep_singletons,
        #     ngh_max, min_mutation, max_mutation, n_components,
        #     n_permutations, lambd, tol_nmf, linkage_method)
        #
        # print("\n------------ confusion_matrices.py ------------ ")

        #
        # end = time.time()
        # print('\n------------ END: One Step of StratiPy = {} ------------ {}\n\n'
        #       .format(datetime.timedelta(seconds=end-start),
        #               datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
