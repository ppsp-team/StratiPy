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

def all_functions(output_folder, ppi_data, patients, patientsNum, phenotypes,
                 marker_shapes, genesNum, connNeighboors, connProbability,
                 influence_weight, simplification,
                 compute, overwrite, alpha, tol, ngh_max, keep_singletons, min_mutation,
                 max_mutation, qn, n_components, n_permutations, run_bootstrap, run_consensus,
                 lambd, tol_nmf, linkage_method):
    if (sys.version_info < (3, 2)):
        raise "Must be using Python ≥ 3.2"

    else:
        start = time.time()
        result_folder = output_folder + 'nbs/'
        os.makedirs(result_folder, exist_ok=True)
        print('\n######################## Starting StratiPy ########################')
        print("\nGraph regulator factor (lambda) =", lambd)
        print("Permutation number of bootstrap =", n_permutations)

        print("\n------------ generate simulated data ------------ {}"
              .format(datetime.datetime.now()
                      .strftime("%Y-%m-%d %H:%M:%S")))
        PPI = simulation_functions.generate_network(
            pathwaysNum, genesNum, connNeighboors, connProbability,
            marker_shapes)

        for BetweenPathwaysConnection in range(0, pathwaysNum*connBetweenPathways):
            PPI = simulation_functions.addBetweenPathwaysConnection(
                PPI, pathwaysNum, genesNum)

        patients, phenotypes = (simulation_functions.
                                generate_all_mutation_profile(
                                    patientsNum, PPI, genesNum, pathwaysNum,
                                    mutationProb))

        simulation_functions.save_dataset(
            PPI, position, patients, phenotypes, pathwaysNum, genesNum,
            connProbability, connNeighboors, connBetweenPathways, patientsNum,
            mutationProb, output_folder, new_data=True)

        # plot_network_patient(PPI, position, patients, patientsNum, phenotypes,
        #                      marker_shapes, output_folder, plot_name="Raw")

        print("\n------------ loading & formatting data ------------ {}"
              .format(datetime.datetime.now()
                      .strftime("%Y-%m-%d %H:%M:%S")))
        ppi_filt = nx.to_scipy_sparse_matrix(PPI, dtype=np.float32)
        ppi_final = ppi_filt
        mut_final = sp.csr_matrix(patients, dtype=np.float32)


        print("\n------------ filtering_diffusion.py ------------ {}"
              .format(datetime.datetime.now()
                      .strftime("%Y-%m-%d %H:%M:%S")))
        final_influence = (
            filtering_diffusion.calcul_final_influence(
                sp.eye(ppi_filt.shape[0], dtype=np.float32), ppi_filt,
                result_folder, influence_weight, simplification,
                compute, overwrite, alpha, tol))

        # plot_network_patient(nx.from_scipy_sparse_matrix(final_influence,
        #                                                  create_using=nx.watts_strogatz_graph(genesNum, connNeighboors, connProbability)), patients, patientsNum, phenotypes,
        #                      marker_shapes, output_folder, plot_name="Diffused")

        mut_type, mut_propag = filtering_diffusion.propagation_profile(
            mut_final, ppi_filt, result_folder, alpha, tol, qn)

        # from sparse to networkx
        final_ppi = nx.from_scipy_sparse_matrix(
            final_influence, create_using=nx.watts_strogatz_graph(
                genesNum, connNeighboors, connProbability))

        simulation_functions.plot_network_patient(
            mut_type, alpha, tol, final_ppi, position, patients, patientsNum,
            phenotypes, marker_shapes, result_folder)


        print("\n------------ clustering.py ------------ {}"
              .format(datetime.datetime.now()
                      .strftime("%Y-%m-%d %H:%M:%S")))
        genes_clustering, patients_clustering = (clustering.bootstrap(
            result_folder, mut_type, mut_propag, ppi_final,
            influence_weight, simplification,
            alpha, tol, keep_singletons, ngh_max, min_mutation,
            max_mutation, n_components, n_permutations, run_bootstrap,
            lambd, tol_nmf))

        distance_genes, distance_patients = clustering.consensus_clustering(
            result_folder, genes_clustering, patients_clustering,
            influence_weight, simplification, mut_type, alpha, tol,
            keep_singletons, ngh_max, min_mutation, max_mutation,
            n_components, n_permutations, run_consensus, lambd, tol_nmf)

        # ------------ hierarchical_clustering.py ------------
        print("\n------------ hierarchical_clustering.py ------------ {}"
              .format(datetime.datetime.now()
                      .strftime("%Y-%m-%d %H:%M:%S")))
        hierarchical_clustering.distance_patients_from_consensus_file(
            result_folder, distance_patients, ppi_data, mut_type,
            influence_weight, simplification, alpha, tol,  keep_singletons,
            ngh_max, min_mutation, max_mutation, n_components,
            n_permutations, lambd, tol_nmf, linkage_method)

        end = time.time()
        print('\n------------ END: One Step of StratiPy = {} ------------ {}\n\n'
              .format(datetime.timedelta(seconds=end-start),
                      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
