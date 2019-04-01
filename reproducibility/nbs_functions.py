import sys
import os
import confusion_matrices
sys.path.append(os.path.dirname(os.path.abspath('.')))
from stratipy import (load_data, formatting_data, filtering_diffusion, nmf_bootstrap, consensus_clustering, hierarchical_clustering)
import scipy.sparse as sp
from scipy.io import loadmat, savemat
import numpy as np
import time, datetime
# from sklearn.model_selection import ParameterGrid

# NOTE sys.stdout.flush()

# Reproducibility
np.random.seed(42)

# @profile
def all_functions(data_folder, patient_data, ppi_data, influence_weight, simplification,
                 compute, overwrite, alpha, tol, ngh_max, keep_singletons, min_mutation,
                 max_mutation, n_components, n_permutations, run_bootstrap, run_consensus,
                 lambd, tol_nmf, linkage_method, mut_type, compute_gene_clustering,
                 sub_perm):
    if (sys.version_info < (3, 2)):
        raise "Must be using Python ≥ 3.2"

    else:
        start = time.time()

        if mut_type == 'raw':
            alpha = 0

        result_folder = 'reproducibility_output/' + 'result_' + patient_data + '_' + ppi_data + '/'
        print('\n######################## Starting StratiPy ########################')
        print("\nGraph regulator factor (lambda) =", lambd)
        print("Permutation number of bootstrap =", n_permutations)

        print("\n------------ load_data.py ------------ {}"
                .format(datetime.datetime.now()
                        .strftime("%Y-%m-%d %H:%M:%S")))

        (patient_id, mutation_profile, gene_id_patient, gene_symbol_profile) = (
            load_data.load_TCGA_UCEC_patient_data(data_folder))

        gene_id_ppi, network = load_data.load_Hofree_PPI_String(
            data_folder, ppi_data)

        print("\n------------ formatting_data.py ------------ {}"
                .format(datetime.datetime.now()
                        .strftime("%Y-%m-%d %H:%M:%S")))

        idx_ppi, idx_ppi_only, ppi_total, mut_total, ppi_filt = (
            formatting_data.formatting(
                network, mutation_profile, gene_id_ppi, gene_id_patient))

        print("\n------------ filtering_diffusion.py ------------ {}"
                .format(datetime.datetime.now()
                        .strftime("%Y-%m-%d %H:%M:%S")))

        ppi_final, mut_propag = (
            filtering_diffusion.filtering(
                    ppi_filt, result_folder, influence_weight, simplification, compute,
                    overwrite, alpha, tol, ppi_total, mut_total, ngh_max,
                    keep_singletons, min_mutation, max_mutation, mut_type))
        
        print("\n------------ Bootstrap ------------ {}"
                .format(datetime.datetime.now()
                        .strftime("%Y-%m-%d %H:%M:%S")))
        final_influence_mutation_directory = result_folder + 'final_influence/'
        final_influence_mutation_file = (
            final_influence_mutation_directory +
            'final_influence_mutation_profile_{}_alpha={}_tol={}.mat'.format(
                mut_type, alpha, tol))
        final_influence_data = loadmat(final_influence_mutation_file)
        mut_propag = final_influence_data['mut_propag']

        ppi_final_file = (
            final_influence_mutation_directory +
            'PPI_final_weight={}_simp={}_alpha={}_tol={}_singletons={}_ngh={}.mat'
            .format(influence_weight, simplification, alpha, tol, keep_singletons,
                    ngh_max))
        ppi_final_data = loadmat(ppi_final_file)
        ppi_final = ppi_final_data['ppi_final']

        nmf_bootstrap.bootstrap(
            result_folder, mut_type, mut_propag, ppi_final,
            influence_weight, simplification, alpha, tol, keep_singletons,
            ngh_max, min_mutation, max_mutation, n_components,
            n_permutations, run_bootstrap, lambd, tol_nmf,
            compute_gene_clustering, sub_perm)


        print("------------ consensus_clustering.py ------------ {}"
            .format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            flush=True)
        distance_genes, distance_patients = (
            consensus_clustering.sub_consensus(
                result_folder, mut_type, influence_weight, simplification, alpha,
                tol, keep_singletons, ngh_max, min_mutation, max_mutation,
                n_components, n_permutations, lambd, tol_nmf,
                compute_gene_clustering, run_consensus))
        
        print("------------ hierarchical_clustering.py ------------ {}"
            .format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            flush=True)
        hierarchical_clustering_file = hierarchical_clustering.hierarchical_file(
            result_folder, mut_type, influence_weight, simplification, alpha, tol,
            keep_singletons, ngh_max, min_mutation, max_mutation, n_components,
            n_permutations, lambd, tol_nmf, linkage_method)
        
        hierarchical_clustering.individual_linkage_dendrogram(hierarchical_clustering_file,
                       distance_patients, ppi_data, mut_type, alpha, ngh_max,
                       n_components, n_permutations, lambd, linkage_method,
                       patient_data, data_folder, result_folder, repro=True)

        end = time.time()
        print('\n------------ END: One Step of StratiPy = {} ------------ {}\n\n'
                .format(datetime.timedelta(seconds=end-start),
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
