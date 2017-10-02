import sys
import os
import confusion_matrices
sys.path.append(os.path.dirname(os.path.abspath('.')))
from stratipy import load_data, formatting_data, filtering_diffusion, clustering, hierarchical_clustering
import scipy.sparse as sp
from scipy.io import loadmat, savemat
import numpy as np
import time, datetime
# from sklearn.model_selection import ParameterGrid

# NOTE sys.stdout.flush()

# @profile
def all_functions(data_folder, patient_data, ppi_data, influence_weight, simplification,
                 compute, overwrite, alpha, tol, ngh_max, keep_singletons, min_mutation,
                 max_mutation, qn, n_components, n_permutations, run_bootstrap, run_consensus,
                 lambd, tol_nmf, linkage_method):
    if (sys.version_info < (3, 2)):
        raise "Must be using Python ≥ 3.2"

    else:
        start = time.time()
        if alpha == 0 and qn is not None:
            print('######################## PASS ########################')
            pass

        else:
            result_folder = 'reproducibility_data/' + 'result_' + patient_data + '_' + ppi_data + '/'
            print('\n######################## Starting StratiPy ########################')
            print("\nGraph regulator factor (lambda) =", lambd)
            print("Permutation number of bootstrap =", n_permutations)

            print("\n------------ load_data.py ------------ {}"
                  .format(datetime.datetime.now()
                          .strftime("%Y-%m-%d %H:%M:%S")))

            (patient_id, mutation_profile, gene_id_patient,
             gene_symbol_profile) = load_data.load_TCGA_UCEC_patient_data(
                 data_folder)

            gene_id_ppi, network = load_data.load_PPI_String(
                data_folder, ppi_data)

            print("\n------------ formatting_data.py ------------ {}"
                  .format(datetime.datetime.now()
                          .strftime("%Y-%m-%d %H:%M:%S")))
            (network, mutation_profile,
             idx_ppi, idx_mut, idx_ppi_only, idx_mut_only) = (
                formatting_data.classify_gene_index(
                    network, mutation_profile, gene_id_ppi, gene_id_patient))

            (ppi_total, mut_total, ppi_filt, mut_filt) = (
                formatting_data.all_genes_in_submatrices(
                    network, idx_ppi, idx_mut, idx_ppi_only, idx_mut_only,
                    mutation_profile))

            print("\n------------ filtering_diffusion.py ------------ {}"
                  .format(datetime.datetime.now()
                          .strftime("%Y-%m-%d %H:%M:%S")))
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
