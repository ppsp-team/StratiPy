#!/usr/bin/env python
# coding: utf-8
import sys
import os
# import os.path
import confusion_matrices
sys.path.append('../stratipy')
import load_data, formatting_data, filtering_diffusion, clustering, hierarchical_clustering
import scipy.sparse as sp
from scipy.io import loadmat, savemat
import numpy as np
import time
import datetime
from sklearn.model_selection import ParameterGrid

"""Tuning parameters' list for Network Based Stratification (NBS)

'param_grid' contains NBS tuning parameters' variable names and its values.


Parameters
----------

data_folder : str
    Path of data folder including mutation profiles, Protein-Protein
    Interaction (PPI) networks and result forders.

patient_data : str
    Raw mutation profile data. Here we work on uterine endometrial carcinoma
    (uterine cancer) with 248 patients' somatic mutation data: TCGA_UCEC.

ppi_data : str
    Protein-Protein Interaction network data. Here we utilize STRING PPI
    network database.

influence_weight : str, 'min' or 'max', default: min
    Choice of influence weight of propagation on the network. For further
    details, see "compare_ij_ji" function in "filtering_diffusion.py".

simplification : boolean, default: True
    Simplification of diffused network matrice after propagation.

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
    For alpha = 1 : complete diffusion.

tol : float, default: 10e-3
    Convergence threshold during diffusion.

ngh_max : int, default: 11
    Number of best influencers in PPI network. 11 is given in the original work.

keep_singletons : boolean, default: False
    If True, proteins not annotated in PPI (genes founded only in patients'
    mutation profiles) will be also considered.
    If False, only annotated proteins in PPI will be considered.

min_mutation, max_mutation : int
    Numbers of lowest mutations and highest mutations per patient. In the
    original work, authors remove patients with fewer than 10 mutations.

qn : str, default: 'mean'
    Type of quantile normalization (QN) after diffusion:
    'None': no normalization.
    'mean': QN based on the mean of the ranked values.
    'median': QN based on the median of the ranked values.

n_components : int, default: 3
    Number of subgroups (clusters) wanted.

n_permutations : int, default: 100
    Permutation number of bootstrap.

run_bootstrap : boolean, default: False
    If True, bootstrap of NMF or GNMF will be launched.

run_consensus : boolean, default: False
    If True, consensus clustering on bootstrap result will be launched. Then,
    the similarity matrix is constructed; each pair of patients was observed to
    share the same membership among all replicates.

lambd : int, default: 1
    Graph regulator factor for GNMF algorithm.

tol_nmf : float, default: 1e-3
    Convergence threshold of NMF and GNMF.

linkage_method : str
    Linkage method of hierarchical clustering.
"""

param_grid = {'data_folder': ['../data/'],
              'patient_data': ['TCGA_UCEC'],
              'ppi_data': ['STRING'],
              'influence_weight': ['min'],
              'simplification': [True],
              'compute': [True],
              'overwrite': [False],
              'alpha': [0.7],
              'tol': [10e-3],
              'ngh_max': [11],
              'keep_singletons': [False],
              'min_mutation': [10],
              'max_mutation': [200000],
              'qn': ["mean"],
              'n_components': [3],
              'n_permutations': [100],
            #   'n_permutations': [100, 1000],
              'run_bootstrap': [True],
              'run_consensus': [True],
              'lambd': [1, 1800],
              'tol_nmf': [1e-3],
              'linkage_method': ['average']
              }

# NOTE sys.stdout.flush()


# @profile
def all_functions(params):

    if alpha == 0 and qn is not None:
        print('######################## PASS ########################')
        pass

    else:
        result_folder = 'reproducibility_data/' + 'result_' + patient_data + '_' + ppi_data + '/'
        print('\n######################## Starting StratiPy ########################')
        print("\nGraph regulator factor (lambda) =", lambd)
        print("Permutation number of bootstrap =", n_permutations)

        print("\n------------ load_data.py ------------ {}"
              .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        (patient_id, mutation_profile, gene_id_patient,
         gene_symbol_profile) = load_data.load_TCGA_UCEC_patient_data(
             data_folder)

        gene_id_ppi, network = load_data.load_PPI_String(data_folder, ppi_data)

        print("\n------------ formatting_data.py ------------ {}"
              .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        (network, mutation_profile,
         idx_ppi, idx_mut, idx_ppi_only, idx_mut_only) = (
            formatting_data.classify_gene_index(
                network, mutation_profile, gene_id_ppi, gene_id_patient))

        (ppi_total, mut_total, ppi_filt, mut_filt) = (
            formatting_data.all_genes_in_submatrices(
                network, idx_ppi, idx_mut, idx_ppi_only, idx_mut_only,
                mutation_profile))

        print("\n------------ filtering_diffusion.py ------------ {}"
              .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
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
              .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
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
        print("\n------------ hierarchical_clustering.py ------------ {}"
              .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
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
        # print(params)
        for i in params.keys():
            exec("%s = %s" % (i, 'params[i]'))
        all_functions(params)
        end = time.time()
        print('\n---------- ONE STEP of StratiPy = {} ---------- {}\n\n'
              .format(datetime.timedelta(seconds=end-start),
                      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    end_all = time.time()
    print('\n---------- ALL = {} ---------- {}\n\n'
          .format(datetime.timedelta(seconds = end_all - start_all),
                  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))



print('\n\n######################## Starting Confusion Matrices ########################')

result_folder = 'reproducibility_data/'

print("\n------------ confusion_matrices.py ------------ {}"
      .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

confusion_matrices.reproducibility_confusion_matrices('lambda=1', 'lambda=1800')
