#!/usr/bin/env python
# coding: utf-8
import sys
import time
from confusion_matrices import get_cluster_idx, repro_confusion_matrix
from nbs_functions import all_functions
from confusion_matrices import get_cluster_idx, repro_confusion_matrix
from sklearn.model_selection import ParameterGrid
import datetime


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
    Desired number of subgroups (clusters).

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
            #   'n_permutations': [1000],
              'run_bootstrap': [True],
              'run_consensus': [True],
              'lambd': [1, 1800],
              'tol_nmf': [1e-3],
              'linkage_method': ['average']
              }

start_all = time.time()

for params in list(ParameterGrid(param_grid)):
    all_functions(**params)

end_all = time.time()
print('\n---------- ALL = {} ---------- {}\n\n'
      .format(datetime.timedelta(seconds=end_all - start_all),
              datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


print('\n\n######################## Starting Confusion Matrices ########################')

print("\n------------ confusion_matrices.py ------------ {}"
      .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

result_folder_repro = 'reproducibility_data/'

nbs_100 = get_cluster_idx(result_folder_repro, method='nbs',
                          n_permutations=100)
stp_100_lamb1 = get_cluster_idx(result_folder_repro, method='stratipy',
                                n_permutations=100, lambd=1)
stp_100_lamb1800 = get_cluster_idx(result_folder_repro, method='stratipy',
                                   n_permutations=100, lambd=1800)

repro_confusion_matrix(result_folder_repro, nbs_100, stp_100_lamb1,
                       'Confusion matrix\nwith reported tuning parameter value',
                       'lambda=1')
repro_confusion_matrix(result_folder_repro, nbs_100, stp_100_lamb1800,
                       'Confusion matrix\nwith actually used tuning parameter value',
                       'lambda=1800')
#
# confusion_matrices.reproducibility_confusion_matrices('reproducibility_data/',
#                                                       'lambda=1', 'lambda=1800')
