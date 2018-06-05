#!/usr/bin/env python
# coding: utf-8
import sys
import os
sys.path.append(os.path.abspath('../../stratipy'))
from stratipy import load_data, formatting_data, filtering_diffusion, clustering, hierarchical_clustering
import importlib  # NOTE for python >= Python3.4
import scipy.sparse as sp
import numpy as np
import pandas as pd
import time
import datetime
import readline
from sklearn.model_selection import ParameterGrid
from scipy.io import loadmat, savemat
from tqdm import tqdm


# import rpy2.rinterface
# rpy2.rinterface.set_initoptions(('rpy2', '--verbose', '--no-save'))
# rpy2.rinterface.initr()

# base = importr('base')
# print('=====', base._libPaths())

from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
utils = importr('utils')
# utils.install_packages('diceR')
dicer = importr('diceR')
# dicer = importr('diceR', lib_loc="/mount/gensoft2/exe/R/3.5.0/lib64/R/library")

# PAC threashold
x1 = 0.05
x2 = 0.95

data_folder = '../data/'
patient_data = 'SSC'
ssc_mutation_data = 'LoF_mis30'
ppi_data = 'APID'
influence_weight = 'min'
simplification = True
compute = True
overwrite = False
tol = 10e-3
ngh_max = 11
keep_singletons = False
min_mutation = 0
max_mutation = 2000
n_permutations = 100
run_bootstrap = True
run_consensus = True
tol_nmf = 1e-3
compute_gene_clustering = True
linkage_method = 'average'

param_grid = {'ssc_subgroups': ['SSC1', 'SSC2'],
              'gene_data': ['all', 'pli', 'sfari', 'brain1SD', 'brain2SD'],
              'alpha': [0],
              'qn': [None],
              'n_components': range(2, 11),
              'lambd': [0]}


def create_dataframe_with_pac(params):
        consensus_directory = result_folder+'consensus_clustering/'
        consensus_mut_type_directory = consensus_directory + mut_type + '/'

        if lambd > 0:
            consensus_factorization_directory = (consensus_mut_type_directory + 'gnmf/')

        else:
            consensus_factorization_directory = (consensus_mut_type_directory + 'nmf/')

        consensus_file = (consensus_factorization_directory +
                      'consensus_weight={}_simp={}_alpha={}_tol={}_singletons={}_ngh={}_minMut={}_maxMut={}_comp={}_permut={}_lambd={}_tolNMF={}.mat'
                      .format(influence_weight, simplification, alpha, tol,
                              keep_singletons, ngh_max,
                              min_mutation, max_mutation,
                              n_components, n_permutations, lambd, tol_nmf))

        existance = os.path.exists(consensus_file)
        if existance:
            consensus_data = loadmat(consensus_file)
            distance_ind = consensus_data['distance_patients']
            distance_gene = consensus_data['distance_genes']

            # calculate PAC score using R function
            pac_ind = dicer.PAC(distance_ind, lower=x1, upper=x2)[0]
            # pac_gene = dicer.PAC(distance_gene, lower=x1, upper=x2)[0]
        else:
            pac_ind = np.nan
            # pac_gene = np.nan

        # return pac_ind, pac_gene
        return pac_ind



ssc_sub_list = []
gene_data_list = []
alpha_list = []
mut_list = []
k_list = []
lambd_list = []
ngh_list = []
pac_ind_list = []
# pac_gene_list = []

for params in tqdm(list(ParameterGrid(param_grid))):
    for i in params.keys():
        exec("%s = %s" % (i, 'params[i]'))

    if alpha == 0 and qn is not None:
        pass
    else:
        result_folder = (data_folder + 'result_' + ssc_mutation_data + '_' +
                             ssc_subgroups + '_' + gene_data + '_' +  ppi_data + '/')

        if alpha > 0:
            if qn == 'mean':
                mut_type = 'mean_qn'
            elif qn == 'median':
                mut_type = 'median_qn'
            else:
                mut_type = 'diff'
        else:
            mut_type = 'raw'

        pac_ind = create_dataframe_with_pac(params)
        # pac_ind, pac_gene = create_dataframe_with_pac(params)

        ssc_sub_list.append(ssc_subgroups)
        gene_data_list.append(gene_data)
        alpha_list.append(alpha)
        mut_list.append(mut_type)
        k_list.append(n_components)
        lambd_list.append(lambd)
        ngh_list.append(ngh_max)
        pac_ind_list.append(pac_ind)
        # pac_gene_list.append(pac_gene)

df_pac = pd.DataFrame({
    'ssc_subgroup':ssc_sub_list,
    'gene_data': gene_data_list,
    'alpha': alpha_list,
    'mut_type': mut_list,
    'k': k_list,
    'lambda': lambd_list,
    'neighbors': ngh_list,
    'PAC_individuals': pac_ind_list
    # 'PAC_genes': pac_gene_list
})

df_pac.to_csv(data_folder + 'pac_raw.csv', sep='\t', index=False)
