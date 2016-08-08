#!/usr/bin/env python
# coding: utf-8
import sys
import importlib  # NOTE for python >= Python3.4
import load_data
import formatting_data
import filtering_diffusion
import clustering
import scipy.sparse as sp
import numpy as np
import time
import datetime
from sklearn.grid_search import ParameterGrid

i = int(sys.argv[1])-1

# TODO PPI type param
param_grid = {'data_folder': ['../data/'],
              'patient_data': ['TCGA_UCEC'],
            #   'patient_data': ['TCGA_UCEC', 'SIMONS'],
            #   'ppi_data': ['STRING', 'Y2H'],
              'ppi_data': ['Y2H'],
            #   'ppi_data': ['String', 'Y2H', '1_ppi', '2_ppi', '3_signal', '4_coexpr', '5_cancer', '6_homology'],
              'compute': [True],
              'overwrite': [False],
              'alpha': [0,0.7],
              'tol': [10e-6],
              'ngh_max': [11],
              'keep_singletons': [False],
              'min_mutation': [10],
              'max_mutation': [2000],
              'qn': [None, 'mean', 'median'],
            #   'n_components': [3],
              'n_components': range(1, 21),
              'n_permutations': [1000],
              'run_bootstrap': [True],
              'run_consensus': [True],
              'lambd': [0, 1, 200],
              'tol_nmf': [1e-3]
              }

# 'lambd': range(0, 2)

# NOTE sys.stdout.flush()
def all_functions(params):

    if alpha == 0 and qn is not None:
        print('############ PASS ############')
        pass

    else:
        result_folder = data_folder + 'result_' + patient_data + '_' + ppi_data + '/'
        print(result_folder)

        # ------------ load_data.py ------------
        print("------------ load_data.py ------------")
        (patient_id, mutation_profile, gene_id_patient, gene_symbol_profile
         ) = load_data.load_TCGA_UCEC_patient_data(data_folder)

        if ppi_data == 'STRING':
            gene_id_ppi, network = load_data.load_PPI_String(
                data_folder, ppi_data)

        elif ppi_data == 'Y2H':
            gene_id_ppi, network = load_data.load_PPI_Y2H(
                data_folder, ppi_data)

        # ------------ formatting_data.py ------------
        print("------------ formatting_data.py ------------")
        (network, mutation_profile,
         idx_ppi, idx_mut, idx_ppi_only, idx_mut_only) = (
            formatting_data.classify_gene_index(
                network, mutation_profile, gene_id_ppi, gene_id_patient))

        (ppi_total, mut_total, ppi_filt, mut_filt) = (
            formatting_data.all_genes_in_submatrices(
                network, idx_ppi, idx_mut, idx_ppi_only, idx_mut_only,
                mutation_profile))

        # ------------ filtering_diffusion.py ------------
        print("------------ filtering_diffusion.py ------------")
        ppi_influence = (
            filtering_diffusion.calcul_ppi_influence(
                sp.eye(ppi_filt.shape[0]), ppi_filt,
                result_folder, compute, overwrite, alpha, tol))

        ppi_final, mut_final = filtering_diffusion.filter_ppi_patients(
            ppi_total, mut_total, ppi_filt, ppi_influence, ngh_max,
            keep_singletons, min_mutation, max_mutation)

        mut_type, mut_propag = filtering_diffusion.propagation_profile(
            mut_final, ppi_filt, alpha, tol, qn)

        # ------------ clustering.py ------------
        print("------------ clustering.py ------------")
        sys.stdout.flush()
        genes_clustering, patients_clustering = (clustering.bootstrap(
            result_folder, mut_type, mut_propag, ppi_final,
            alpha, tol, ngh_max, min_mutation, max_mutation,
            n_components, n_permutations,
            run_bootstrap, lambd, tol_nmf))

        distance_genes, distance_patients = clustering.consensus_clustering(
            result_folder, genes_clustering, patients_clustering, mut_type,
            alpha, tol, ngh_max, min_mutation, max_mutation,
            n_components, n_permutations, run_consensus, lambd, tol_nmf)


start = time.time()

params = list(ParameterGrid(param_grid))
print(params[i])

for k in params[i].keys():
    exec("%s = %s" % (k, 'params[i][k]'))

all_functions(params[i])

end = time.time()
print('---------- ONE STEP = {} ---------- {}'
      .format(datetime.timedelta(seconds=end-start),
              datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
