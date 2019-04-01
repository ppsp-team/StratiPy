#!/usr/bin/env python
# coding: utf-8
import sys
import os
sys.path.append(os.path.abspath('../../stratipy'))
from sklearn.model_selection import ParameterGrid


# TODO PPI type param
param_grid = {'data_folder': ['../data/'],
              'patient_data': ['SSC'],
              'ssc_mutation_data': ['MAF50_LoF_mis15'],
              # 'ssc_subgroups': ['SSC1_probands', 'SSC1_siblings',
              #                   'SSC2_probands', 'SSC2_siblings'],
              'ssc_subgroups': ['SSC1', 'SSC2'],
              # 'ssc_subgroups': ['SSC1'],
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
              'alpha': [0.3],
              'tol': [10e-3],
              'ngh_max': [11],
              'keep_singletons': [False],
              'min_mutation': [0],
              'max_mutation': [20000],
              'mut_type': ['raw', 'diff', 'mean_qn', 'median_qn'],
              # 'n_components': [2],
              'n_components': range(2, 51),
            #   'n_permutations': [1000],
              'n_permutations': [300],
              'sub_perm': [1],
              'run_bootstrap': ['split'],
              # 'run_bootstrap': ['full', 'split', None],
              'run_consensus': [True],
              'lambd': [0],
              # 'lambd': [200],
              # 'lambd': [0, 200],
              'tol_nmf': [1e-3],
              'compute_gene_clustering': [True],
              'linkage_method': ['average'],
              'p_val_threshold': [0.05]
              }


def get_params(i):
    d = list(ParameterGrid(param_grid))[i]
    print("=== parameters")
    print(sorted(d.items()))

    data_folder = d.get('data_folder')
    patient_data = d.get('patient_data')
    ssc_mutation_data = d.get('ssc_mutation_data')
    ssc_subgroups = d.get('ssc_subgroups')
    gene_data = d.get('gene_data')
    ppi_data = d.get('ppi_data')
    influence_weight = d.get('influence_weight')
    simplification = d.get('simplification')
    compute = d.get('compute')
    overwrite = d.get('overwrite')

    alpha = d.get('alpha')
    tol = d.get('tol')
    ngh_max = d.get('ngh_max')
    keep_singletons = d.get('keep_singletons')
    min_mutation = d.get('min_mutation')
    max_mutation = d.get('max_mutation')
    mut_type = d.get('mut_type')
    # qn = d.get('qn')
    n_components = d.get('n_components')
    n_permutations = d.get('n_permutations')
    sub_perm = d.get('sub_perm')
    run_bootstrap = d.get('run_bootstrap')
    run_consensus = d.get('run_consensus')
    lambd = d.get('lambd')
    tol_nmf = d.get('tol_nmf')
    compute_gene_clustering = d.get('compute_gene_clustering')
    linkage_method = d.get('linkage_method')
    p_val_threshold = d.get('p_val_threshold')

    n_sub_perm, rest = divmod(n_permutations, sub_perm)
    if rest != 0:
        print('Total permutation number of bootstrap must be divisible by sub-permutation number.')

    return (data_folder, patient_data, ssc_mutation_data, ssc_subgroups,
            gene_data, ppi_data, influence_weight, simplification, compute,
            overwrite, alpha, tol, ngh_max, keep_singletons, min_mutation,
            max_mutation, mut_type, n_components, n_permutations, sub_perm, sub_perm,
            run_bootstrap, run_consensus, lambd, tol_nmf,
            compute_gene_clustering, linkage_method, p_val_threshold)
