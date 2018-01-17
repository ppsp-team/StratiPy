#!/usr/bin/env python
# coding: utf-8
import sys
import time
from simulation_functions import patient_data, generate_network, addBetweenPathwaysConnection, generate_all_mutation_profile, plot_network_patient, save_dataset
from simulation_nbs_functions import all_functions
from confusion_matrix import minimal_element_0_to_1, simulated_confusion_matrix
import datetime
from sklearn.model_selection import ParameterGrid
import networkx as nx
import numpy as np
import scipy.sparse as sp
import pickle

if (sys.version_info < (3, 2)):
    raise "Must be using Python ≥ 3.2"

output_folder = "output/"

# Pathways parameters
# pathwaysNum = 6
# genesNum = 12
# connProbability = 0.4
# connNeighboors = 4
# connBetweenPathways = 2
# marker_shapes = ['o', 'p', '<', 'd', 's', '*']

pathwaysNum = 12
genesNum = 24
connProbability = 0.4
connNeighboors = 4
connBetweenPathways = 2

# Simulate patients with a specific mutation profile
patientsNum = 300
mutationProb = 0.2

# load PPI nodes' fixed positions
with open('input/ppi_node_position.txt', 'rb') as handle:
    position = pickle.load(handle)

with open('input/{}_patients.txt'.format(patientsNum), 'rb') as handle:
    load_data = pickle.load(handle)
    patients = load_data['patients']
    phenotypes = load_data['phenotypes']

phenotype_idx = minimal_element_0_to_1(phenotypes)

print("\n------------ generate simulated data ------------ {}"
      .format(datetime.datetime.now()
              .strftime("%Y-%m-%d %H:%M:%S")))
PPI = generate_network(
    pathwaysNum, genesNum, connNeighboors, connProbability, marker_shapes)

for BetweenPathwaysConnection in range(0, pathwaysNum*connBetweenPathways):
    PPI = addBetweenPathwaysConnection(PPI, pathwaysNum, genesNum)

save_dataset(PPI, position, patients, phenotypes, pathwaysNum, genesNum,
             connProbability, connNeighboors, connBetweenPathways, patientsNum,
             mutationProb, output_folder, new_data=True)


print("\n------------ loading & formatting data ------------ {}"
      .format(datetime.datetime.now()
              .strftime("%Y-%m-%d %H:%M:%S")))
ppi_filt = nx.to_scipy_sparse_matrix(PPI, dtype=np.float32)
ppi_final = ppi_filt
mut_final = sp.csr_matrix(patients, dtype=np.float32)


# simulated NBS parameters
param_grid = {'output_folder': ['output/'],
              'ppi_data': ['simulation'],
            #   'ppi_filt': [ppi_filt],
            #   'mut_final': [mut_final],
              'influence_weight': ['min'],
              'simplification': [True],
              'compute': [True],
              'overwrite': [False],
              'alpha': [0, 0.7],
              'tol': [10e-3],
              'ngh_max': [3],
              'keep_singletons': [False],
              'min_mutation': [0],
              'max_mutation': [100],
              'qn': [None, "mean", "median"],
              'n_components': range(2, 13),
              'n_permutations': [1000],
              'run_bootstrap': [True],
              'run_consensus': [True],
              'lambd': [0, 200, 1800],
              'tol_nmf': [1e-3],
              'linkage_method': ['average']
              }


for params in list(ParameterGrid(param_grid)):
    # load PPI nodes' fixed positions
    start_all = time.time()

    for i in params.keys():
        exec("%s = %s" % (i, 'params[i]'))
    # all_functions(**params)

    simulated_confusion_matrix(**params)

    end_all = time.time()
    print('---------- ALL = {} ---------- {}'
          .format(datetime.timedelta(seconds = end_all - start_all),
                  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


param_grid2 = {'output_folder': ['output/'],
              'influence_weight': ['min'],
              'simplification': [True],
              'alpha': [0, 0.7],
              'tol': [10e-3],
              'ngh_max': [3],
              'keep_singletons': [False],
              'min_mutation': [0],
              'max_mutation': [100],
              'qn': [None, "mean", "median"],
              'n_components': range(2, 13),
              'n_permutations': [1000],
              'lambd': [0, 200, 1800],
              'tol_nmf': [1e-3],
              'linkage_method': ['average'],
              'phenotype_idx': [phenotype_idx],
              'pathwaysNum': [pathwaysNum]
              }
for params in list(ParameterGrid(param_grid2)):
    for i in params.keys():
        exec("%s = %s" % (i, 'params[i]'))
    simulated_confusion_matrix(**params)









# plot_network_patient(PPI, position, mut_propag, patientsNum, phenotypes,
#                      marker_shapes, output_folder, plot_name="Diffused")


# from scipy.io import loadmat
# import networkx as nx
# import matplotlib.pyplot as plt
# from stratipy import filtering_diffusion
# import scipy.sparse as sp
# import numpy as np
#
# final_influence_directory = 'simulation/output/nbs/final_influence/'
# final_influence_file = (
#     final_influence_directory +
#     'final_influence_PPI_simp=True_alpha=0.7_tol=0.01.mat')
# final_influence_data = loadmat(final_influence_file)
# final_influence = final_influence_data['final_influence_min']
#
# final_influence.todense().shape
#
# a = nx.from_scipy_sparse_matrix(final_influence)
# a
#
# mut_type, mut_propag = filtering_diffusion.propagation_profile(
#     sp.csr_matrix(patients, dtype=np.float32),
#     nx.to_scipy_sparse_matrix(PPI, dtype=np.float32),
#     result_folder = output_folder + 'nbs/', alpha=0.7, tol=10e-3, qn='mean')
# mut_propag.shape
#
#
#
