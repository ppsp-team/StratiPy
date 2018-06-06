#!/usr/bin/env python
# coding: utf-8
import sys
import time
from simulation_functions import patients_data, nodes_position, generate_network, generate_network_shapes, addBetweenPathwaysConnection, save_dataset, plot_network_patient
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
pathwaysNum = 6
genesNum = 12
connProbability = 0.4
connNeighboors = 4
connBetweenPathways = 2
marker_shapes = ['o', 'p', '<', 'd', 's', '*']

# pathwaysNum = 10
# genesNum = 100
# connProbability = 0.4
# connNeighboors = 4
# connBetweenPathways = 2

# Simulate patients with a specific mutation profile
patientsNum = 10
mutationProb = 0.2


print("\n------------ generate simulated data ------------ {}"
      .format(datetime.datetime.now()
              .strftime("%Y-%m-%d %H:%M:%S")))
PPI = generate_network_shapes(
    pathwaysNum, genesNum, connNeighboors, connProbability, marker_shapes)

# PPI = generate_network(
#     pathwaysNum, genesNum, connNeighboors, connProbability)

for BetweenPathwaysConnection in range(0, pathwaysNum*connBetweenPathways):
    PPI = addBetweenPathwaysConnection(PPI, pathwaysNum, genesNum)

position = nodes_position(PPI, pathwaysNum, genesNum)

patients, phenotypes = patients_data(
    patientsNum, PPI, genesNum, pathwaysNum, mutationProb)

phenotype_idx = minimal_element_0_to_1(phenotypes)
print(phenotype_idx)


# save_dataset(PPI, position, patients, phenotypes, pathwaysNum, genesNum,
#              connProbability, connNeighboors, connBetweenPathways, patientsNum,
#              mutationProb, output_folder, new_data=True)


ppi_data = 'simulation'
influence_weight = 'min'
simplification = True
tol = 10e-3
keep_singletons = False
compute = True
overwrite = False
min_mutation = 0
max_mutation = 100
n_components = 6
n_permutations = 1000
tol_nmf = 1e-3
linkage_method = 'average'
phenotype_idx = phenotype_idx
run_bootstrap = True
run_consensus = True

alpha = 0
ngh_max = 5 #10
qn = None #"mean", "median"
lambd = 0 # 200

# all_functions(output_folder, ppi_data, patientsNum, mutationProb,
#                   marker_shapes, pathwaysNum, genesNum, connNeighboors,
#                   connProbability, connBetweenPathways, position,
#                  influence_weight, simplification,
#                  compute, overwrite, alpha, tol, ngh_max, keep_singletons, min_mutation,
#                  max_mutation, qn, n_components, n_permutations, run_bootstrap, run_consensus,
#                  lambd, tol_nmf, linkage_method)


# print("\n------------ loading & formatting data ------------ {}"
#       .format(datetime.datetime.now()
#               .strftime("%Y-%m-%d %H:%M:%S")))
ppi_filt = nx.to_scipy_sparse_matrix(PPI, dtype=np.float32)
ppi_final = ppi_filt
mut_final = sp.csr_matrix(patients, dtype=np.float32)

# simulated NBS parameters
param_grid = {'output_folder': ['output/'],
              'ppi_data': ['simulation'],
              'ppi_filt': [ppi_filt],
              'mut_final': [mut_final],
              'influence_weight': ['min'],
              'simplification': [True],
              'compute': [True],
              'overwrite': [False],
              'alpha': [0, 0.5],
              'tol': [10e-3],
              'ngh_max': [5], #10
              'keep_singletons': [False],
              'min_mutation': [0],
              'max_mutation': [100],
              'qn': [None],
              'n_components': [6],
              'n_permutations': [1000],
              'run_bootstrap': [True],
              'run_consensus': [True],
              'lambd': [0], # 200
              'tol_nmf': [1e-3],
              'linkage_method': ['average']
              }

start_all = time.time()
for params in list(ParameterGrid(param_grid)):
    for i in params.keys():
        exec("%s = %s" % (i, 'params[i]'))

    if alpha == 0 and qn is not None:
        print('############ PASS ############')
        pass

    else:
        print(qn)
        print(alpha)
        mut_type, mut_propag = all_functions(**params)

        plot_network_patient(
            mut_type, alpha, tol, PPI,
            position, np.array(mut_propag), patientsNum, phenotypes, marker_shapes,
            output_folder)


    # simulation_functions.plot_network_patient(
    #     mut_type, alpha, tol, nx.from_scipy_sparse_matrix(
    #         final_influence, create_using=nx.watts_strogatz_graph(
    #             genesNum, connNeighboors, connProbability)),
    #     position, patients, patientsNum, phenotypes, marker_shapes,
    #     output_folder)


end_all = time.time()
print('---------- ALL = {} ---------- {}'
      .format(datetime.timedelta(seconds = end_all - start_all),
              datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


# print("\n------------ confusion matrices ------------ {}"
#       .format(datetime.datetime.now()
#               .strftime("%Y-%m-%d %H:%M:%S")))
#
# param_grid2 = {'output_folder': ['output/'],
#               'influence_weight': ['min'],
#               'simplification': [True],
#               'alpha': [0],
#               'tol': [10e-3],
#               'ngh_max': [5], #[5, 10]
#               'keep_singletons': [False],
#               'min_mutation': [0],
#               'max_mutation': [100],
#               'qn': [None], # "mean", "median"
#               'n_components': [6],
#               'n_permutations': [1000],
#               'lambd': [0],
#               'tol_nmf': [1e-3],
#               'linkage_method': ['average'],
#               'phenotype_idx': [phenotype_idx],
#               'pathwaysNum': [pathwaysNum]
#               }
# for params in list(ParameterGrid(param_grid2)):
#     for i in params.keys():
#         exec("%s = %s" % (i, 'params[i]'))
#     simulated_confusion_matrix(**params)



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
