#!/usr/bin/env python
# coding: utf-8
import sys
import time
from simulation_functions import generate_network, addBetweenPathwaysConnection, generate_all_mutation_profile, plot_network_patient, save_dataset
from simulated_nbs_functions import all_functions
import datetime
from sklearn.model_selection import ParameterGrid
import networkx as nx
import pickle

output_folder = "output/"

# Pathways parameters
pathwaysNum = 6
genesNum = 12
connProbability = 0.4
connNeighboors = 4
connBetweenPathways = 2
marker_shapes = ['o', 'p', '<', 'd', 's', '*']

# Simulate patients with a specific mutation profile
patientsNum = 10
mutationProb = 0.2

# load PPI nodes' fixed positions
with open('input/ppi_node_position.txt', 'rb') as handle:
    position = pickle.load(handle)

PPI = generate_network(pathwaysNum, genesNum, connNeighboors,
                       connProbability, marker_shapes)

for BetweenPathwaysConnection in range(0, pathwaysNum*connBetweenPathways):
    PPI = addBetweenPathwaysConnection(PPI, pathwaysNum, genesNum)

patients, phenotypes = generate_all_mutation_profile(patientsNum, PPI,
                                                     genesNum, pathwaysNum,
                                                     mutationProb)

save_dataset(PPI, position, patients, phenotypes, pathwaysNum, genesNum,
                 connProbability, connNeighboors, connBetweenPathways,
                 patientsNum, mutationProb, output_folder, new_data=True)

plot_network_patient(PPI, position, patients, patientsNum, phenotypes,
                     marker_shapes, output_folder, plot_name="Raw")


# simulated NBS parameters
param_grid = {'output_folder': ['output/'],
              'ppi_data': ['simulation'],
              'patients': [patients],
              'patientsNum': [10],
              'mutationProb': [0.2],
              'phenotypes': [phenotypes],
              'marker_shapes': [marker_shapes],
              'pathwaysNum': [6],
              'genesNum': [12],
              'connNeighboors': [4],
              'connProbability': [0.4],
              'connBetweenPathways': [2],
              'position': [position],
              'influence_weight': ['min'],
              'simplification': [True],
              'compute': [True],
              'overwrite': [False],
              'alpha': [0.7],
              'tol': [10e-3],
              'ngh_max': [11],
              'keep_singletons': [False],
              'min_mutation': [0],
              'max_mutation': [100],
              'qn': ["mean"],
              'n_components': [3],
              'n_permutations': [1000],
              'run_bootstrap': [True],
              'run_consensus': [True],
              'lambd': [1],
              'tol_nmf': [1e-3],
              'linkage_method': ['average']
              }

for params in list(ParameterGrid(param_grid)):
    all_functions(**params)


plot_network_patient(PPI, position, mut_propag, patientsNum, phenotypes,
                     marker_shapes, output_folder, plot_name="Diffused")


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
