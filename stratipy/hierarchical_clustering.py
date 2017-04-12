#!/usr/bin/env python
# coding: utf-8
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist
import sys
import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp
from scipy.io import loadmat, savemat
import warnings
import time
import datetime
import os
import glob
# import pylab
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.switch_backend('agg')


def distance_patients_from_consensus_file(
    result_folder, distance_patients, ppi_data, mut_type,
    influence_weight, simplification,
    alpha, tol,  keep_singletons, ngh_max, min_mutation, max_mutation,
    n_components, n_permutations, lambd, tol_nmf, linkage_method):

    consensus_directory = result_folder+'consensus_clustering/'
    consensus_mut_type_directory = consensus_directory + mut_type + '/'

    hierarchical_directory = result_folder+'hierarchical_clustering/'
    os.makedirs(hierarchical_directory, exist_ok=True)
    hierarchical_mut_type_directory = hierarchical_directory + mut_type + '/'
    os.makedirs(hierarchical_mut_type_directory, exist_ok=True)

    if lambd > 0:
        consensus_factorization_directory = (
            consensus_mut_type_directory + 'gnmf/')
        hierarchical_factorization_directory = (
            hierarchical_mut_type_directory + 'gnmf/')
    else:
        consensus_factorization_directory = (
            consensus_mut_type_directory + 'nmf/')
        hierarchical_factorization_directory = (
            hierarchical_mut_type_directory + 'nmf/')
    os.makedirs(hierarchical_factorization_directory, exist_ok=True)

    hierarchical_clustering_file = (
        hierarchical_factorization_directory +
        'hierarchical_clustering_Patients_weight={}_simp={}_alpha={}_tol={}_singletons={}_ngh={}_minMut={}_maxMut={}_comp={}_permut={}_lambd={}_tolNMF={}_method={}.mat'
        .format(influence_weight, simplification, alpha, tol, keep_singletons,
                ngh_max, min_mutation, max_mutation, n_components,
                n_permutations, lambd, tol_nmf, linkage_method))
    existance_same_param = os.path.exists(hierarchical_clustering_file)

    if existance_same_param:
        print("already exists")
    else:
        print(type(distance_patients), distance_patients.shape)
        # hierarchical clustering on distance matrix (here: distance_patients)
        Z = linkage(distance_patients, method=linkage_method)

        # Plot setting
        matplotlib.rcParams.update({'font.size': 14})
        fig = plt.figure(figsize=(20, 20))
        fig.suptitle('Hierarchical clustering\n\nPatients', fontsize=30, x=0.13, y=0.95)

        # Compute and plot dendrogram
        ax_dendro = fig.add_axes([0, 0.71, 0.6, 0.15])
        P = dendrogram(Z, count_sort='ascending', no_labels=True)
        ax_dendro.set_xticks([])
        ax_dendro.set_yticks([])

        # Plot distance matrix.
        ax_matrix = fig.add_axes([0, 0.1, 0.6, 0.6])
        idx = np.array(P['leaves'])
        D = distance_patients[idx, :][:, idx]
        im = ax_matrix.imshow(D, interpolation='nearest', cmap=cm.viridis)
        ax_matrix.set_xticks([])
        ax_matrix.set_yticks([])

        # Plot colorbar.
        ax_color = fig.add_axes([0.62, 0.1, 0.02, 0.6])
        ax_color.set_xticks([])
        plt.colorbar(im, cax=axcolor)

        # forms flat clusters from Z
        clust_nb = fcluster(Z, n_components, criterion='maxclust') # given k -> maxclust
        # cophenetic correlation distance
        coph_dist, coph_matrix = cophenet(Z, pdist(distance_patients))
        print('cophenetic correlation distance = ', coph_dist)

        ax_dendro.set_title(
            'network = {}\nalpha = {}\nmutation type = {}\ninfluence weight = {}\nsimplification = {}\ncomponent number = {}\nlambda = {}\nmethod = {}\ncophenetic corr = {}\n'
            .format(ppi_data, alpha, mut_type,
                    influence_weight, simplification,
                    n_components, lambd, linkage_method, format(coph_dist,
                                                                '.2f')))

        plot_name = "similarity_matrix_Patients" + (
            '_alpha={}_tol={}_singletons={}_ngh={}_minMut={}_maxMut={}_comp={}_permut={}_lambd={}_tolNMF={}_method={}'
            .format(alpha, tol, keep_singletons, ngh_max, min_mutation,
                    max_mutation, n_components, n_permutations, lambd, tol_nmf,
                    linkage_method))
        plt.savefig('{}{}.pdf'.format(result_folder, plot_name),
                    bbox_inches='tight')
        plt.savefig('{}{}.svg'.format(result_folder, plot_name),
                    bbox_inches='tight')

        start = time.time()
        savemat(hierarchical_clustering_file,
                {'Z_linkage_matrix': Z,
                 'dendrogram_data_dictionary': P,
                 'dendrogram_index': idx,
                 'flat_cluster_number': clust_nb,
                 'cophenetic_correlation_distance': coph_dist,
                 'cophenetic_correlation_matrix': coph_matrix},
                do_compression=True)
        end = time.time()
        print("---------- Save time = {} ---------- {}"
              .format(datetime.timedelta(seconds=end-start),
                      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
