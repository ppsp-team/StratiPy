import sys
import os
sys.path.append(os.path.abspath('../../stratipy'))
# from stratipy import consensus_clustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist
import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp
from scipy.io import loadmat, savemat
import warnings
import time
import datetime
import os
import glob
import pandas as pd
from scipy.stats import fisher_exact, ks_2samp, chisquare, kruskal
from statistics import median
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.switch_backend('agg')
# to avoid recursion error during hierarchical clustering
sys.setrecursionlimit(100000)

def hierarchical_file(result_folder, mut_type, influence_weight,
                      simplification, alpha, tol, keep_singletons, ngh_max,
                      min_mutation, max_mutation, n_components, n_permutations,
                      lambd, tol_nmf, linkage_method):
    hierarchical_directory = result_folder+'hierarchical_clustering/'
    # os.makedirs(hierarchical_directory, exist_ok=True)
    hierarchical_mut_type_directory = hierarchical_directory + mut_type + '/'
    # os.makedirs(hierarchical_mut_type_directory, exist_ok=True)

    if lambd > 0:
        hierarchical_factorization_directory = (
            hierarchical_mut_type_directory + 'gnmf/')
    else:
        hierarchical_factorization_directory = (
            hierarchical_mut_type_directory + 'nmf/')
    os.makedirs(hierarchical_factorization_directory, exist_ok=True)

    hierarchical_clustering_file = (
        hierarchical_factorization_directory +
        'hierarchical_clustering_weight={}_simp={}_alpha={}_tol={}_singletons={}_ngh={}_minMut={}_maxMut={}_comp={}_permut={}_lambd={}_tolNMF={}_method={}.mat'
        .format(influence_weight, simplification, alpha, tol, keep_singletons,
                ngh_max, min_mutation, max_mutation, n_components,
                n_permutations, lambd, tol_nmf, linkage_method))
    return hierarchical_clustering_file


def linkage_dendrogram(hierarchical_clustering_file, distance_genes,
                       distance_individuals, ppi_data, mut_type, alpha, ngh_max,
                       n_components, n_permutations, lambd, linkage_method,
                       patient_data, data_folder, ssc_subgroups,
                       ssc_mutation_data, gene_data):

    existance_same_param = os.path.exists(hierarchical_clustering_file)

    if existance_same_param:
        h = loadmat(hierarchical_clustering_file)
        # cluster index for each individual
        clust_nb_individuals = np.squeeze(h['flat_cluster_number_individuals'])
        # individuals' index
        idx_individuals = np.squeeze(h['dendrogram_index_individuals'])
        print(' **** Same parameters file of hierarchical clustering already exists')
    else:
        # hierarchical clustering on distance matrix (here: distance_individuals)
        start = time.time()
        Z_individuals = linkage(distance_individuals, method=linkage_method)
        end = time.time()
        print(" ==== Linkage based on Individual distance = {} ---------- {}"
              .format(datetime.timedelta(seconds=end-start),
                      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
              flush=True)

        start = time.time()
        Z_genes = linkage(distance_genes, method=linkage_method)
        end = time.time()
        print(" ==== Linkage based on Gene distance = {} ---------- {}"
              .format(datetime.timedelta(seconds=end-start),
                      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
              flush=True)

        # dendrogram too slow (too many recursion for genes)
        P_individuals = dendrogram(
            Z_individuals, count_sort='ascending', no_labels=True)
        P_genes = dendrogram(Z_genes, count_sort='ascending', no_labels=True)
        idx_individuals = np.array(P_individuals['leaves'])
        idx_genes = np.array(P_genes['leaves'])

        # forms flat clusters from Z
        # given k -> maxclust
        clust_nb_individuals = fcluster(
            Z_individuals, n_components, criterion='maxclust')
        clust_nb_genes = fcluster(Z_genes, n_components, criterion='maxclust')

        # start = time.time()
        savemat(hierarchical_clustering_file,
                {'Z_linkage_matrix_individuals': Z_individuals,
                 'dendrogram_data_dictionary_individuals': P_individuals,
                 'dendrogram_index_individuals': idx_individuals,
                 'flat_cluster_number_individuals': clust_nb_individuals,
                 'Z_linkage_matrix_genes': Z_genes,
                 'dendrogram_data_dictionary_genes': P_genes,
                 'dendrogram_index_genes': idx_genes,
                 'flat_cluster_number_genes': clust_nb_genes},
                do_compression=True)

        D_individuals = distance_individuals[idx_individuals, :][:, idx_individuals]

        fig = plt.figure(figsize=(3, 3))
        im = plt.imshow(D_individuals, interpolation='nearest', cmap=cm.viridis)
        plt.axis('off')
        if patient_data == 'SSC':
            fig_directory = (
                data_folder + 'figures/similarity/' + ssc_mutation_data + '_' +
                ssc_subgroups + '_' + gene_data + '_' + ppi_data + '/')
        else:
            fig_directory = (data_folder + 'figures/similarity/' +
                             patient_data + '_' + ppi_data + '/')
        os.makedirs(fig_directory, exist_ok=True)
        fig_name = ('{}_{}_k={}_ngh={}_permut={}_lambd={}'.format(
            mut_type, alpha, n_components, ngh_max, n_permutations, lambd))
        plt.savefig('{}{}.png'.format(fig_directory, fig_name),
                    bbox_inches='tight')


def hierarchical(result_folder, distance_genes, distance_individuals, ppi_data,
                 mut_type, influence_weight, simplification, alpha, tol,
                 keep_singletons, ngh_max, min_mutation, max_mutation,
                 n_components, n_permutations, lambd, tol_nmf, linkage_method,
                 patient_data, data_folder, ssc_subgroups, ssc_mutation_data,
                 gene_data):
    hierarchical_clustering_file = hierarchical_file(
        result_folder, mut_type, influence_weight, simplification, alpha, tol,
        keep_singletons, ngh_max, min_mutation, max_mutation, n_components,
        n_permutations, lambd, tol_nmf, linkage_method)

    linkage_dendrogram(hierarchical_clustering_file, distance_genes,
                       distance_individuals, ppi_data, mut_type, alpha, ngh_max,
                       n_components, n_permutations, lambd, linkage_method,
                       patient_data, data_folder, ssc_subgroups,
                       ssc_mutation_data, gene_data)
