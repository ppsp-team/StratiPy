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
sys.setrecursionlimit(10000)

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
                       distance_patients, ppi_data, mut_type, alpha, ngh_max,
                       n_components, n_permutations, lambd, linkage_method,
                       patient_data, data_folder, ssc_subgroups,
                       ssc_mutation_data, gene_data):

    existance_same_param = os.path.exists(hierarchical_clustering_file)

    if existance_same_param:
        h = loadmat(hierarchical_clustering_file)
        # cluster index for each individual
        clust_nb_patients = np.squeeze(h['flat_cluster_number_individuals'])
        # individuals' index
        idx_patients = np.squeeze(h['dendrogram_index_individuals'])
        print(' **** Same parameters file of hierarchical clustering already exists')
    else:
        # hierarchical clustering on distance matrix (here: distance_patients)
        start = time.time()
        Z_patients = linkage(distance_patients, method=linkage_method)
        end = time.time()
        print("---------- Linkage based on Individual distance = {} ---------- {}"
              .format(datetime.timedelta(seconds=end-start),
                      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
              flush=True)

        start = time.time()
        Z_genes = linkage(distance_genes, method=linkage_method)
        end = time.time()
        print("---------- Linkage based on Gene distance = {} ---------- {}"
              .format(datetime.timedelta(seconds=end-start),
                      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
              flush=True)

        P_patients = dendrogram(
            Z_patients, count_sort='ascending', no_labels=True)
        P_genes = dendrogram(Z_genes, count_sort='ascending', no_labels=True)


        idx_patients = np.array(P_patients['leaves'])
        idx_genes = np.array(P_genes['leaves'])

        # forms flat clusters from Z
        # given k -> maxclust
        clust_nb_patients = fcluster(
            Z_patients, n_components, criterion='maxclust')
        clust_nb_genes = fcluster(Z_genes, n_components, criterion='maxclust')

        # start = time.time()
        savemat(hierarchical_clustering_file,
                {'Z_linkage_matrix_individuals': Z_patients,
                 'dendrogram_data_dictionary_individuals': P_patients,
                 'dendrogram_index_individuals': idx_patients,
                 'flat_cluster_number_individuals': clust_nb_patients,
                 'Z_linkage_matrix_genes': Z_genes,
                 'dendrogram_data_dictionary_genes': P_genes,
                 'dendrogram_index_genes': idx_genes,
                 'flat_cluster_number_genes': clust_nb_genes},
                do_compression=True)

        D_patients = distance_patients[idx_patients, :][:, idx_patients]

        fig = plt.figure(figsize=(3, 3))
        im = plt.imshow(D_patients, interpolation='nearest', cmap=cm.viridis)
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


# for no SSC data
def individual_linkage_dendrogram(hierarchical_clustering_file,
                       distance_patients, ppi_data, mut_type, alpha, ngh_max,
                       n_components, n_permutations, lambd, linkage_method,
                       patient_data, data_folder, result_folder, repro):

    existance_same_param = os.path.exists(hierarchical_clustering_file)

    if existance_same_param:
        h = loadmat(hierarchical_clustering_file)
        # cluster index for each individual
        clust_nb_patients = np.squeeze(h['flat_cluster_number_individuals'])
        # individuals' index
        idx_patients = np.squeeze(h['dendrogram_index_individuals'])
        print(' **** Same parameters file of hierarchical clustering already exists')
    else:
        # hierarchical clustering on distance matrix (here: distance_patients)
        start = time.time()
        Z_patients = linkage(distance_patients, method=linkage_method)
        end = time.time()
        print("---------- Linkage based on Individual distance = {} ---------- {}"
              .format(datetime.timedelta(seconds=end-start),
                      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
              flush=True)

        P_patients = dendrogram(
            Z_patients, count_sort='ascending', no_labels=True)
        
        idx_patients = np.array(P_patients['leaves'])

        # forms flat clusters from Z
        # given k -> maxclust
        clust_nb_patients = fcluster(
            Z_patients, n_components, criterion='maxclust')

        # start = time.time()
        savemat(hierarchical_clustering_file,
                {'Z_linkage_matrix_individuals': Z_patients,
                 'dendrogram_data_dictionary_individuals': P_patients,
                 'dendrogram_index_individuals': idx_patients,
                 'flat_cluster_number_individuals': clust_nb_patients},
                do_compression=True)

        D_patients = distance_patients[idx_patients, :][:, idx_patients]

        fig = plt.figure(figsize=(3, 3))
        im = plt.imshow(D_patients, interpolation='nearest', cmap=cm.viridis)
        plt.axis('off')
        if repro:
            directory = result_folder
        else:
            directory = data_folder
        fig_directory = directory + 'figures/similarity/' + patient_data + '_' + ppi_data + '/'
        os.makedirs(fig_directory, exist_ok=True)
        fig_name = ('{}_{}_k={}_ngh={}_permut={}_lambd={}'.format(
            mut_type, alpha, n_components, ngh_max, n_permutations, lambd))
        print('saving plot')
        plt.savefig('{}{}.png'.format(fig_directory, fig_name),
                    bbox_inches='tight')


def hierarchical(result_folder, distance_genes, distance_patients, ppi_data,
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
                       distance_patients, ppi_data, mut_type, alpha, ngh_max,
                       n_components, n_permutations, lambd, linkage_method,
                       patient_data, data_folder, ssc_subgroups,
                       ssc_mutation_data, gene_data)

# # TODO formatting for not SSC data
# def distance_patients_from_consensus_file(
#     result_folder, distance_patients, ppi_data, mut_type,
#     influence_weight, simplification,
#     alpha, tol,  keep_singletons, ngh_max, min_mutation, max_mutation,
#     n_components, n_permutations, lambd, tol_nmf, linkage_method,
#     patient_data, data_folder, ssc_subgroups, ssc_mutation_data, gene_data):
#
#     consensus_directory = result_folder+'consensus_clustering/'
#     consensus_mut_type_directory = consensus_directory + mut_type + '/'
#
#     hierarchical_directory = result_folder+'hierarchical_clustering/'
#     os.makedirs(hierarchical_directory, exist_ok=True)
#     hierarchical_mut_type_directory = hierarchical_directory + mut_type + '/'
#     os.makedirs(hierarchical_mut_type_directory, exist_ok=True)
#
#     if lambd > 0:
#         consensus_factorization_directory = (
#             consensus_mut_type_directory + 'gnmf/')
#         hierarchical_factorization_directory = (
#             hierarchical_mut_type_directory + 'gnmf/')
#     else:
#         consensus_factorization_directory = (
#             consensus_mut_type_directory + 'nmf/')
#         hierarchical_factorization_directory = (
#             hierarchical_mut_type_directory + 'nmf/')
#     os.makedirs(hierarchical_factorization_directory, exist_ok=True)
#
#     hierarchical_clustering_file = (
#         hierarchical_factorization_directory +
#         'hierarchical_clustering_Patients_weight={}_simp={}_alpha={}_tol={}_singletons={}_ngh={}_minMut={}_maxMut={}_comp={}_permut={}_lambd={}_tolNMF={}_method={}.mat'
#         .format(influence_weight, simplification, alpha, tol, keep_singletons,
#                 ngh_max, min_mutation, max_mutation, n_components,
#                 n_permutations, lambd, tol_nmf, linkage_method))
#     existance_same_param = os.path.exists(hierarchical_clustering_file)
#
#     if existance_same_param:
#         print(' **** Same parameters file of hierarchical clustering already exists')
#     else:
#         # print(type(distance_patients), distance_patients.shape)
#         # hierarchical clustering on distance matrix (here: distance_patients)
#         Z = linkage(distance_patients, method=linkage_method)
#
#         # Plot setting
#         matplotlib.rcParams.update({'font.size': 14})
#         fig = plt.figure(figsize=(20, 20))
#         fig.suptitle(
#             'Hierarchical clustering\n\nPatients', fontsize=30, x=0.13, y=0.95)
#
#         # Compute and plot dendrogram
#         ax_dendro = fig.add_axes([0, 0.71, 0.6, 0.15])
#         P = dendrogram(Z, count_sort='ascending', no_labels=True)
#         ax_dendro.set_xticks([])
#         ax_dendro.set_yticks([])
#
#         # Plot distance matrix.
#         ax_matrix = fig.add_axes([0, 0.1, 0.6, 0.6])
#         idx = np.array(P['leaves'])
#         D = distance_patients[idx, :][:, idx]
#         im = ax_matrix.imshow(D, interpolation='nearest', cmap=cm.viridis)
#         ax_matrix.set_xticks([])
#         ax_matrix.set_yticks([])
#
#         # Plot colorbar.
#         ax_color = fig.add_axes([0.62, 0.1, 0.02, 0.6])
#         ax_color.set_xticks([])
#         plt.colorbar(im, cax=ax_color)
#
#         # forms flat clusters from Z
#         # given k -> maxclust
#         clust_nb = fcluster(Z, n_components, criterion='maxclust')
#         # cophenetic correlation distance
#         coph_dist, coph_matrix = cophenet(Z, pdist(distance_patients))
#         print(' ==== cophenetic correlation distance = ', coph_dist)
#
#         ax_dendro.set_title(
#             'network = {}\nalpha = {}\nmutation type = {}\ninfluence weight = {}\nsimplification = {}\ncomponent number = {}\nlambda = {}\nmethod = {}\ncophenetic corr = {}\n'
#             .format(ppi_data, alpha, mut_type,
#                     influence_weight, simplification,
#                     n_components, lambd, linkage_method,
#                     format(coph_dist, '.2f')), loc='right')
#
#         plot_name = "similarity_matrix_Patients" + (
#             '_alpha={}_tol={}_singletons={}_ngh={}_minMut={}_maxMut={}_comp={}_permut={}_lambd={}_tolNMF={}_method={}'
#             .format(alpha, tol, keep_singletons, ngh_max, min_mutation,
#                     max_mutation, n_components, n_permutations, lambd, tol_nmf,
#                     linkage_method))
#         plt.savefig('{}{}.pdf'.format(hierarchical_factorization_directory,
#                                       plot_name), bbox_inches='tight')
#         plt.savefig('{}{}.svg'.format(hierarchical_factorization_directory,
#                                       plot_name), bbox_inches='tight')
#
#         # start = time.time()
#         savemat(hierarchical_clustering_file,
#                 {'Z_linkage_matrix': Z,
#                  'dendrogram_data_dictionary': P,
#                  'dendrogram_index': idx,
#                  'flat_cluster_number': clust_nb,
#                  'cophenetic_correlation_distance': coph_dist,
#                  'cophenetic_correlation_matrix': coph_matrix},
#                 do_compression=True)
#         # # end = time.time()
#         # print("---------- Save time = {} ---------- {}"
#         #       .format(datetime.timedelta(seconds=end-start),
#         #               datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
#
#         fig = plt.figure(figsize=(3, 3))
#         im = plt.imshow(D, interpolation='nearest', cmap=cm.viridis)
#         plt.axis('off')
#         if patient_data == 'SSC':
#             fig_directory = (data_folder + 'figures/similarity/' +
#                              ssc_mutation_data + '_' + ssc_subgroups + '_' + gene_data +
#                              '_' + ppi_data + '/')
#         else:
#             fig_directory = (data_folder + 'figures/similarity/' +
#                              patient_data + '_' + ppi_data + '/')
#         os.makedirs(fig_directory, exist_ok=True)
#         fig_name = ('{}_{}_k={}_ngh={}_permut={}_lambd={}'.format(
#             mut_type, alpha, n_components, ngh_max, n_permutations, lambd))
#         plt.savefig('{}{}.png'.format(fig_directory, fig_name),
#                     bbox_inches='tight')
