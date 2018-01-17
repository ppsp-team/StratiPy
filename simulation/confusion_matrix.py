#!/usr/bin/env python
# coding: utf-8
import sys
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat, savemat
from scipy import cluster
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.switch_backend('agg')


def mutation_profile_type(qn, alpha):
    if qn == None:
        if alpha == 0:
            mut_type = 'raw'
            return mut_type
        else:
            mut_type = 'diff'
    else:
        mut_type = qn + '_qn'

    return mut_type


def minimal_element_0_to_1(alist):
    if min(alist) == 0:
        alist = [x+1 for x in alist]
        return alist
    else:
        return alist


def replace_list_element(l, before, after):
    """Helper function for get_cluster_idx
    """
    for i, e in enumerate(l):
        if e == before:
            l[i] = after
    return l


def get_cluster_idx(output_folder, pathwaysNum, influence_weight,
                    simplification, mut_type, alpha, tol, keep_singletons, ngh_max,
                    min_mutation, max_mutation, n_components, n_permutations,
                    lambd, tol_nmf, linkage_method):
    # load NBS results from simulated data
    hierarchical_directory = (
        output_folder+'nbs/hierarchical_clustering/' + mut_type + '/')
    os.makedirs(hierarchical_directory, exist_ok=True)

    if lambd > 0:
        hierarchical_factorization_directory = (
            hierarchical_directory + 'gnmf/')
    else:
        hierarchical_factorization_directory = (
            hierarchical_directory + 'nmf/')
    os.makedirs(hierarchical_factorization_directory, exist_ok=True)

    hierarchical_clustering_file = (
        hierarchical_factorization_directory +
        'hierarchical_clustering_Patients_weight={}_simp={}_alpha={}_tol={}_singletons={}_ngh={}_minMut={}_maxMut={}_comp={}_permut={}_lambd={}_tolNMF={}_method={}.mat'
        .format(influence_weight, simplification, alpha, tol, keep_singletons,
                ngh_max, min_mutation, max_mutation, n_components,
                n_permutations, lambd, tol_nmf, linkage_method))
    existance_same_param = os.path.exists(hierarchical_clustering_file)

    data = loadmat(hierarchical_clustering_file)
    if n_components != pathwaysNum:
        # cut_tree threashold depends on component number
        Z = list(data['Z_linkage_matrix'])
        cluster_idx = cluster.hierarchy.cut_tree(Z, n_clusters=pathwaysNum)

    else:
        cluster_idx = list(data['flat_cluster_number'][0])

    cluster_idx = minimal_element_0_to_1(cluster_idx)

    coph_dist = list(data['cophenetic_correlation_distance'][0])[0]

    return cluster_idx, coph_dist


def plot_confusion_matrix(output_folder, M, pathwaysNum, mut_type, influence_weight, simplification,
                          alpha, tol, keep_singletons, ngh_max, min_mutation,
                          max_mutation, n_components, n_permutations, lambd,
                          tol_nmf, linkage_method, coph_dist):
    confusion_mut_type_directory = (
        output_folder+'nbs/confusion_matrix/' + mut_type + '/')
    os.makedirs(confusion_mut_type_directory, exist_ok=True)

    if lambd > 0:
        confusion_factorization_directory = (
            confusion_mut_type_directory + 'gnmf/')
    else:
        confusion_factorization_directory = (
            confusion_mut_type_directory + 'nmf/')
    os.makedirs(confusion_factorization_directory, exist_ok=True)

    confusion_file = (
        confusion_factorization_directory +
        'hierarchical_clustering_Patients_weight={}_simp={}_alpha={}_tol={}_singletons={}_ngh={}_minMut={}_maxMut={}_comp={}_permut={}_lambd={}_tolNMF={}_method={}.mat'
        .format(influence_weight, simplification, alpha, tol, keep_singletons,
                ngh_max, min_mutation, max_mutation, n_components,
                n_permutations, lambd, tol_nmf, linkage_method))
    existance_same_param = os.path.exists(confusion_file)

    if existance_same_param:
        print(' **** Same parameters file of confusion matrix already exists')
    else:
        norm_conf = []
        for i in M:
            a = 0
            tmp_arr = []
            a = sum(i, 0)
            for j in i:
                tmp_arr.append(float(j)/float(a))
            norm_conf.append(tmp_arr)

        fig = plt.figure()
        rcParams.update({'font.size': 12})

        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        res = ax.imshow(
            np.array(norm_conf), cmap=plt.cm.viridis, interpolation='nearest')

        width, height = M.shape

        for x in range(width):
            for y in range(height):
                ax.annotate(str(M[x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center')

        levels = np.linspace(0, 1, 11, endpoint=True)
        # cb = fig.colorbar(res, ticks=levels)
        cb = fig.colorbar(res, ticks=[0, 0.5, 1])
        cb.ax.set_yticklabels(['0%', '50%', '100%'], fontsize=11)
        alphabet = ''.join(map(str, [x+1 for x in list(range(M.shape[0]))]))
        plt.xticks(range(width), alphabet[:width])
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.yticks(range(height), alphabet[:height])

        plt.xlabel('Subgroups')
        plt.title('Confusion matrix with simulated data\n\n(Known input data vs NBS results)',
                  fontsize=14, x=1.1, y=1.2)

        ax_right = fig.add_axes([0.92, 0.1, 0.4, 0.2])
        ax_right.set_title(
            'components = {}\npathways = {}\n\nalpha = {}\nQN = {}\n\nlambda = {}\n\ncophenetic corr = {}'
            .format(n_components, pathwaysNum, alpha, mut_type, lambd,
                    format(coph_dist, '.2f')), loc='left')
        ax_right.axis('off')

        plot_name = "confusion_matrix" + (
            '_pathNum={}_alpha={}_ngh={}_minMut={}_maxMut={}_comp={}_permut={}_lambd={}'
            .format(pathwaysNum, alpha, ngh_max, min_mutation, max_mutation,
                    n_components, n_permutations, lambd))
        plt.savefig('{}{}.pdf'.format(confusion_factorization_directory,
                                      plot_name), bbox_inches='tight')
        # plt.savefig('{}{}.svg'.format(confusion_factorization_directory,
        #                               plot_name), bbox_inches='tight')


def simulated_confusion_matrix(output_folder, phenotype_idx, pathwaysNum,
                               influence_weight, simplification, qn,
                               alpha, tol, keep_singletons, ngh_max,
                               min_mutation, max_mutation, n_components,
                               n_permutations, lambd, tol_nmf, linkage_method):
    if alpha == 0 and qn is not None:
        pass

    else:
        # print("components = ", n_components)
        mut_type = mutation_profile_type(qn, alpha)

        # load cluster indexes and cophenetic correlation distance of NBS results
        cluster_idx, coph_dist = get_cluster_idx(
            output_folder, pathwaysNum, influence_weight, simplification, mut_type,
            alpha, tol, keep_singletons, ngh_max, min_mutation, max_mutation,
            n_components, n_permutations, lambd, tol_nmf, linkage_method)

        conf_matrix = confusion_matrix(phenotype_idx, cluster_idx)
        # conf_matrix = np.around((conf_matrix.astype('float') /
        #                          conf_matrix.sum(axis=1)[:, np.newaxis]),
        #                         decimals=2)

        plot_confusion_matrix(output_folder, conf_matrix, pathwaysNum, mut_type,
                              influence_weight, simplification, alpha, tol,
                              keep_singletons, ngh_max, min_mutation, max_mutation,
                              n_components, n_permutations, lambd, tol_nmf,
                              linkage_method, coph_dist)


#
# output_folder='simulation/output/'
# pathwaysNum=6
# influence_weight='min'
# simplification=True
# qn='median'
# alpha=0.7
# tol=10e-3
# keep_singletons=False
# ngh_max=3
# min_mutation=0
# max_mutation=100
# n_components=6
# n_permutations=1000
# lambd=200
# tol_nmf=1e-3
# linkage_method='average'
#
# mut_type = mutation_profile_type(qn, alpha)
# import pickle
# with open('simulation/input/{}_patients.txt'.format(100), 'rb') as handle:
#     load_data = pickle.load(handle)
#     patients = load_data['patients']
#     phenotypes = load_data['phenotypes']
#
# cl_idx, coph = get_cluster_idx(output_folder, pathwaysNum, influence_weight,
#                     simplification, mut_type, alpha, tol, keep_singletons, ngh_max,
#                     min_mutation, max_mutation, n_components, n_permutations,
#                     lambd, tol_nmf, linkage_method)
#
# phe = minimal_element_0_to_1(phenotypes)
# # type(cl_idx)
#
# cm = confusion_matrix(cl_idx, phe)
#
# plot_confusion_matrix(output_folder, cm, pathwaysNum, mut_type, influence_weight, simplification,
#                           alpha, tol, keep_singletons, ngh_max, min_mutation,
#                           max_mutation, n_components, n_permutations, lambd,
#                           tol_nmf, linkage_method, coph)
