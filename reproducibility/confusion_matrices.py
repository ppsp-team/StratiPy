#!/usr/bin/env python
# coding: utf-8
import sys
import scipy.sparse as sp
import numpy as np
from scipy.io import loadmat, savemat
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import rcParams


def replace_list_element(l, before, after):
    """Helper function for get_cluster_idx
    """
    for i, e in enumerate(l):
        if e == before:
            l[i] = after
    return l


def get_cluster_idx(result_folder_repro, method, n_permutations, replace=False,
                    **kwargs):
    # NBS (Hofree) data
    if method == "nbs":
        data = loadmat(result_folder_repro + 'results_NBS_Hofree_{}.mat'
                       .format(n_permutations))
        cluster_idx = data['NBS_cc_label'].squeeze().tolist()

    # StratiPy data
    else:
        lambd = kwargs['lambd']
        filename = ('result_TCGA_UCEC_STRING/hierarchical_clustering/mean_qn/gnmf/hierarchical_clustering_Patients_weight=min_simp=True_alpha=0.7_tol=0.01_singletons=False_ngh=11_minMut=10_maxMut=200000_comp=3_permut={}_lambd={}_tolNMF=0.001_method=average.mat'
                    .format(n_permutations, lambd))
        data = loadmat(result_folder_repro + filename)
        cluster_idx = list(data['flat_cluster_number'][0])
        replace = True

    if replace:
        # Cooridnate Stratipy's cluster index with Hofree's cluster index
        # clust(Hofree) 2(1) <-> clust(Stratipy) 1(2)
        cluster_idx = replace_list_element(cluster_idx, 2, 0) # 2 -> 0
        cluster_idx = replace_list_element(cluster_idx, 1, 2) # 1 -> 2
        cluster_idx = replace_list_element(cluster_idx, 0, 1) # 0 -> 2

    return cluster_idx

#
# def loading_hierarchical_clustering_data_lamb1_1800(result_folder_repro):
#     print(" ==== loading Hierarchical clustering data ")
#     # # Hofree's result --- 100 permutations of bootstrap
#     # hof_100_data = loadmat(result_folder_repro + 'results_NBS_Hofree_100.mat')
#     #
#     # # StratiPy's result --- 100 permutations of bootstrap
#     # # result with lambda = 1 and 1800
#     # stp_100_data_lamb1 = loadmat(result_folder_repro +
#     #                              'result_TCGA_UCEC_STRING/hierarchical_clustering/mean_qn/gnmf/hierarchical_clustering_Patients_weight=min_simp=True_alpha=0.7_tol=0.01_singletons=False_ngh=11_minMut=10_maxMut=200000_comp=3_permut=100_lambd=1_tolNMF=0.001_method=average.mat')
#     # stp_100_data_lamb1800 = loadmat(result_folder_repro +
#     #                                 'result_TCGA_UCEC_STRING/hierarchical_clustering/mean_qn/gnmf/hierarchical_clustering_Patients_weight=min_simp=True_alpha=0.7_tol=0.01_singletons=False_ngh=11_minMut=10_maxMut=200000_comp=3_permut=100_lambd=1800_tolNMF=0.001_method=average.mat')
#     #
#     # # Get number of subgroups for each patient
#     # hof_100_3cluster = hof_100_data['NBS_cc_label'].squeeze().tolist()
#     # stp_100_lamb1_3cluster = list(stp_100_data_lamb1['flat_cluster_number'][0])
#     # stp_100_lamb1800_3cluster = list(stp_100_data_lamb1800['flat_cluster_number'][0])
#     #
#     # # Cooridnate Stratipy's cluster index with Hofree's cluster index
#     # # clust(Hofree) 2(1) <-> clust(Stratipy) 1(2)
#     # stp_100_lamb1_3cluster = replace_list_element(stp_100_lamb1_3cluster, 2, 0) # 2 -> 0
#     # stp_100_lamb1_3cluster = replace_list_element(stp_100_lamb1_3cluster, 1, 2) # 1 -> 2
#     # stp_100_lamb1_3cluster = replace_list_element(stp_100_lamb1_3cluster, 0, 1) # 0 -> 2
#     #
#     # stp_100_lamb1800_3cluster = replace_list_element(stp_100_lamb1800_3cluster, 2, 0) # 2 -> 0
#     # stp_100_lamb1800_3cluster = replace_list_element(stp_100_lamb1800_3cluster, 1, 2) # 1 -> 2
#     # stp_100_lamb1800_3cluster = replace_list_element(stp_100_lamb1800_3cluster, 0, 1) # 0 -> 2
#
#     print(" ==== creating Confusion matrices ")
#     # confusion matrice between Hofree & StratiPy (lambda = 1)
#     conf_lamb1 = confusion_matrix(hof_100_3cluster, stp_100_lamb1_3cluster)
#     conf_lamb1 = np.around((conf_lamb1.astype('float') / conf_lamb1.sum(axis=1)[:, np.newaxis]), decimals=2)
#
#     # confusion matrice between Hofree & StratiPy (lambda = 1800)
#     conf_lamb1800 = confusion_matrix(hof_100_3cluster, stp_100_lamb1800_3cluster)
#     conf_lamb1800 = np.around((conf_lamb1800.astype('float') / conf_lamb1800.sum(axis=1)[:, np.newaxis]), decimals=2)
#
#     return conf_lamb1, conf_lamb1800


def plot_confusion_matrix(result_folder_repro, M, plot_title, param_value):
    norm_conf = []
    for i in M:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    rcParams.update({'font.size': 12})
    fig = plt.figure()
    # plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.viridis, interpolation='nearest')

    width, height = M.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(M[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    levels = np.linspace(0, 1, 11, endpoint=True)
    cb = fig.colorbar(res, ticks=levels)
    # cb.set_clim(vmin=0, vmax=0.98)
    alphabet = '123'
    plt.xticks(range(width), alphabet[:width])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.yticks(range(height), alphabet[:height])

    plt.xlabel('Subgroups')
    plt.title(plot_title + '\n\n' + param_value, fontsize=14, y=1.3)

    plot_name = 'confusion_matrix_' + param_value.replace(" ", "")
    plt.savefig('{}{}.pdf'.format(result_folder_repro, plot_name),
                bbox_inches='tight')

#
# def reproducibility_confusion_matrices(result_folder_repro, param_val1, param_val2):
#     conf_lamb1, conf_lamb1800 = loading_hierarchical_clustering_data_lamb1_1800(
#         result_folder_repro)
#     print(" ==== plotting and saving Confusion matrices ")
#     plot_confusion_matrix(result_folder_repro, conf_lamb1,
#                            'Confusion matrix\nwith reported tuning parameter value',
#                            param_val1)
#     plot_confusion_matrix(result_folder_repro, conf_lamb1800,
#                            'Confusion matrix\nwith actually used tuning parameter value',
#                            param_val2)


def repro_confusion_matrix(result_folder_repro, data1, data2, plot_title,
                           param_value):
    conf_matrix = confusion_matrix(data1, data2)
    conf_matrix = np.around((conf_matrix.astype('float') /
                             conf_matrix.sum(axis=1)[:, np.newaxis]),
                            decimals=2)
    plot_confusion_matrix(result_folder_repro, conf_matrix, plot_title,
                          param_value)
