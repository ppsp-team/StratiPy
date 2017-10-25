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


def get_cluster_idx(result_folder_repro, method, n_permutations, replace_1by2=False,
                    replace_1by2_2by3_3by1=False, **kwargs):
    # NBS (Hofree) data
    if method == "nbs":
        data = loadmat('../data/results_NBS_Hofree_{}.mat'
                       .format(n_permutations))
        cluster_idx = data['NBS_cc_label'].squeeze().tolist()

    # StratiPy data
    elif method == "stratipy":
        lambd = kwargs['lambd']
        filename = ('result_TCGA_UCEC_STRING/hierarchical_clustering/mean_qn/gnmf/hierarchical_clustering_Patients_weight=min_simp=True_alpha=0.7_tol=0.01_singletons=False_ngh=11_minMut=10_maxMut=200000_comp=3_permut={}_lambd={}_tolNMF=0.001_method=average.mat'
                    .format(n_permutations, lambd))
        data = loadmat(result_folder_repro + filename)
        cluster_idx = list(data['flat_cluster_number'][0])

    # Cooridnate Stratipy's cluster index with Hofree's cluster index
    if replace_1by2:
        # clust(Hofree) 2(1) <-> clust(Stratipy) 1(2)
        cluster_idx = replace_list_element(cluster_idx, 2, 0) # 2 -> 0
        cluster_idx = replace_list_element(cluster_idx, 1, 2) # 1 -> 2
        cluster_idx = replace_list_element(cluster_idx, 0, 1) # 0 -> 1

    elif replace_1by2_2by3_3by1:
        cluster_idx = replace_list_element(cluster_idx, 3, 0)
        cluster_idx = replace_list_element(cluster_idx, 2, 3)
        cluster_idx = replace_list_element(cluster_idx, 1, 2)
        cluster_idx = replace_list_element(cluster_idx, 0, 1)

    return cluster_idx


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

    plt.xlabel('Subgroups\n')
    plt.title(plot_title + '\n\n' + param_value, fontsize=14, y=1.3)

    plot_name = 'confusion_matrix_' + param_value.replace(" ", "")
    plt.savefig('{}{}.pdf'.format(result_folder_repro, plot_name),
                bbox_inches='tight')


def repro_confusion_matrix(result_folder_repro, data1, data2, plot_title,
                           param_value):
    conf_matrix = confusion_matrix(data1, data2)
    conf_matrix = np.around((conf_matrix.astype('float') /
                             conf_matrix.sum(axis=1)[:, np.newaxis]),
                            decimals=2)
    plot_confusion_matrix(result_folder_repro, conf_matrix, plot_title,
                          param_value)
