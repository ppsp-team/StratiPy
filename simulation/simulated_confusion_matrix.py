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


def get_cluster_idx(output_folder, method, n_permutations, replace_1by2=False,
                    replace_1by2_2by3_3by1=False, **kwargs):
    # load NBS results from simulated data

    method == "simulation":
    lambd = kwargs['lambd']

    hierarchical_mut_type_directory = (
        output_folder+'hierarchical_clustering/' + mut_type + '/')

    if lambd > 0:
        hierarchical_factorization_directory = (
            hierarchical_mut_type_directory + 'gnmf/')
    else:
        hierarchical_factorization_directory = (
            hierarchical_mut_type_directory + 'nmf/')




        

    filename = ('nbs/hierarchical_clustering/mean_qn/gnmf/hierarchical_clustering_Patients_weight=min_simp=True_alpha=0.7_tol=0.01_singletons=False_ngh=11_minMut=10_maxMut=200000_comp=3_permut={}_lambd={}_tolNMF=0.001_method=average.mat'
                .format(n_permutations, lambd))
    data = loadmat(output_folder + filename)
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


def plot_confusion_matrix(output_folder, M, plot_title, lambd,
                          tilt=False):
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
    alphabet = '123'
    plt.xticks(range(width), alphabet[:width])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.yticks(range(height), alphabet[:height])

    if tilt:
        lambd = "~ " + str(lambd)

    plt.xlabel('Subgroups')
    plt.title(plot_title + " (" + (r'$\lambda = $%s)')%lambd,
              fontsize=14, x=0.57, y=1.2)

    if tilt:
        lambd = lambd.replace(" ", "")

    plot_name = ('confusion_matrix_lambd=%s'%lambd)
    plt.savefig('{}{}.pdf'.format(output_folder, plot_name),
                bbox_inches='tight')


def repro_confusion_matrix(output_folder, data1, data2, plot_title,
                           lambd, tilt=False):
    conf_matrix = confusion_matrix(data1, data2)
    conf_matrix = np.around((conf_matrix.astype('float') /
                             conf_matrix.sum(axis=1)[:, np.newaxis]),
                            decimals=2)
    # if tilt:
    #     lambd = "~ " + str(lambd)
    #     print(lambd)
    plot_confusion_matrix(output_folder, conf_matrix, plot_title,
                          lambd, tilt)
