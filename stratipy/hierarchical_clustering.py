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
import pandas as pd
from scipy.stats import fisher_exact, ks_2samp, chisquare, kruskal
from statistics import median
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.switch_backend('agg')


# TODO formatting for not SSC data
def distance_patients_from_consensus_file(
    result_folder, distance_patients, ppi_data, mut_type,
    influence_weight, simplification,
    alpha, tol,  keep_singletons, ngh_max, min_mutation, max_mutation,
    n_components, n_permutations, lambd, tol_nmf, linkage_method,
    patient_data, data_folder, ssc_subgroups, ssc_mutation_data, gene_data):

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
        print(' **** Same parameters file of hierarchical clustering already exists')
    else:
        # print(type(distance_patients), distance_patients.shape)
        # hierarchical clustering on distance matrix (here: distance_patients)
        Z = linkage(distance_patients, method=linkage_method)

        # Plot setting
        matplotlib.rcParams.update({'font.size': 14})
        fig = plt.figure(figsize=(20, 20))
        fig.suptitle(
            'Hierarchical clustering\n\nPatients', fontsize=30, x=0.13, y=0.95)

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
        plt.colorbar(im, cax=ax_color)

        # forms flat clusters from Z
        # given k -> maxclust
        clust_nb = fcluster(Z, n_components, criterion='maxclust')
        # cophenetic correlation distance
        coph_dist, coph_matrix = cophenet(Z, pdist(distance_patients))
        print(' ==== cophenetic correlation distance = ', coph_dist)

        ax_dendro.set_title(
            'network = {}\nalpha = {}\nmutation type = {}\ninfluence weight = {}\nsimplification = {}\ncomponent number = {}\nlambda = {}\nmethod = {}\ncophenetic corr = {}\n'
            .format(ppi_data, alpha, mut_type,
                    influence_weight, simplification,
                    n_components, lambd, linkage_method,
                    format(coph_dist, '.2f')), loc='right')

        plot_name = "similarity_matrix_Patients" + (
            '_alpha={}_tol={}_singletons={}_ngh={}_minMut={}_maxMut={}_comp={}_permut={}_lambd={}_tolNMF={}_method={}'
            .format(alpha, tol, keep_singletons, ngh_max, min_mutation,
                    max_mutation, n_components, n_permutations, lambd, tol_nmf,
                    linkage_method))
        plt.savefig('{}{}.pdf'.format(hierarchical_factorization_directory,
                                      plot_name), bbox_inches='tight')
        plt.savefig('{}{}.svg'.format(hierarchical_factorization_directory,
                                      plot_name), bbox_inches='tight')

        # start = time.time()
        savemat(hierarchical_clustering_file,
                {'Z_linkage_matrix': Z,
                 'dendrogram_data_dictionary': P,
                 'dendrogram_index': idx,
                 'flat_cluster_number': clust_nb,
                 'cophenetic_correlation_distance': coph_dist,
                 'cophenetic_correlation_matrix': coph_matrix},
                do_compression=True)
        # # end = time.time()
        # print("---------- Save time = {} ---------- {}"
        #       .format(datetime.timedelta(seconds=end-start),
        #               datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        fig = plt.figure(figsize=(3, 3))
        im = plt.imshow(D, interpolation='nearest', cmap=cm.viridis)
        plt.axis('off')
        if patient_data == 'SSC':
            fig_directory = (data_folder + 'figures/similarity/' +
                             ssc_mutation_data + '_' + ssc_subgroups + '_' + gene_data +
                             '_' + ppi_data + '/')
        else:
            fig_directory = (data_folder + 'figures/similarity/' +
                             patient_data + '_' + ppi_data + '/')
        os.makedirs(fig_directory, exist_ok=True)
        fig_name = ('{}_{}_k={}_ngh={}_permut={}_lambd={}'.format(
            mut_type, alpha, n_components, ngh_max, n_permutations, lambd))
        plt.savefig('{}{}.png'.format(fig_directory, fig_name),
                    bbox_inches='tight')


def distances_from_consensus_file(
    result_folder, distance_genes, distance_patients, ppi_data, mut_type,
    influence_weight, simplification,
    alpha, tol,  keep_singletons, ngh_max, min_mutation, max_mutation,
    n_components, n_permutations, lambd, tol_nmf, linkage_method,
    patient_data, data_folder, ssc_subgroups, ssc_mutation_data, gene_data):

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
        print(' **** Same parameters file of hierarchical clustering already exists')
    else:
        # hierarchical clustering on distance matrix (here: distance_patients)
        Z = linkage(distance_patients, method=linkage_method)
        start = time.time()
        Z_genes = linkage(distance_genes, method=linkage_method)
        end = time.time()
        print("---------- Linkage based on Gene distance = {} ---------- {}"
              .format(datetime.timedelta(seconds=end-start),
                      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        P = dendrogram(Z, count_sort='ascending', no_labels=True)
        start = time.time()
        P_genes = dendrogram(Z_genes, count_sort='ascending', no_labels=True)
        end = time.time()
        print("---------- Dendrogram based on Gene distance = {} ---------- {}"
              .format(datetime.timedelta(seconds=end-start),
                      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        idx = np.array(P['leaves'])
        idx_genes = np.array(P_genes['leaves'])

        # forms flat clusters from Z
        # given k -> maxclust
        clust_nb = fcluster(Z, n_components, criterion='maxclust')
        clust_nb_genes = fcluster(Z_genes, n_components, criterion='maxclust')

        # start = time.time()
        savemat(hierarchical_clustering_file,
                {'Z_linkage_matrix_individuals': Z,
                 'dendrogram_data_dictionary_individuals': P,
                 'dendrogram_index_individuals': idx,
                 'flat_cluster_number_individuals': clust_nb,
                 'Z_linkage_matrix_genes': Z_genes,
                 'dendrogram_data_dictionary_genes': P_genes,
                 'dendrogram_index_genes': idx_genes,
                 'flat_cluster_number_genes': clust_nb_genes},
                do_compression=True)


        D = distance_patients[idx, :][:, idx]

        fig = plt.figure(figsize=(3, 3))
        im = plt.imshow(D, interpolation='nearest', cmap=cm.viridis)
        plt.axis('off')
        if patient_data == 'SSC':
            fig_directory = (data_folder + 'figures/similarity/' +
                             ssc_mutation_data + '_' + ssc_subgroups + '_' + gene_data +
                             '_' + ppi_data + '/')
        else:
            fig_directory = (data_folder + 'figures/similarity/' +
                             patient_data + '_' + ppi_data + '/')
        os.makedirs(fig_directory, exist_ok=True)
        fig_name = ('{}_{}_k={}_ngh={}_permut={}_lambd={}'.format(
            mut_type, alpha, n_components, ngh_max, n_permutations, lambd))
        plt.savefig('{}{}.png'.format(fig_directory, fig_name),
                    bbox_inches='tight')


def round_elements_keeping_sum(float_list, benchmark_list):
    """Helper function for chi2test

    Round a list of float numbers maintaining the sum. Adjustment of the
    difference will be made on the biggest element in the list.

    Parameters
    ----------
    float_list : list (float)
        List containing decimal numbers to round off.

    benchmark_list: list
        Sum of all its elements will be a benchmark for the other.

    Returns
    -------
    round_list : list (int)
        Rounded list
    """
    round_list = [round(x) for x in float_list]
    # difference between sums of lists
    diff = sum(benchmark_list)-sum(round_list)
    # difference adjustment
    if diff != 0:
        # on the biggest element
        round_list[round_list.index(max(round_list))] += diff
    return round_list


def chi2test(list1, total_list):
    rate = sum(list1)/sum(total_list)
    expected_list1 = [x*rate for x in total_list]
    expected_list1 = round_elements_keeping_sum(expected_list1, list1)
    p_val = chisquare(list1, f_exp=expected_list1)[1]
    return p_val


def get_lists_from_clusters(data_folder, patient_data, ssc_mutation_data,
                            ssc_subgroups, ppi_data, gene_data, result_folder,
                            mut_type, influence_weight, simplification, alpha,
                            tol, keep_singletons, ngh_max, min_mutation,
                            max_mutation, n_components, n_permutations, lambd,
                            tol_nmf, linkage_method):

    hierarchical_mut_type_directory = result_folder+'hierarchical_clustering/' + mut_type + '/'
    if lambd > 0:
        hierarchical_factorization_directory = (
            hierarchical_mut_type_directory + 'gnmf/')
    else:
        hierarchical_factorization_directory = (
            hierarchical_mut_type_directory + 'nmf/')

    hierarchical_clustering_file = (
        hierarchical_factorization_directory +
        'hierarchical_clustering_Patients_weight={}_simp={}_alpha={}_tol={}_singletons={}_ngh={}_minMut={}_maxMut={}_comp={}_permut={}_lambd={}_tolNMF={}_method={}.mat'
        .format(influence_weight, simplification, alpha, tol, keep_singletons,
                ngh_max, min_mutation, max_mutation, n_components,
                n_permutations, lambd, tol_nmf, linkage_method))

    h = loadmat(hierarchical_clustering_file)
    clust_nb = np.squeeze(h['flat_cluster_number_individuals']) # cluster index for each individual
    idx = np.squeeze(h['dendrogram_index_individuals']) # individuals' index

    # load individual data including sex and IQ
    df_ssc = pd.read_csv(data_folder + '{}_indiv_sex_iq.csv'
                         .format(ssc_subgroups), sep='\t')
    ind_ssc_raw = df_ssc['individual'].tolist()
    df_ssc_iq = df_ssc[df_ssc['iq'].notnull()]
    ind_ssc_iq = df_ssc_iq['individual'].tolist()
    max_iq_value = int(df_ssc_iq['iq'].max())

    # load distance_CEU data
    df_dist = pd.read_csv(data_folder + 'SSC_distanceCEU.csv', sep="\t")
    ind_dist = df_dist.individual.tolist()

    # NOTE to do for other data
    mutation_profile_file = (
        data_folder + "{}_{}_{}_mutation_profile.mat"
        .format(ssc_mutation_data, ssc_subgroups, gene_data))
    loadfile = loadmat(mutation_profile_file)
    mutation_profile = loadfile['mutation_profile']

    clusters = list(set(clust_nb))
    total_cluster_list = []
    siblings_cluster_list = []
    probands_cluster_list = []
    female_cluster_list = []
    male_cluster_list = []
    iq_cluster_list = []
    distCEU_list = []
    mutation_nb_cluster_list = []
    ind_cluster_list = []

    for i, cluster in enumerate(clusters):
        idCluster = [i for i, c in enumerate(clust_nb) if c == cluster]
        subjs = [ind_ssc_raw[i] for i in idx[idCluster]]
        total_cluster_list.append(len(subjs))
        ind_cluster_list.append(subjs)

        # get probands/siblings count in each cluster
        sib_indiv = [i for i in subjs if i[-2:-1] == 's']  # individuals' ID list
        siblings_cluster_list.append(len(sib_indiv))
        prob_indiv = [i for i in subjs if i[-2:-1] == 'p']  # individuals' ID list
        probands_cluster_list.append(len(prob_indiv))

        # sex count in each cluster
        sex_list = [df_ssc['sex'].iloc[ind_ssc_raw.index(i)] for i in subjs if i in ind_ssc_raw]
        female_cluster_list.append(sex_list.count('female'))
        male_cluster_list.append(sex_list.count('male'))

        # get IQ list for each cluster
        iq_list = [df_ssc_iq['iq'].iloc[ind_ssc_iq.index(i)] for i in subjs if i in ind_ssc_iq]
        iq_list = [int(i) for i in iq_list]  # element type: np.float -> int
        iq_cluster_list.append(iq_list)

        # get distance CEU for each cluster
        distCEU_list.append([df_dist['distanceCEU'].iloc[ind_dist.index(i)] for i in subjs if i in ind_dist])

        # mutation number median for each cluster
        mutation_nb_list = [int(mutation_profile[ind_ssc_raw.index(i), :].sum(axis=1)) for i in subjs]
        mutation_nb_cluster_list.append(mutation_nb_list)

    # create text output file
    if patient_data == 'SSC':
        file_directory = (data_folder + 'text/clusters_stat/' +
                     ssc_mutation_data + '_' + ssc_subgroups + '_' + gene_data +
                     '_' + ppi_data + '/')
    else:
        file_directory = (data_folder + 'text/clusters_stat/' +
                         patient_data + '_' + ppi_data + '/')
    os.makedirs(file_directory, exist_ok=True)

    text_file = file_directory + (
        '{}_{}_k={}_ngh={}_permut={}_lambd={}.txt'
        .format(mut_type, alpha, n_components, ngh_max, n_permutations, lambd))

    return (total_cluster_list, probands_cluster_list, siblings_cluster_list,
            male_cluster_list, female_cluster_list, iq_cluster_list,
            distCEU_list, mutation_nb_cluster_list, text_file)


def bio_statistics(n_components, total_cluster_list, probands_cluster_list,
                   siblings_cluster_list, male_cluster_list,
                   female_cluster_list, iq_cluster_list, distCEU_list,
                   mutation_nb_cluster_list, text_file):
    p_val_threshold = 0.05

    if n_components <= 2:
        # Fisher's exact test between probands/siblings
        p_prob_sib = fisher_exact([probands_cluster_list,
                                   siblings_cluster_list])[1]

        # Fisher's exact test between sex
        p_sex = fisher_exact([male_cluster_list, female_cluster_list])[1]

        # IQ
        p_iq = ks_2samp(iq_cluster_list[0], iq_cluster_list[1])[1]

        # Distance_CEU distribution between 2 samples
        p_ancestral = ks_2samp(distCEU_list[0], distCEU_list[1])[1]

        # Mutation number median
        p_mutations = ks_2samp(mutation_nb_cluster_list[0],
                               mutation_nb_cluster_list[1])[1]

    else:
        p_prob_sib = chi2test(probands_cluster_list, total_cluster_list)

        p_sex = chi2test(male_cluster_list, total_cluster_list)

        p_iq = kruskal(*iq_cluster_list)[1]

        p_ancestral = kruskal(*distCEU_list)[1]

        p_mutations = kruskal(*mutation_nb_cluster_list)[1]

    all_p_dict = {"p_prob_sib": p_prob_sib,
                  "p_sex": p_sex,
                  "p_iq": p_iq,
                  "p_ancestral": p_ancestral,
                  "p_mutations": p_mutations}
    signif_p_dict = {k: v for k, v in all_p_dict.items()
                     if v <= p_val_threshold}
    # create output only for significant p-values
    if bool(signif_p_dict):
        with open(text_file, 'w+') as f:
            for x in signif_p_dict:
                print("{} \n{} \n".format(x, all_p_dict[x]), file=f)


def analysis_from_clusters(data_folder, patient_data, ssc_mutation_data,
                           ssc_subgroups, ppi_data, gene_data, result_folder,
                           mut_type, influence_weight,
                           simplification, alpha, tol, keep_singletons,
                           ngh_max, min_mutation, max_mutation, n_components,
                           n_permutations, lambd, tol_nmf, linkage_method):
    hierarchical_mut_type_directory = result_folder+'hierarchical_clustering/' + mut_type + '/'
    if lambd > 0:
        hierarchical_factorization_directory = (
            hierarchical_mut_type_directory + 'gnmf/')
    else:
        hierarchical_factorization_directory = (
            hierarchical_mut_type_directory + 'nmf/')

    hierarchical_clustering_file = (
        hierarchical_factorization_directory +
        'hierarchical_clustering_Patients_weight={}_simp={}_alpha={}_tol={}_singletons={}_ngh={}_minMut={}_maxMut={}_comp={}_permut={}_lambd={}_tolNMF={}_method={}.mat'
        .format(influence_weight, simplification, alpha, tol, keep_singletons,
                ngh_max, min_mutation, max_mutation, n_components,
                n_permutations, lambd, tol_nmf, linkage_method))

    h = loadmat(hierarchical_clustering_file)
    clust_nb = np.squeeze(h['flat_cluster_number_individuals']) # cluster index for each individual
    idx = np.squeeze(h['dendrogram_index_individuals']) # individuals' index

    # load individual data including sex and IQ
    df_ssc = pd.read_csv(data_folder + '{}_indiv_sex_iq.csv'
                         .format(ssc_subgroups), sep='\t')
    ind_ssc_raw = df_ssc['individual'].tolist()
    df_ssc_iq = df_ssc[df_ssc['iq'].notnull()]
    ind_ssc_iq = df_ssc_iq['individual'].tolist()
    max_iq_value = int(df_ssc_iq['iq'].max())

    # load distance_CEU data
    df_dist = pd.read_csv(data_folder + 'SSC_distanceCEU.csv', sep="\t")
    ind_dist = df_dist.individual.tolist()

    # load mutation profile data
    overall_mutation_profile_file = (
        data_folder + "{}_overall_mutation_profile.mat".format(ssc_mutation_data))
    loadfile = loadmat(overall_mutation_profile_file)
    mutation_profile = loadfile['mutation_profile']
    indiv = (loadfile['indiv'].flatten()).tolist()

    clusters = list(set(clust_nb))
    total_cluster_list = []
    siblings_cluster_list = []
    probands_cluster_list = []
    female_cluster_list = []
    male_cluster_list = []
    iq_cluster_list = []
    distCEU_list = []
    mutation_nb_cluster_list = []
    ind_cluster_list = []

    for i, cluster in enumerate(clusters):
        idCluster = [i for i, c in enumerate(clust_nb) if c == cluster]
        subjs = [indiv[i] for i in idx[idCluster]]
        total_cluster_list.append(len(subjs))
        ind_cluster_list.append(subjs)

        sib_indiv = [i for i in subjs if i[-2:-1] == 's'] # individuals' ID list
        siblings_cluster_list.append(len(sib_indiv))
        prob_indiv = [i for i in subjs if i[-2:-1] == 'p'] # individuals' ID list
        probands_cluster_list.append(len(prob_indiv))

        # sex count in each cluster
        sex_list = [df_ssc['sex'].iloc[ind_ssc_raw.index(i)] for i in subjs if i in ind_ssc_raw]
        female_cluster_list.append(sex_list.count('female'))
        male_cluster_list.append(sex_list.count('male'))

        # get IQ list for each cluster
        iq_list = [df_ssc_iq['iq'].iloc[ind_ssc_iq.index(i)] for i in subjs if i in ind_ssc_iq]
        iq_list = [int(i) for i in iq_list] # element type: np.float -> int
        # create frequency (count individuals for each IQ) list
#         iq_count_list = [0]*(max_iq_value+1)
#         for j in iq_list:
#             iq_count_list[j] = iq_list.count(j)
#         iq_cluster_list.append(iq_count_list)
        iq_cluster_list.append(iq_list)

        # get distance CEU for each cluster
        distCEU_list.append([df_dist['distanceCEU'].iloc[ind_dist.index(i)] for i in subjs if i in ind_dist])

        # mutation number median for each cluster
        mutation_nb_list = [int(mutation_profile[indiv.index(i), :].sum(axis=1)) for i in subjs]
        mutation_nb_cluster_list.append(mutation_nb_list)

    if patient_data == 'SSC':
        file_directory = (data_folder + 'text/clusters_stat/{}_{}_{}_{}/'.
                          format(ssc_mutation_data, ssc_subgroups, gene_data,
                                 ppi_data))
    else:
        file_directory = (data_folder + 'text/clusters_stat/{}_{}/'.
                          format(patient_data, ppi_data))
    os.makedirs(file_directory, exist_ok=True)

    text_file = file_directory + (
        '{}_{}_k={}_ngh={}_permut={}_lambd={}.txt'
        .format(mut_type, alpha, n_components, ngh_max, n_permutations, lambd))
    print(text_file)
    # create text output file
    with open(text_file, 'w+') as f:
        # Individual numbers in clusters
        print("individuals: \n{} ({}%) / {} ({}%)"
          .format(total_cluster_list[0], round(total_cluster_list[0]*100/len(idx), 1),
                 total_cluster_list[1], round(total_cluster_list[1]*100/len(idx), 1)), file=f)

        p_val_threshold = 0.05
        # Fisher's exact test between probands/siblings
        prob_sib = [probands_cluster_list, siblings_cluster_list]
        p = fisher_exact(prob_sib)[1]
        if p <= p_val_threshold:
            print("\nProbands / Siblings (Fisher's exact):\n{}".format(p), file=f)

        # Fisher's exact test between sex
        male_female = [male_cluster_list, female_cluster_list]
        p = fisher_exact(male_female)[1]
        if p <= p_val_threshold:
            print("\nMales / Females (Fisher's exact):\n{}".format(p), file=f)

        # Distance_CEU distribution between 2 samples
        p = ks_2samp(distCEU_list[0], distCEU_list[1])[1]
        if p <= p_val_threshold:
            print("\nDistance_CEU (Kolmogorov-Smirnov):\n{}".format(p), file=f)

        # IQ
        iq_array = np.array(iq_cluster_list)
        p = ks_2samp(iq_array[0], iq_array[1])[1]
        if p <= p_val_threshold:
            print("\nProbands' IQ median: \n{} / {}".format(median(iq_array[0]), median(iq_array[1])), file=f)
            print("{}".format(p), file=f)

        # Mutation number median
        p = ks_2samp(mutation_nb_cluster_list[0], mutation_nb_cluster_list[1])[1]
        if p <= p_val_threshold:
            print("\nMutation number median: \n{} / {}".format(median(mutation_nb_cluster_list[0]),
                                                                         median(mutation_nb_cluster_list[1])), file=f)
            print("{}".format(p), file=f)

    # return ind_cluster_list, iq_array, siblings_cluster_list, probands_cluster_list, female_cluster_list, male_cluster_list


def get_entrezgene_from_cluster(data_folder, result_folder, ssc_mutation_data,
                                patient_data, ssc_subgroups, alpha, n_components,
                                ngh_max, n_permutations, lambd, influence_weight,
                                simplification, tol, keep_singletons, min_mutation,
                                max_mutation, tol_nmf, linkage_method,
                                gene_data, ppi_data, gene_id_ppi, idx_ppi,
                                idx_ppi_only, mut_type):
    hierarchical_mut_type_directory = result_folder+'hierarchical_clustering/' + mut_type + '/'
    if lambd > 0:
        hierarchical_factorization_directory = (
            hierarchical_mut_type_directory + 'gnmf/')
    else:
        hierarchical_factorization_directory = (
            hierarchical_mut_type_directory + 'nmf/')

    hierarchical_clustering_file = (
        hierarchical_factorization_directory +
        'hierarchical_clustering_Patients_weight={}_simp={}_alpha={}_tol={}_singletons={}_ngh={}_minMut={}_maxMut={}_comp={}_permut={}_lambd={}_tolNMF={}_method={}.mat'
        .format(influence_weight, simplification, alpha, tol, keep_singletons,
                ngh_max, min_mutation, max_mutation, n_components,
                n_permutations, lambd, tol_nmf, linkage_method))

    h = loadmat(hierarchical_clustering_file)
    clust_nb_genes = np.squeeze(h['flat_cluster_number_genes'])  # cluster index for each gene
    idx_genes = np.squeeze(h['dendrogram_index_genes'])  # gene's index

    # EntrezGene ID to int
    entrez_ppi = [int(i) for i in gene_id_ppi]
    # EntrezGene indexes in PPI after formatting
    idx_filtred = idx_ppi + idx_ppi_only

    # EntrezGene ID
    df_id_ref = pd.DataFrame({'ref_idx': list(range(len(entrez_ppi))),
                              'entrez_id': entrez_ppi})
    # Indexes after formatting
    df_filtered = pd.DataFrame({'filt_idx': list(range(len(idx_filtred))),
                               'idx_in_ppi': idx_filtred})

    # Dendrogram results: clusters
    df_dendro = pd.DataFrame({'dendro_idx': idx_genes.tolist(),
                                 'cluster': clust_nb_genes.tolist()})

    # link EntrezGene and Indexes in PPI
    df1 = df_filtered.merge(df_id_ref, how='inner', left_on='idx_in_ppi', right_on='ref_idx')
    # link EntrezGene and Cluster number
    df2 = df1.merge(df_dendro, how='inner', left_on='filt_idx', right_on='dendro_idx')

    # create text output file
    if patient_data == 'SSC':
        file_directory = (data_folder + 'text/clusters_EntrezGene/{}_{}_{}_{}/k={}/'.
                          format(ssc_mutation_data, ssc_subgroups, gene_data,
                                 ppi_data, n_components))
    else:
        file_directory = (data_folder + 'text/clusters_EntrezGene/{}_{}/k={}/'.
                          format(patient_data, ppi_data, n_components))
    os.makedirs(file_directory, exist_ok=True)

    text_file = file_directory + (
        '{}_{}_ngh={}_permut={}_lambd={}'
        .format(mut_type, alpha, ngh_max, n_permutations, lambd))

    # Slice dataframe by cluster then save EntrezGene ID in .txt file
    for k in range(n_components):
        cluster = k+1
        with open(text_file + '_cluster={}.txt'.format(cluster), 'w') as f:
            entrez_list = df2[df2['cluster'] == cluster]['entrez_id'].tolist()
            for i in entrez_list:
                f.write("{}\n".format(i))


def stacked_bar_plot(n_components, siblings_cluster_list, probands_cluster_list, odd, p_val,
                     female_cluster_list, male_cluster_list):
    fig = plt.figure()
    ax = plt.subplot(111)
    X = range(n_components)

    ax.bar(X, probands_cluster_list, label='Probands', align='edge', width=-0.3, edgecolor='w', color='gray',
           bottom=siblings_cluster_list)
    ax.bar(X, siblings_cluster_list, label='Siblings', align='edge', width=-0.3, edgecolor='w', color='lightgray')

    ax.bar(X, male_cluster_list, label='Male', align='edge', width=0.3, edgecolor='w', color='steelblue',
           bottom=female_cluster_list)
    ax.bar(X, female_cluster_list, label='Female', align='edge', width=0.3, edgecolor='w', color='lightcoral')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     ax.set_title("{} - {}: {}\n---------------------------\nFisher's exact test\nodds ratio = {:0.2e}\np-value = {:0.2e}"
#                  .format(ssc_type, ssc_subgroups, mut_type, round(odd), (p_val)))
    ax.set_title("{} - {}: {}\n".format(ssc_type, ssc_subgroups, mut_type))
    plt.xticks(X, list(range(1,n_components+1)))
    plt.show()
