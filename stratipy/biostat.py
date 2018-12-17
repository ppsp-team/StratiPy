import sys
import os
sys.path.append(os.path.abspath('../../stratipy_cluster'))
from stratipy import hierarchical_clustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
import numpy as np
from scipy.io import loadmat
import pandas as pd
from scipy.stats import fisher_exact, ks_2samp, chisquare, kruskal
from statistics import median


def biostatistics_file(result_folder, mut_type, influence_weight,
                       simplification, alpha, tol, keep_singletons, ngh_max,
                       min_mutation, max_mutation, n_components,
                       n_permutations, lambd, tol_nmf, linkage_method):
    biostat_directory = result_folder+'biostat/'
    biostat_mut_type_directory = biostat_directory + mut_type + '/'

    if lambd > 0:
        biostat_factorization_directory = (
            biostat_mut_type_directory + 'gnmf/')
    else:
        biostat_factorization_directory = (
            biostat_mut_type_directory + 'nmf/')
    os.makedirs(biostat_factorization_directory, exist_ok=True)

    biostat_file = (
        biostat_factorization_directory +
        'biostat_weight={}_simp={}_alpha={}_tol={}_singletons={}_ngh={}_minMut={}_maxMut={}_comp={}_permut={}_lambd={}_tolNMF={}_method={}.pkl'
        .format(influence_weight, simplification, alpha, tol, keep_singletons,
                ngh_max, min_mutation, max_mutation, n_components,
                n_permutations, lambd, tol_nmf, linkage_method))

    return biostat_factorization_directory, biostat_file


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


def load_SSC_clinical_data(data_folder, ssc_mutation_data, ssc_subgroups, gene_data):
    # load individual data including sex and IQ
    df_ssc = pd.read_csv(data_folder + '{}_phenotype.csv'
                         .format(ssc_subgroups), sep='\t')
    ind_ssc_raw = df_ssc['individual'].tolist()
#     # get only not null IQ data
#     df_ssc_iq = df_ssc[df_ssc['iq'].notnull()]
#     ind_ssc_iq = df_ssc_iq['individual'].tolist()
#     max_iq_value = int(df_ssc_iq['iq'].max())

    # load distance_CEU data
    df_dist = pd.read_csv(data_folder + 'SSC_distanceCEU.csv', sep="\t")
    ind_dist = df_dist.individual.tolist()

    # NOTE to do for other data
    mutation_profile_file = (
        data_folder + "{}_{}_{}_mutation_profile.mat"
        .format(ssc_mutation_data, ssc_subgroups, gene_data))
    loadfile = loadmat(mutation_profile_file)
    mutation_profile = loadfile['mutation_profile']

#     return (df_ssc, ind_ssc_raw, df_ssc_iq, ind_ssc_iq, max_iq_value, df_dist,
#             ind_dist, mutation_profile)
    return (df_ssc, ind_ssc_raw, df_dist, ind_dist, mutation_profile)


def get_individual_lists_from_clusters(hierarchical_clustering_file,
                                       data_folder, patient_data,
                                       ssc_mutation_data, ssc_subgroups,
                                       ppi_data, gene_data, mut_type, alpha,
                                       ngh_max, n_components, n_permutations,
                                       lambd):

    h = loadmat(hierarchical_clustering_file)
    clust_nb_indiv = np.squeeze(h['flat_cluster_number_individuals'])
    idx_indiv = np.squeeze(h['dendrogram_index_individuals'])

#     (df_ssc, ind_ssc_raw, df_ssc_iq, ind_ssc_iq, max_iq_value, df_dist,
#      ind_dist, mutation_profile) = load_SSC_clinical_data(
#          data_folder, ssc_mutation_data, ssc_subgroups, gene_data)
    (df_ssc, ind_ssc_raw, df_dist, ind_dist,
     mutation_profile) = load_SSC_clinical_data(
        data_folder, ssc_mutation_data, ssc_subgroups, gene_data)

    clusters = list(set(clust_nb_indiv))
    total_cluster_list = []
    ind_cluster_list = []
    siblings_cluster_list = []
    probands_cluster_list = []
    female_cluster_list = []
    male_cluster_list = []
    iq_cluster_list = []
    srs_cluster_list = []
    vineland_cluster_list = []
    distCEU_list = []
    mutation_nb_cluster_list = []

    # for each cluster
    for i, cluster in enumerate(clusters):
        idCluster = [i for i, c in enumerate(clust_nb_indiv) if c == cluster]
        subjs = [ind_ssc_raw[i] for i in idx_indiv[idCluster]]
        total_cluster_list.append(len(subjs))
        ind_cluster_list.append(subjs)

        # get probands/siblings count
        sib_indiv = [i for i in subjs if i[-2:-1] == 's']  # individuals' ID list
        siblings_cluster_list.append(len(sib_indiv))
        prob_indiv = [i for i in subjs if i[-2:-1] == 'p']  # individuals' ID list
        probands_cluster_list.append(len(prob_indiv))

        # sex count
        sex_list = [df_ssc['sex'].iloc[ind_ssc_raw.index(i)] for i in subjs if i in ind_ssc_raw]
        female_cluster_list.append(sex_list.count('female'))
        male_cluster_list.append(sex_list.count('male'))

        # get IQ list
#         iq_list = [df_ssc_iq['iq'].iloc[ind_ssc_iq.index(i)] for i in subjs if i in ind_ssc_iq]
#         iq_list = [int(i) for i in iq_list]  # element type: np.float -> int
#         iq_cluster_list.append(iq_list)
        iq_list = [df_ssc['iq'].iloc[ind_ssc_raw.index(i)] for i in subjs if i in ind_ssc_raw]
        iq_cluster_list.append(iq_list)

        # get SRS list
        srs_list = [df_ssc['srs'].iloc[ind_ssc_raw.index(i)] for i in subjs if i in ind_ssc_raw]
        srs_cluster_list.append(srs_list)

        # get Vineland II list
        vineland_list = [df_ssc['vineland'].iloc[ind_ssc_raw.index(i)] for i in subjs if i in ind_ssc_raw]
        vineland_cluster_list.append(vineland_list)

        # get distance CEU for each cluster
        distCEU_list.append([df_dist['distanceCEU'].iloc[ind_dist.index(i)] for i in subjs if i in ind_dist])

        # mutation number median for each cluster
        mutation_nb_list = [int(mutation_profile[ind_ssc_raw.index(i), :].sum(axis=1)) for i in subjs]
        mutation_nb_cluster_list.append(mutation_nb_list)

    # Siblings and Probands rapports
    # np.float64() returns nan if division by 0
    sib_vs_total = [np.float64(indiv)/total for indiv, total
                    in zip(siblings_cluster_list, total_cluster_list)]
    prob_vs_total = [np.float64(indiv)/total for indiv, total
                     in zip(probands_cluster_list, total_cluster_list)]
    sib_within_sibtotal = [np.float64(i)/sum(siblings_cluster_list) for i
                           in siblings_cluster_list]
    prob_within_probtotal = [np.float64(i)/sum(probands_cluster_list) for i
                             in probands_cluster_list]
    # sex rapports
    female_vs_total = [np.float64(indiv)/total for indiv, total
                       in zip(female_cluster_list, total_cluster_list)]
    male_vs_total = [np.float64(indiv)/total for indiv, total
                     in zip(male_cluster_list, total_cluster_list)]
    female_within_femaletotal = [np.float64(i)/sum(female_cluster_list) for i
                                 in female_cluster_list]
    male_within_maletotal = [np.float64(i)/sum(male_cluster_list) for i
                             in male_cluster_list]
    # list of median values for each cluster, ignoring NaN values
    iq_cluster_median = [np.nanmedian(i) for i in iq_cluster_list]
    srs_cluster_median = [np.nanmedian(i) for i in srs_cluster_list]
    vineland_cluster_median = [np.nanmedian(i) for i in vineland_cluster_list]
    distCEU_cluster_median = [np.nanmedian(i) for i in distCEU_list]
    mutation_nb_cluster_median = [np.nanmedian(i) for i in mutation_nb_cluster_list]


    # create text output file
    if patient_data == 'SSC':
        file_directory = (data_folder + 'text/clusters_stat/' +
                          ssc_mutation_data + '_' + ssc_subgroups + '_' +
                          gene_data + '_' + ppi_data + '/')
    else:
        file_directory = (data_folder + 'text/clusters_stat/' + patient_data +
                          '_' + ppi_data + '/')
    os.makedirs(file_directory, exist_ok=True)

    text_file = file_directory + (
        '{}_{}_k={}_ngh={}_permut={}_lambd={}.txt'
        .format(mut_type, alpha, n_components, ngh_max, n_permutations, lambd))

    return (total_cluster_list, ind_cluster_list, siblings_cluster_list,
            probands_cluster_list, female_cluster_list, male_cluster_list,
            iq_cluster_list, srs_cluster_list, vineland_cluster_list,
            distCEU_list, mutation_nb_cluster_list, sib_vs_total, prob_vs_total,
            sib_within_sibtotal, prob_within_probtotal, female_vs_total,
            male_vs_total, female_within_femaletotal, male_within_maletotal,
            iq_cluster_median, srs_cluster_median, vineland_cluster_median,
            distCEU_cluster_median, mutation_nb_cluster_median, text_file)


def individual_cluster_analysis(n_components, total_cluster_list,
                                probands_cluster_list, siblings_cluster_list,
                                male_cluster_list, female_cluster_list,
                                iq_cluster_list, srs_cluster_list,
                                vineland_cluster_list, distCEU_list,
                                mutation_nb_cluster_list, text_file,
                                p_val_threshold):
    if n_components <= 2:
        # Fisher's exact test between probands/siblings
        p_prob_sib = fisher_exact([probands_cluster_list,
                                   siblings_cluster_list])[1]
        # Fisher's exact test between sex
        p_sex = fisher_exact([male_cluster_list, female_cluster_list])[1]
        # Kolmogorov-Smirnov statistic on 2 samples of IQ
        p_iq = ks_2samp(iq_cluster_list[0], iq_cluster_list[1])[1]
        # SRS
        p_srs = ks_2samp(srs_cluster_list[0], srs_cluster_list[1])[1]
        # Vineland II
        p_vineland = ks_2samp(vineland_cluster_list[0], vineland_cluster_list[1])[1]
        # Distance_CEU distribution
        p_ancestral = ks_2samp(distCEU_list[0], distCEU_list[1])[1]
        # Mutation number median
        p_mutations = ks_2samp(mutation_nb_cluster_list[0],
                               mutation_nb_cluster_list[1])[1]

    else:
        p_prob_sib = chi2test(probands_cluster_list, total_cluster_list)
        p_sex = chi2test(male_cluster_list, total_cluster_list)

        #  ‘omit’ performs the calculations ignoring nan values
        if sum([sum(~np.isnan(i)) for i in iq_cluster_list]) != 0:
            p_iq = kruskal(*iq_cluster_list, nan_policy='omit')[1]
        else:
            print(" no IQ score")
            p_iq = np.nan
        p_srs = kruskal(*srs_cluster_list, nan_policy='omit')[1]
        p_vineland = kruskal(*vineland_cluster_list, nan_policy='omit')[1]
        p_ancestral = kruskal(*distCEU_list)[1]
        p_mutations = kruskal(*mutation_nb_cluster_list)[1]

    all_p_dict = {"p_prob_sib": p_prob_sib,
                  "p_sex": p_sex,
                  "p_iq": p_iq,
                  "p_srs": p_srs,
                  "p_vineland": p_vineland,
                  "p_ancestral": p_ancestral,
                  "p_mutations": p_mutations}
    signif_p_dict = {k: v for k, v in all_p_dict.items()
                     if v <= p_val_threshold}
    # create output only for significant p-values
    if bool(signif_p_dict):
        with open(text_file, 'w+') as f:
            for x in signif_p_dict:
                print("{} \n{} \n".format(x, all_p_dict[x]), file=f)

    return p_prob_sib, p_sex, p_iq, p_srs, p_vineland, p_ancestral, p_mutations


def biostat_individuals(hierarchical_clustering_file, biostat_file,
                        data_folder, patient_data, ssc_mutation_data,
                        ssc_subgroups, ppi_data, gene_data, mut_type, alpha,
                        ngh_max, n_components, n_permutations, lambd,
                        p_val_threshold):
    existance_same_param = os.path.exists(biostat_file)

    if existance_same_param:
        print(' **** Same parameters file of biostat already exists')
    else:
        (total_cluster_list, ind_cluster_list, siblings_cluster_list,
         probands_cluster_list, female_cluster_list, male_cluster_list,
         iq_cluster_list, srs_cluster_list, vineland_cluster_list,
         distCEU_list, mutation_nb_cluster_list, sib_vs_total, prob_vs_total,
         sib_within_sibtotal, prob_within_probtotal, female_vs_total,
         male_vs_total, female_within_femaletotal, male_within_maletotal,
         iq_cluster_median, srs_cluster_median, vineland_cluster_median,
         distCEU_cluster_median, mutation_nb_cluster_median,
         text_file) = get_individual_lists_from_clusters(
             hierarchical_clustering_file, data_folder, patient_data,
             ssc_mutation_data, ssc_subgroups, ppi_data, gene_data, mut_type,
             alpha, ngh_max, n_components, n_permutations, lambd)

        (p_prob_sib, p_sex, p_iq, p_srs, p_vineland, p_ancestral,
         p_mutations) = individual_cluster_analysis(
             n_components, total_cluster_list, probands_cluster_list,
             siblings_cluster_list, male_cluster_list, female_cluster_list,
             iq_cluster_list, srs_cluster_list, vineland_cluster_list,
             distCEU_list, mutation_nb_cluster_list, text_file,
             p_val_threshold)

        if lambd == 0:
            nmf = 'NMF'
        else:
            nmf = 'GNMF'
        # save
        df = pd.DataFrame(
            {'data_gene': gene_data,
             'data_ssc': ssc_subgroups,
             'data_ppi': ppi_data,
             'data_mut_type': mut_type,
             'data_nmf': nmf,
             'data_k': n_components,

             'ind_count': [total_cluster_list],
             'ind_id': [ind_cluster_list],

             'sp_sib': [siblings_cluster_list],
             'sp_prob': [probands_cluster_list],
             'sp_sib_Total': [sib_vs_total],
             'sp_prob_Total': [prob_vs_total],
             'sp_sib_sTotal': [sib_within_sibtotal],
             'sp_prob_pTotal': [prob_within_probtotal],
             'sp_pval': p_prob_sib,

             'sex_f': [female_cluster_list],
             'sex_m': [male_cluster_list],
             'sex_f_Total': [female_vs_total],
             'sex_m_Total': [male_vs_total],
             'sex_f_fTotal': [female_within_femaletotal],
             'sex_m_mTotal': [male_within_maletotal],
             'sex_pval': p_sex,

             'iq_median': [iq_cluster_median],
             'iq_pval': p_iq,

             'srs_median': [srs_cluster_median],
             'srs_pval': p_srs,

             'vineland_median': [vineland_cluster_median],
             'vineland_pval': p_vineland,

             'distCEU_median': [distCEU_cluster_median],
             'distCEU_pval': p_ancestral,

             'mutation_nb_median': [mutation_nb_cluster_median],
             'mutation_pval': p_mutations})

        df.to_pickle(biostat_file)


def get_entrezgene_from_cluster(hierarchical_clustering_file, data_folder,
                                ssc_mutation_data, patient_data, ssc_subgroups,
                                alpha, n_components, ngh_max, n_permutations,
                                lambd, gene_data, ppi_data, gene_id_ppi,
                                idx_ppi, idx_ppi_only, mut_type):
    h = loadmat(hierarchical_clustering_file)
    # cluster index for each gene
    clust_nb_genes = np.squeeze(h['flat_cluster_number_genes'])
    # gene's index
    idx_genes = np.squeeze(h['dendrogram_index_genes'])

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
    df1 = df_filtered.merge(
        df_id_ref, how='inner', left_on='idx_in_ppi', right_on='ref_idx')
    # link EntrezGene and Cluster number
    df2 = df1.merge(
        df_dendro, how='inner', left_on='filt_idx', right_on='dendro_idx')

    # create text output file
    if patient_data == 'SSC':
        file_directory = (
            data_folder + 'text/clusters_EntrezGene/{}_{}_{}_{}/k={}/'.
             format(ssc_mutation_data, ssc_subgroups, gene_data, ppi_data,
                    n_components))
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


def biostat_analysis(data_folder, result_folder, patient_data,
                     ssc_mutation_data, ssc_subgroups, ppi_data, gene_data,
                     mut_type, influence_weight, simplification, alpha, tol,
                     keep_singletons, ngh_max, min_mutation, max_mutation,
                     n_components, n_permutations, lambd, tol_nmf,
                     linkage_method, p_val_threshold, gene_id_ppi,
                     idx_ppi, idx_ppi_only):
    hierarchical_clustering_file = hierarchical_clustering.hierarchical_file(
        result_folder, mut_type, influence_weight, simplification, alpha, tol,
        keep_singletons, ngh_max, min_mutation, max_mutation, n_components,
        n_permutations, lambd, tol_nmf, linkage_method)

    biostat_factorization_directory, biostat_file = biostatistics_file(
        result_folder, mut_type, influence_weight, simplification, alpha, tol,
        keep_singletons, ngh_max, min_mutation, max_mutation, n_components,
        n_permutations, lambd, tol_nmf, linkage_method)

    biostat_individuals(
        hierarchical_clustering_file, biostat_file, data_folder,
        patient_data, ssc_mutation_data, ssc_subgroups, ppi_data, gene_data,
        mut_type, alpha, ngh_max, n_components, n_permutations, lambd,
        p_val_threshold)

    get_entrezgene_from_cluster(
        hierarchical_clustering_file, data_folder, ssc_mutation_data,
        patient_data, ssc_subgroups, alpha, n_components, ngh_max,
        n_permutations, lambd, gene_data, ppi_data, gene_id_ppi, idx_ppi,
        idx_ppi_only, mut_type)



# def analysis_from_clusters(data_folder, patient_data, ssc_mutation_data,
#                            ssc_subgroups, ppi_data, gene_data, result_folder,
#                            mut_type, influence_weight,
#                            simplification, alpha, tol, keep_singletons,
#                            ngh_max, min_mutation, max_mutation, n_components,
#                            n_permutations, lambd, tol_nmf, linkage_method):
#     hierarchical_mut_type_directory = result_folder+'hierarchical_clustering/' + mut_type + '/'
#     if lambd > 0:
#         hierarchical_factorization_directory = (
#             hierarchical_mut_type_directory + 'gnmf/')
#     else:
#         hierarchical_factorization_directory = (
#             hierarchical_mut_type_directory + 'nmf/')
#
#     hierarchical_clustering_file = (
#         hierarchical_factorization_directory +
#         'hierarchical_clustering_Patients_weight={}_simp={}_alpha={}_tol={}_singletons={}_ngh={}_minMut={}_maxMut={}_comp={}_permut={}_lambd={}_tolNMF={}_method={}.mat'
#         .format(influence_weight, simplification, alpha, tol, keep_singletons,
#                 ngh_max, min_mutation, max_mutation, n_components,
#                 n_permutations, lambd, tol_nmf, linkage_method))
#
#     h = loadmat(hierarchical_clustering_file)
#     clust_nb = np.squeeze(h['flat_cluster_number_individuals']) # cluster index for each individual
#     idx = np.squeeze(h['dendrogram_index_individuals']) # individuals' index
#
#     # load individual data including sex and IQ
#     df_ssc = pd.read_csv(data_folder + '{}_indiv_sex_iq.csv'
#                          .format(ssc_subgroups), sep='\t')
#     ind_ssc_raw = df_ssc['individual'].tolist()
#     df_ssc_iq = df_ssc[df_ssc['iq'].notnull()]
#     ind_ssc_iq = df_ssc_iq['individual'].tolist()
#     max_iq_value = int(df_ssc_iq['iq'].max())
#
#     # load distance_CEU data
#     df_dist = pd.read_csv(data_folder + 'SSC_distanceCEU.csv', sep="\t")
#     ind_dist = df_dist.individual.tolist()
#
#     # load mutation profile data
#     overall_mutation_profile_file = (
#         data_folder + "{}_overall_mutation_profile.mat".format(ssc_mutation_data))
#     loadfile = loadmat(overall_mutation_profile_file)
#     mutation_profile = loadfile['mutation_profile']
#     indiv = (loadfile['indiv'].flatten()).tolist()
#
#     clusters = list(set(clust_nb))
#     total_cluster_list = []
#     siblings_cluster_list = []
#     probands_cluster_list = []
#     female_cluster_list = []
#     male_cluster_list = []
#     iq_cluster_list = []
#     distCEU_list = []
#     mutation_nb_cluster_list = []
#     ind_cluster_list = []
#
#     for i, cluster in enumerate(clusters):
#         idCluster = [i for i, c in enumerate(clust_nb) if c == cluster]
#         subjs = [indiv[i] for i in idx[idCluster]]
#         total_cluster_list.append(len(subjs))
#         ind_cluster_list.append(subjs)
#
#         sib_indiv = [i for i in subjs if i[-2:-1] == 's'] # individuals' ID list
#         siblings_cluster_list.append(len(sib_indiv))
#         prob_indiv = [i for i in subjs if i[-2:-1] == 'p'] # individuals' ID list
#         probands_cluster_list.append(len(prob_indiv))
#
#         # sex count in each cluster
#         sex_list = [df_ssc['sex'].iloc[ind_ssc_raw.index(i)] for i in subjs if i in ind_ssc_raw]
#         female_cluster_list.append(sex_list.count('female'))
#         male_cluster_list.append(sex_list.count('male'))
#
#         # get IQ list for each cluster
#         iq_list = [df_ssc_iq['iq'].iloc[ind_ssc_iq.index(i)] for i in subjs if i in ind_ssc_iq]
#         iq_list = [int(i) for i in iq_list] # element type: np.float -> int
#         # create frequency (count individuals for each IQ) list
# #         iq_count_list = [0]*(max_iq_value+1)
# #         for j in iq_list:
# #             iq_count_list[j] = iq_list.count(j)
# #         iq_cluster_list.append(iq_count_list)
#         iq_cluster_list.append(iq_list)
#
#         # get distance CEU for each cluster
#         distCEU_list.append([df_dist['distanceCEU'].iloc[ind_dist.index(i)] for i in subjs if i in ind_dist])
#
#         # mutation number median for each cluster
#         mutation_nb_list = [int(mutation_profile[indiv.index(i), :].sum(axis=1)) for i in subjs]
#         mutation_nb_cluster_list.append(mutation_nb_list)
#
#     if patient_data == 'SSC':
#         file_directory = (data_folder + 'text/clusters_stat/{}_{}_{}_{}/'.
#                           format(ssc_mutation_data, ssc_subgroups, gene_data,
#                                  ppi_data))
#     else:
#         file_directory = (data_folder + 'text/clusters_stat/{}_{}/'.
#                           format(patient_data, ppi_data))
#     os.makedirs(file_directory, exist_ok=True)
#
#     text_file = file_directory + (
#         '{}_{}_k={}_ngh={}_permut={}_lambd={}.txt'
#         .format(mut_type, alpha, n_components, ngh_max, n_permutations, lambd))
#     print(text_file)
#     # create text output file
#     with open(text_file, 'w+') as f:
#         # Individual numbers in clusters
#         print("individuals: \n{} ({}%) / {} ({}%)"
#           .format(total_cluster_list[0], round(total_cluster_list[0]*100/len(idx), 1),
#                  total_cluster_list[1], round(total_cluster_list[1]*100/len(idx), 1)), file=f)
#
#         p_val_threshold = 0.05
#         # Fisher's exact test between probands/siblings
#         prob_sib = [probands_cluster_list, siblings_cluster_list]
#         p = fisher_exact(prob_sib)[1]
#         if p <= p_val_threshold:
#             print("\nProbands / Siblings (Fisher's exact):\n{}".format(p), file=f)
#
#         # Fisher's exact test between sex
#         male_female = [male_cluster_list, female_cluster_list]
#         p = fisher_exact(male_female)[1]
#         if p <= p_val_threshold:
#             print("\nMales / Females (Fisher's exact):\n{}".format(p), file=f)
#
#         # Distance_CEU distribution between 2 samples
#         p = ks_2samp(distCEU_list[0], distCEU_list[1])[1]
#         if p <= p_val_threshold:
#             print("\nDistance_CEU (Kolmogorov-Smirnov):\n{}".format(p), file=f)
#
#         # IQ
#         iq_array = np.array(iq_cluster_list)
#         p = ks_2samp(iq_array[0], iq_array[1])[1]
#         if p <= p_val_threshold:
#             print("\nProbands' IQ median: \n{} / {}".format(median(iq_array[0]), median(iq_array[1])), file=f)
#             print("{}".format(p), file=f)
#
#         # Mutation number median
#         p = ks_2samp(mutation_nb_cluster_list[0], mutation_nb_cluster_list[1])[1]
#         if p <= p_val_threshold:
#             print("\nMutation number median: \n{} / {}".format(median(mutation_nb_cluster_list[0]),
#                                                                          median(mutation_nb_cluster_list[1])), file=f)
#             print("{}".format(p), file=f)
#
#     # return ind_cluster_list, iq_array, siblings_cluster_list, probands_cluster_list, female_cluster_list, male_cluster_list
