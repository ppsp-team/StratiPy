import sys
import os
sys.path.append(os.path.abspath('../../stratipy_cluster'))
from stratipy import load_data, formatting_data
# GO Enrich
import pandas as pd
import goenrich
import goenrich.tools as tools
import numpy as np
from tqdm import trange, tqdm


def ratio(x, y):
    try:
        return x/y
    except ZeroDivisionError:
        return 0


def get_enrichment_p_count(O, interesting, min_category_size,
                           max_category_size, max_category_depth):
    query = set(interesting['GeneID'])
    options = {'min_category_size': min_category_size,
               'max_category_size': max_category_size,
               'max_category_depth': max_category_depth}
    enrichment = goenrich.enrich.analyze(O,
                                     query,
                                     'reference',
                                     **options).dropna().sort_values(by='q')

    count_bio = enrichment[
        enrichment['namespace'] == 'biological_process'].shape[0]
    count_mol = enrichment[
        enrichment['namespace'] == 'molecular_function'].shape[0]
    count_cel = enrichment[
        enrichment['namespace'] == 'cellular_component'].shape[0]

    enrichment = enrichment[enrichment['p'] < 0.05]

    p_count_bio = enrichment[
        enrichment['namespace'] == 'biological_process'].shape[0]
    p_count_mol = enrichment[
        enrichment['namespace'] == 'molecular_function'].shape[0]
    p_count_cel = enrichment[
        enrichment['namespace'] == 'cellular_component'].shape[0]

    ratio_bio = ratio(p_count_bio, count_bio)
    ratio_mol = ratio(p_count_mol, count_mol)
    ratio_cel = ratio(p_count_cel, count_cel)

#     print('\nbio {} : {} = {}'.format(count_bio, p_count_bio, ratio_bio))
#     print('mol {} : {} = {}'.format(count_mol, p_count_mol, ratio_mol))
#     print('cel {} : {} = {}'.format(count_cel, p_count_cel, ratio_cel))

#     enrichment['LOD'] = (enrichment.x / enrichment.n) / \
#                     (enrichment.N / enrichment.M - enrichment.N)
#     enrichment['Z'] = st.norm.ppf(enrichment.p)
#     enrichment['Score'] = -np.log(enrichment.p)
#     ns = enrichment.namespace.unique()
#     enrichment.loc[enrichment.namespace == ns[2],
#                           ['x', 'n', 'name', 'p', 'q', 'Score', 'Z', 'LOD']]

    return p_count_bio, p_count_mol, p_count_cel, ratio_bio, ratio_mol, ratio_cel


def merge_p_count(data_folder, ssc_mutation_data, ssc_subgroups, gene_data,
                  ppi_data, n_components, mut_type, alpha, ngh_max,
                  n_permutations, lambd, O, min_category_size,
                  max_category_size, max_category_depth):
    p_count_bio_list = []
    p_count_mol_list = []
    p_count_cel_list = []
    ratio_bio_list = []
    ratio_mol_list = []
    ratio_cel_list = []

    for k in range(n_components):
        cluster = k+1
        file_directory = (data_folder +
                          'text/clusters_EntrezGene/{}_{}_{}_{}/k={}/'.
                          format(ssc_mutation_data, ssc_subgroups, gene_data,
                                 ppi_data, n_components))
        text_file = file_directory + (
            '{}_{}_ngh={}_permut={}_lambd={}'
            .format(mut_type, alpha, ngh_max, n_permutations, lambd))
        cluster_file = text_file + '_cluster={}.txt'.format(cluster)
        interesting = pd.read_table(cluster_file, names=['GeneID'])

        (p_count_bio, p_count_mol, p_count_cel,
         ratio_bio, ratio_mol, ratio_cel) = get_enrichment_p_count(
             O, interesting, min_category_size, max_category_size,
             max_category_depth)

        p_count_bio_list.append(p_count_bio)
        p_count_mol_list.append(p_count_mol)
        p_count_cel_list.append(p_count_cel)
        ratio_bio_list.append(ratio_bio)
        ratio_mol_list.append(ratio_mol)
        ratio_cel_list.append(ratio_cel)

#     print("---")
#     print("bio ", sum(p_count_bio_list), max(ratio_bio_list) )
#     print("mol ", sum(p_count_mol_list), max(ratio_mol_list) )
#     print("cel ", sum(p_count_cel_list), max(ratio_cel_list) )

    return(sum(p_count_bio_list), max(ratio_bio_list),
           sum(p_count_mol_list), max(ratio_mol_list),
           sum(p_count_cel_list), max(ratio_cel_list))


def p_df_by_subgroup(data_folder, ssc_mutation_data, ssc_subgroups, gene_data,
                     ppi_data, n_components, mut_type, alpha, ngh_max,
                     n_permutations, lambd, O, min_category_size,
                     max_category_size, max_category_depth):
    k = []
    p_bio_list = []
    p_mol_list = []
    p_cel_list = []
    r_bio_list =[]
    r_mol_list =[]
    r_cel_list =[]

    for n_components in tqdm(range(2, 21), desc='k'):
        p_bio, r_bio, p_mol, r_mol, p_cel, r_cel = merge_p_count(
            data_folder, ssc_mutation_data, ssc_subgroups, gene_data, ppi_data,
            n_components, mut_type, alpha, ngh_max, n_permutations, lambd,
            O, min_category_size, max_category_size, max_category_depth)

        k.append(n_components)
        p_bio_list.append(p_bio)
        p_mol_list.append(p_mol)
        p_cel_list.append(p_cel)
        r_bio_list.append(r_bio)
        r_mol_list.append(r_mol)
        r_cel_list.append(r_cel)

    df = pd.DataFrame({'k': k,
                       'p_bio': p_bio_list,
                       'r_bio': r_bio_list,
                       'p_mol': p_mol_list,
                       'r_mol': r_mol_list,
                       'p_cel': p_cel_list,
                      'r_cel': r_cel_list,})
    return df


def merge_2_subgroups(data_folder, ssc_mutation_data, ssc_subgroups, gene_data,
                      ppi_data, n_components, mut_type, alpha, ngh_max,
                      n_permutations, lambd, O, min_category_size,
                      max_category_size, max_category_depth):
    ssc_subgroups = 'SSC1'
    print(ssc_subgroups)
    df1 = p_df_by_subgroup(
        data_folder, ssc_mutation_data, ssc_subgroups, gene_data, ppi_data,
        n_components, mut_type, alpha, ngh_max, n_permutations, lambd, O,
        min_category_size, max_category_size, max_category_depth)

    ssc_subgroups = 'SSC2'
    print(ssc_subgroups)
    df2 = p_df_by_subgroup(
        data_folder, ssc_mutation_data, ssc_subgroups, gene_data, ppi_data,
        n_components, mut_type, alpha, ngh_max, n_permutations, lambd, O,
        min_category_size, max_category_size, max_category_depth)

    df = df1.merge(df2, how='inner', left_on='k', right_on='k',
                   suffixes=[1, 2])

    directory = (
        data_folder + 'result_biostat_genes/' + ssc_mutation_data +
        '_' + gene_data + '/')
    os.makedirs(directory, exist_ok=True)

    file_name = '{}_lambd={}_{}.pkl'.format(mut_type, lambd, ppi_data)
    # save
    df.to_pickle('{}{}'.format(directory, file_name))


def biostat_go_enrichment(
    alpha, result_folder, mut_type, patient_data, data_folder,
    ssc_mutation_data, ssc_subgroups, gene_data, ppi_data, lambd, n_components,
    ngh_max, n_permutations):

    O = goenrich.obo.ontology(data_folder + 'go-basic.obo')
    gene2go = goenrich.read.gene2go(data_folder + 'gene2go.gz')
    min_category_size = 2
    # max_category_size = 10000
    max_category_size = 500
    max_category_depth = 100000

    mutation_profile, gene_id_patient, patient_id = (
    load_data.load_specific_SSC_mutation_profile(
        data_folder, ssc_mutation_data, ssc_subgroups, gene_data))

    gene_id_ppi, network = load_data.load_PPI_network(data_folder, ppi_data)

    network, mutation_profile, idx_ppi, idx_mut, idx_ppi_only, idx_mut_only = (
        formatting_data.classify_gene_index(
             network, mutation_profile, gene_id_ppi, gene_id_patient))

    # idx_mut : List of common genes' indexes in patients' mutation profiles.
    reference = pd.DataFrame({'GeneID': [gene_id_patient[i] for i in idx_mut]})
    background = tools.generate_background(
        gene2go, reference, 'GO_ID', 'GeneID')
    goenrich.enrich.propagate(O, background, 'reference')

    merge_2_subgroups(
        data_folder, ssc_mutation_data, ssc_subgroups, gene_data, ppi_data,
        n_components, mut_type, alpha, ngh_max, n_permutations, lambd, O,
        min_category_size, max_category_size, max_category_depth)
