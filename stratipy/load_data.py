import sys
import os
from scipy.io import loadmat, savemat
import scipy.sparse as sp
import numpy as np
import pandas as pd
from numpy import genfromtxt
from stratipy.nbs_class import Ppi

# NOTE some variable names changed:
# dataFolder -> data_folder
# net -> network, ids -> gene_id_ppi,
# mutations -> mutation_profile, genes -> gene_id_patient
# geneSymbol_profile -> gene_symbol_profile
# subnet -> idx_ppi, good -> idx_mut, subnetNotmutated ->idx_ppi_only,
# bad -> idx_mut_only
# nnnet -> ppi_total, nnmut -> mut_total
# nnnetFiltered -> ppi_filt, nnmut-> mut_filt


# @profile
def load_TCGA_UCEC_patient_data(data_folder):
    # TODO patients' ID, phenotypes in dictionary of dictionary or ...?
    print(" ==== TCGA patients' ID  ")
    phenotypes = loadmat(data_folder+'UCEC_clinical_phenotype.mat')
    patient_id = [c[0][0] for c in phenotypes['UCECppheno'][0][0][0]]

    # mutation profiles
    print(' ==== TCGA mutation profiles  ')
    somatic = loadmat(data_folder+'somatic_data_UCEC.mat')
    mutation_profile = sp.csc_matrix(somatic['gene_indiv_mat']
                                     .astype(np.float32))

    # Entrez gene ID and gene symbols in mutation profiles
    print(' ==== TCGA Entrez gene ID and gene symbols in mutation profiles  ')
    gene_id_patient = [x[0] for x in somatic['gene_id_all']]
    gene_symbol_profile = [x[0][0] for x in somatic['gene_id_symbol']]
    # dictionnary = key:entrez gene ID, value:symbol
    # mutation_id_symb = dict(zip(gene_id_patient, gene_symbol_profile))

    # print('mutation_profile', mutation_profile.dtype)
    return patient_id, mutation_profile, gene_id_patient, gene_symbol_profile


# @profile
def load_Faroe_Islands_data(data_folder):
    # TODO patients' ID, phenotypes in dictionary of dictionary or ...?
    print(" ==== Faroe Islands data ")
    df = pd.read_csv(data_folder + "Faroe_LGD_10percents_binary.txt", sep="\t")
    subjects = df.columns[1:]
    # http://www.genenames.org/cgi-bin/download?col=gd_app_sym&col=md_eg_id&status_opt=2&where=&order_by=gd_app_sym_sort&format=text&limit=&hgnc_dbtag=on&submit=submit
    hgnc = pd.read_csv(data_folder + "hgnc_2016-10-17.tsv", sep="\t")
    hgnc.rename(columns={'Approved Symbol': 'gene',
                         'Entrez Gene ID(supplied by NCBI)': 'EntrezID'},
                inplace=True)
    hgnc = hgnc.loc[~hgnc.loc[:, 'gene'].str.contains('withdrawn')]
    mutations = df.merge(hgnc, on='gene', how='outer')

    mutations = mutations.loc[np.isfinite(mutations.EntrezID)]
    # mutations.loc[:, subjects] = mutations.loc[:, subjects].fillna(0)
    mutations = mutations.dropna()
    mutation_profile = sp.csc_matrix((mutations.loc[:, subjects].values.T)
                                     .astype(np.float32))
    mutations.EntrezID = mutations.EntrezID.astype(int)
    gene_id_patient = mutations.EntrezID.tolist()

    return mutation_profile, gene_id_patient


def get_indiv_list(indiv_from_df):
    # create individual ID list
    alist = []
    for i in indiv_from_df:
        for j in i:
            alist.append(j)
    # set and keep unique ID (sort)
    alist = sorted(set(alist))
    return alist


def mutation_profile_coordinate(df, indiv_list):
    coord_gene = []
    coord_indiv = []

    for i in trange(df.shape[0], desc="mutation profile coordinates"):
        # for each row (each gene), we get list of individuals' ID
        individuals_per_gene = coordinate(df.iloc[i, 1], indiv_list)
        # for each element of coordinate listes x/y:
        for j in individuals_per_gene:
            # gene is saved as gene's INDEX (not EntrezGene) in dataframe
            coord_gene.append(i)
            # individual is saved as his/her INDEX (not ID) in indiv_list
            coord_indiv.append(j)

    return coord_indiv, coord_gene


def load_SSC_mutation_profile(ssc_type, data_folder):
    print(' ==== load_SSC_{}_mutation_profile'.format(ssc_type))
    mutation_profile_file = (data_folder + "SSC_{}_mutation_profile.mat"
                             .format(ssc_type))
    existance_file = os.path.exists(mutation_profile_file)

    if existance_file:
        print('***** SSC_{}_mutation_profile file already exists *****'
              .format(ssc_type))
        loadfile = loadmat(mutation_profile_file)
        mutation_profile = loadfile['mutation_profile']
        gene_id = loadfile['gene_id']
        indiv = loadfile['indiv']

    else:
        print('SSC_{}_mutation_profile file is calculating.....'
              .format(ssc_type))
        df = pd.read_csv(data_folder + 'SSC_EntrezGene_{}.csv'
                         .format(ssc_type), sep='\t')
        # individuals' ID list for each row is transformed in string of list
        # we thus reformate to list
        df.individuals = df.individuals.apply(eval)

        # create gene ID (EntrezGene ID) list
        gene_id = [int(i) for i in df.entrez_id.tolist()]

        # create individual ID list
        indiv = get_indiv_list(df.individuals)

        # calculate coordinates genes x individuals -> sparse matrix
        coord_indiv, coord_gene = mutation_profile_coordinate(df, indiv)
        # mutation weight = 1
        weight = np.ones(len(coord_gene))
        mutation_profile = sp.coo_matrix((weight, (coord_indiv, coord_gene)),
                                         shape=(len(indiv), len(gene_id)),
                                         dtype=np.float32)

        savemat(mutation_profile_file, {'mutation_profile': mutation_profile,
                                        'gene_id': gene_id,
                                        'indiv': indiv}, do_compression=True)

    return mutation_profile, gene_id, indiv


# @profile
def load_PPI(data_folder, ppi_data, load_gene_id_ppi=True):
    print(' ==== load_PPI ')
    filename = 'PPI_' + ppi_data + '.mat'
    loadfile = loadmat(data_folder + filename)
    network = loadfile['adj_mat'].astype(np.float32)
    if load_gene_id_ppi:
        print(' ==== load_gene_id_ppi ')
        gene_id_ppi = (loadfile['entrez_id'].flatten()).tolist()
        return gene_id_ppi, network
    else:
        return network


# @profile
def load_PPI_String(data_folder, ppi_data):
    # Entrez gene ID in PPI
    print(' ==== load_PPI_String and gene_id_ppi')
    entrez_to_idmat = loadmat(data_folder+'entrez_to_idmat.mat')
    gene_id_ppi = [x[0][0] for x in entrez_to_idmat['entrezid'][0]]
    # NOTE nan values in gene_id_ppi (choice of gene ID type)

    network = load_PPI(data_folder, ppi_data, load_gene_id_ppi=False)
    # print('---network ', type(network), network.dtype)
    return gene_id_ppi, network


# @profile
def coordinate(prot_list, all_list):
    coo_list = []
    for prot in prot_list:
        i = all_list.index(prot)
        coo_list.append(i)
    return coo_list


# def load_PPI_Y2H(data_folder, ppi_data):
#     print(' ==== load_PPI_Y2H ')
#     PPI_file = data_folder + 'PPI_Y2H.mat'
#     existance_file = os.path.exists(PPI_file)
#
#     if existance_file:
#         print('***** PPI_Y2H file already exists *****')
#         gene_id_ppi, network = load_PPI(
#             data_folder, ppi_data, load_gene_id_ppi=True)
#
#     else:
#         print('PPI_Y2H file is calculating.....')
#         data = genfromtxt(data_folder+'PPI_Y2H_raw.tsv',
#                           delimiter='\t', dtype=int)
#         # List of all proteins with Entrez gene ID
#         prot1 = data[1:, 0]
#         prot2 = data[1:, 1]
#         edge_list = np.vstack((prot1, prot2)).T
#         gene_id_ppi = (edge_list.flatten()).tolist()
#         gene_id_ppi = list(set(gene_id_ppi))
#
#         # From ID list to coordinate list
#         print(' ==== coordinates ')
#         coo1 = coordinate(prot1.tolist(), gene_id_ppi)
#         coo2 = coordinate(prot2.tolist(), gene_id_ppi)
#
#         # Adjacency matrix
#         print(' ==== Adjacency matrix ')
#         n = len(gene_id_ppi)
#         weight = np.ones(len(coo1))  # if interaction -> 1
#         network = sp.coo_matrix((weight, (coo1, coo2)), shape=(n, n))
#         network = network + network.T  # symmetric matrix
#         len(gene_id_ppi)
#         savemat(PPI_file, {'adj_mat': network, 'entrez_id': gene_id_ppi},
#                 do_compression=True)
#     return gene_id_ppi, network


def create_adjacency_matrix(prot1, prot2):
    # prot1 and prot2 are two columns including Entrez gene ID, from PPI data
    edge_list = np.vstack((prot1, prot2)).T
    gene_id_ppi = (edge_list.flatten()).tolist()
    gene_id_ppi = list(set(gene_id_ppi))

    # From ID list to coordinate list
    print(' ==== coordinates ')
    coo1 = coordinate(prot1.tolist(), gene_id_ppi)
    coo2 = coordinate(prot2.tolist(), gene_id_ppi)

    # Adjacency matrix
    print(' ==== Adjacency matrix ')
    n = len(gene_id_ppi)
    weight = np.ones(len(coo1))  # if interaction -> 1
    network = sp.coo_matrix((weight, (coo1, coo2)), shape=(n, n))
    network = network + network.T  # symmetric matrix
    savemat(PPI_file, {'adj_mat': network, 'entrez_id': gene_id_ppi},
            do_compression=True)
    return gene_id_ppi, network


def load_PPI_Y2H_or_APID(data_folder, ppi_data):
    print(' ==== load_PPI_{}'.format(ppi_data))
    PPI_file = data_folder + 'PPI_' + ppi_data + '.mat'
    existance_file = os.path.exists(PPI_file)

    if existance_file:
        print('***** PPI_{} file already exists *****'.format(ppi_data))
        gene_id_ppi, network = load_PPI(
            data_folder, ppi_data, load_gene_id_ppi=True)

    else:
        print('PPI_{} file is calculating.....'.format(ppi_data))
        if ppi_data == "Y2H":
            data = genfromtxt(data_folder+'PPI_Y2H_raw.tsv',
                              delimiter='\t', dtype=int)
            # List of all proteins with Entrez gene ID
            prot1 = data[1:, 0]
            prot2 = data[1:, 1]
        else:
            newest_file = max(glob.iglob(data_folder + 'PPI_APID*.csv'),
                              key=os.path.getctime)
            data = pd.read_csv(newest_file, sep="\t")
            prot1 = data.EntrezGene_1
            prot2 = data.EntrezGene_2

        gene_id_ppi, network = create_adjacency_matrix(prot1, prot2)

    return gene_id_ppi, network
