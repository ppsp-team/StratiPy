#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('stratipy')
import os
from scipy.io import loadmat, savemat
import scipy.sparse as sp
import numpy as np
import pandas as pd
from numpy import genfromtxt
from nbs_class import Ppi

# NOTE some variable names changed:
# dataFolder -> data_folder
# net -> network, ids -> gene_id_ppi,
# mutations -> mutation_profile, genes -> gene_id_patient
# geneSymbol_profile -> gene_symbol_profile
# subnet -> idx_ppi, good -> idx_mut, subnetNotmutated ->idx_ppi_only,
# bad -> idx_mut_only
# nnnet -> ppi_total, nnmut -> mut_total
# nnnetFiltered -> ppi_filt, nnmut-> mut_filt


@profile
def load_TCGA_UCEC_patient_data(data_folder):
    # TODO patients' ID, phenotypes in dictionary of dictionary or ...?
    print(" ==== TCGA patients' ID  ")
    phenotypes = loadmat(data_folder+'UCEC_clinical_phenotype.mat')
    patient_id = [c[0][0] for c in phenotypes['UCECppheno'][0][0][0]]

    # mutation profiles
    print(' ==== TCGA mutation profiles  ')
    somatic = loadmat(data_folder+'somatic_data_UCEC.mat')
    mutation_profile = sp.csc_matrix(somatic['gene_indiv_mat'].astype(np.float32))

    # Entrez gene ID and gene symbols in mutation profiles
    print(' ==== TCGA Entrez gene ID and gene symbols in mutation profiles  ')
    gene_id_patient = [x[0] for x in somatic['gene_id_all']]
    gene_symbol_profile = [x[0][0] for x in somatic['gene_id_symbol']]
    # dictionnary = key:entrez gene ID, value:symbol
    # mutation_id_symb = dict(zip(gene_id_patient, gene_symbol_profile))

    print('mutation_profile', mutation_profile.dtype)
    return patient_id, mutation_profile, gene_id_patient, gene_symbol_profile


@profile
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
    mutation_profile = sp.csc_matrix((mutations.loc[:, subjects].values.T).astype(np.float32))
    mutations.EntrezID = mutations.EntrezID.astype(int)
    gene_id_patient = mutations.EntrezID.tolist()

    return mutation_profile, gene_id_patient


@profile
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


@profile
def load_PPI_String(data_folder, ppi_data):
    # Entrez gene ID in PPI
    print(' ==== load_PPI_String and gene_id_ppi')
    entrez_to_idmat = loadmat(data_folder+'entrez_to_idmat.mat')
    gene_id_ppi = [x[0][0] for x in entrez_to_idmat['entrezid'][0]]
    # NOTE nan values in gene_id_ppi (choice of gene ID type)

    network = load_PPI(data_folder, ppi_data, load_gene_id_ppi=False)
    print('---network ', type(network), network.dtype)
    return gene_id_ppi, network


@profile
def coordinate(prot_list, all_list):
    coo_list = []
    for prot in prot_list:
        i = all_list.index(prot)
        coo_list.append(i)
    return coo_list


@profile
def load_PPI_Y2H(data_folder, ppi_data):
    print(' ==== load_PPI_Y2H ')
    PPI_file = data_folder + 'PPI_Y2H.mat'
    existance_file = os.path.exists(PPI_file)

    if existance_file:
        print('***** PPI_Y2H file already exists *****')
        gene_id_ppi, network = load_PPI(
            data_folder, ppi_data, load_gene_id_ppi=True)

    else:
        print('PPI_Y2H file is calculating.....')
        data = genfromtxt(data_folder+'PPI_Y2H_raw.tsv',
                          delimiter='\t', dtype=int)
        # List of all proteins with Entrez gene ID
        prot1 = data[1:, 0]
        prot2 = data[1:, 1]
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
        len(gene_id_ppi)
        savemat(PPI_file, {'adj_mat': network, 'entrez_id': gene_id_ppi},
                do_compression=True)
    return gene_id_ppi, network
