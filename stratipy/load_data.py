#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('stratipy')
from scipy.io import loadmat
import scipy.sparse as sp
import numpy as np
from numpy import genfromtxt
from nbs import Ppi

# NOTE some variable names changed:
# dataFolder -> data_folder
# net -> network, ids -> gene_id_ppi,
# mutations -> mutation_profile, genes -> gene_id_patient
# geneSymbol_profile -> gene_symbol_profile
# subnet -> idx_ppi, good -> idx_mut, subnetNotmutated ->idx_ppi_only,
# bad -> idx_mut_only
# nnnet -> ppi_total, nnmut -> mut_total
# nnnetFiltered -> ppi_filt, nnmut-> mut_filt


def load_patient_data(data_folder):
    # Entrez gene ID in PPI
    print(' ==== Entrez gene ID in PPI ')
    entrez_to_idmat = loadmat(data_folder+'entrez_to_idmat.mat')
    gene_id_ppi = [x[0][0] for x in entrez_to_idmat['entrezid'][0]]
    # NOTE nan values in gene_id_ppi (choice of gene ID type)

    # TODO patients' ID, phenotypes in dictionary of dictionary or ...?
    print(" ==== patients' ID  ")
    phenotypes = loadmat(data_folder+'UCEC_clinical_phenotype.mat')
    patient_id = [c[0][0] for c in phenotypes['UCECppheno'][0][0][0]]

    # mutation profiles
    print(' ==== mutation profiles  ')
    somatic = loadmat(data_folder+'somatic_data_UCEC.mat')
    mutation_profile = sp.csc_matrix(somatic['gene_indiv_mat'])

    # Entrez gene ID and gene symbols in mutation profiles
    print(' ==== Entrez gene ID and gene symbols in mutation profiles  ')
    gene_id_patient = [x[0] for x in somatic['gene_id_all']]
    gene_symbol_profile = [x[0][0] for x in somatic['gene_id_symbol']]
    # dictionnary = key:entrez gene ID, value:symbol
    # mutation_id_symb = dict(zip(gene_id_patient, gene_symbol_profile))

    return (gene_id_ppi, patient_id, mutation_profile,
            gene_id_patient, gene_symbol_profile)


def load_PPI_Hofree(data_folder):
    print(' ==== load_Hofree_data ')
    print(' ==== Adjacency matrix ')
    network = loadmat(data_folder+'adj_mat.mat')
    network = network['adj_mat']

    return network


def coordinate(prot_list, all_list):
    coo_list = []
    for prot in prot_list:
        i = all_list.index(prot)
        coo_list.append(i)
    return coo_list


def load_PPI_TR(data_folder):
    print(' ==== load_TR_data ')
    data = genfromtxt(data_folder+'PPI_Systematic.tsv',
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

    return network
