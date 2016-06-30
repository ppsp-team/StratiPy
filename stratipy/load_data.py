#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('stratipy')
from scipy.io import loadmat
import scipy.sparse as sp
import numpy as np
from numpy import genfromtxt
# from stratipy.nbs import Ppi
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


def load_Hofree_data(data_folder):
    print(' ==== load_Hofree_data ')
    print(' ==== PPI (adjacency matrix) ')
    # PPI (adjacency matrix)
    network = loadmat(data_folder+'adj_mat.mat')
    network = network['adj_mat']

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
    # NOTE mutation_profile in sparse

    # Entrez gene ID and gene symbols in mutation profiles
    print(' ==== Entrez gene ID and gene symbols in mutation profiles  ')
    gene_id_patient = [x[0] for x in somatic['gene_id_all']]
    gene_symbol_profile = [x[0][0] for x in somatic['gene_id_symbol']]
    # dictionnary = key:entrez gene ID, value:symbol
    # mutation_id_symb = dict(zip(gene_id_patient, gene_symbol_profile))

    return (network, gene_id_ppi, patient_id, mutation_profile,
            gene_id_patient, gene_symbol_profile)


def coordinate(prot_list, all_list):
    coo_list = []
    for prot in prot_list:
        i = all_list.index(prot)
        coo_list.append(i)
    return coo_list


def load_TR_data(data_folder, data_mutation_profile):
    print(" ==== patients' ID  ")
    phenotypes = loadmat(data_mutation_profile+'UCEC_clinical_phenotype.mat')
    patient_id = [c[0][0] for c in phenotypes['UCECppheno'][0][0][0]]

    print(' ==== mutation profiles  ')
    somatic = loadmat(data_mutation_profile+'somatic_data_UCEC.mat')
    mutation_profile = sp.csc_matrix(somatic['gene_indiv_mat'])

    print(' ==== Entrez gene ID and gene symbols in mutation profiles  ')
    gene_id_patient = [x[0] for x in somatic['gene_id_all']]
    gene_symbol_profile = [x[0][0] for x in somatic['gene_id_symbol']]

    print(' ==== load_TR_data ')
    data = genfromtxt(data_folder+'prepare_PPIs_noBS.tsv',
                      delimiter='\t', dtype=int)
    # List of all proteins with Entrez gene ID
    prot1 = data[1:, 0]
    prot2 = data[1:, 1]
    edge_list = np.vstack((prot1, prot2)).T
    gene_id_ppi = (edge_list.flatten()).tolist()
    gene_id_ppi = list(set(gene_id_ppi))

    # From ID list to coordinate list
    print(' ==== coorinates ')
    coo1 = coordinate(prot1.tolist(),gene_id_ppi)
    coo2 = coordinate(prot2.tolist(),gene_id_ppi)

    # Adjacency matrix
    print(' ==== Adjacency matrix ')
    n = len(gene_id_ppi)
    weigh = np.ones(len(coo1)) # if interaction -> 1
    network = sp.coo_matrix((weigh, (coo1, coo2)), shape=(n, n))
    network = network + network.T # symmetric matrix

    return (network, gene_id_ppi, patient_id, mutation_profile,
            gene_id_patient, gene_symbol_profile)


def classify_gene_index(network, gene_id_ppi, gene_id_patient):
    """Gene index classification

    Compares genes' ID between PPI and mutaton profile in order to classify
    them by indexes.

    Parameters
    ----------
    network : sparse matrix
        PPI data matrix, called also 'adjacency matrix'.

    gene_id_ppi : list
        List of Entrez Gene ID in PPI. The ID must be in the same order as in
        network.

    gene_id_patient : list
        List of Entrez Gene ID in patients' mutation profiles. The ID must be
        in the same order as in mutation profiles.

    Returns
    -------
    idx_ppi : list
        List of common genes' indexes in PPI.

    idx_mut : list
        List of common genes' indexes in patients' mutation profiles.

    idx_ppi_only : list
        List of genes' indexes only in PPI.

    idx_mut_only : list
        List of genes' indexes only in patients' mutation profiles.
    """
    print(' ==== classify_gene_index  ')
    idx_ppi = []
    idx_mut = []
    idx_mut_only = []
    for j, g in enumerate(gene_id_patient):
        try:
            i = gene_id_ppi.index(g)
            idx_ppi.append(i)
            idx_mut.append(j)
        except:
            i = np.nan
            idx_mut_only.append(j)
    print(' ==== before idx  ')
    idx_ppi_only = [g for g in range(network.shape[1]) if not(g in idx_ppi)]
    print(' ==== after idx  ')
    return idx_ppi, idx_mut, idx_ppi_only, idx_mut_only


def all_genes_in_submatrices(network, idx_ppi, idx_mut, idx_ppi_only,
                             idx_mut_only, mutation_profile):
    """Processing of sub-matrices for each case of genes

    Extract the sub-matrices of references genes and Zero-padding of the
    adjacency matrix.
    ┌ - - - - - - - ┐
    |       |  |    |
    |   AA  |AB| AC |
    |       |  |    |
    |- - - - - - - -|
    |   BA  |BB| BC |
    |- - - - - - - -|
    |   CA  |CB| CC |
    └ - - - - - - - ┘
    AA, BB and CC are all adjacency matrces (0/1 matrices with 0 on diagonal) :
        AA : common genes between PPI and patients' mutation profiles
        BB : genes founded only in PPI
        CC : genes founded only in patients' mutation profiles = zero matrix

    Parameters
    ----------
    network : sparse matrix, shape (len(gene_id_ppi),len(gene_id_ppi))
        PPI data matrix, called also 'adjacency matrix'.

    idx_ppi : list
        List of common genes' indexes in PPI.

    idx_ppi_only : list
        List of genes' indexes only in PPI.


    idx_mut_only : list
        List of genes' indexes only in patients' mutation profiles.

    Returns
    -------
    ppi_total : sparse matrix
        Built from all sparse sub-matrices (AA, ... , CC).

    mut_total : sparse matrix
        Patients' mutation profiles of all genes (rows: patients,
        columns: genes of AA, BB and CC).

    ppi_filt : sparse matrix
        Filtration from ppi_total : only genes in PPI are considered.
        ┌ - - - - -┐
        |       |  |
        |   AA  |AB|
        |       |  |
        |- - - - - |
        |   BA  |BB|
        └ - - - - -┘

    mut_filt : sparse matrix
        Filtration from mut_total : only genes in PPI are considered.
    """
    print(' ==== all_genes_in_submatrices ')
    AA = network[idx_ppi][:, idx_ppi]
    AB = network[idx_ppi][:, idx_ppi_only]
    AC = sp.csc_matrix((len(idx_ppi), len(idx_mut_only)))
    BA = network[idx_ppi_only][:, idx_ppi]
    BB = network[idx_ppi_only][:, idx_ppi_only]
    BC = sp.csc_matrix((len(idx_ppi_only), len(idx_mut_only)))
    # TODO condition: if mutOnly = 0
    CA = sp.csc_matrix((len(idx_mut_only), len(idx_ppi)))
    CB = sp.csc_matrix((len(idx_mut_only), len(idx_ppi_only)))
    CC = sp.csc_matrix((len(idx_mut_only), len(idx_mut_only)))
    print(' ==== ABC  ')
    ppi_total = sp.bmat([[AA, AB, AC], [BA, BB, BC], [CA, CB, CC]],
                        format='csc')
    # NOTE ppi_total in COO matrix -> csc matrix
    # ppi_total = ppi_total.tocsc()
    print(' ==== mut_total  ')
    mut_total = sp.bmat([[mutation_profile[:, idx_mut],
                          sp.csc_matrix((mutation_profile.shape[0],
                                         len(idx_ppi_only))),
                          mutation_profile[:, idx_mut_only]]])
    # filter only genes in PPI
    print(' ==== filter only genes in PPI  ')
    degree = Ppi(ppi_total).deg
    ppi_filt = ppi_total[degree > 0, :][:, degree > 0]
    mut_filt = mut_total[:, degree > 0]
    print(' ==== all_genes_in_submatrices finish  ')
    return ppi_total, mut_total, ppi_filt, mut_filt

    # TODO for numpy docring: Raises (errors), Note, Examples
    # errors : sparse (ppi, patients), 0 on diagonal (ppi)
