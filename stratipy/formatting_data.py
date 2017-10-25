import sys
import scipy.sparse as sp
import numpy as np
from stratipy.nbs_class import Ppi
import warnings

def check_sparsity(X):
    if not sp.issparse(X):
        X = sp.csc_matrix(X)
    return X


def check_shape_matching(X, L, array_name, list_name):
    if X.shape[0] != len(L):
        raise Exception("Numbers in {} shape ({}) and in {} ({}) don't match "
                        .format(array_name, X.shape(), list_name, len(L)))


#TODO check ID order in list and network


def check_PPI_data(network, gene_id_ppi):
    if network.shape[0] != network.shape[1]:
        raise Exception("Network data format is not a square matrix")


def check_patient_data(mutation_profile, gene_id_patient):
    if mutation_profile.shape[1] != len(gene_id_patient):
        raise Exception("Column number of mutation profile does not correspond to patient's gene number")
    #NOTE mutation_profile.shape[0] != len(patient_id) not verified because len(patient_id) could be bigger than mutation_profile.shape[0]. It depends on how many patients are annotated phenotypically.


# @profile
def classify_gene_index(network, mutation_profile, gene_id_ppi, gene_id_patient):
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
    # check step
    check_shape_matching(network, gene_id_ppi,
                         'PPI network matrix', 'List of Entrez Gene ID in PPI')
    check_PPI_data(network, gene_id_ppi)
    check_patient_data(mutation_profile, gene_id_patient)

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

    network = check_sparsity(network)
    mutation_profile = check_sparsity(mutation_profile)

    idx_ppi_only = [g for g in range(network.shape[1]) if not(g in idx_ppi)]

    # print('---network ', type(network), network.dtype)
    # print('mutation_profile', mutation_profile.dtype)

    return network, mutation_profile, idx_ppi, idx_mut, idx_ppi_only, idx_mut_only


# @profile
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
    if AA.shape[0] == 0:
        warnings.warn("There are no common genes between PPI network and patients' mutation profile")
    AB = network[idx_ppi][:, idx_ppi_only]
    AC = sp.csc_matrix((len(idx_ppi), len(idx_mut_only))).astype(np.float32)
    BA = network[idx_ppi_only][:, idx_ppi]
    BB = network[idx_ppi_only][:, idx_ppi_only]
    BC = sp.csc_matrix((len(idx_ppi_only), len(idx_mut_only))).astype(np.float32)

    # TODO condition: if mutOnly = 0
    CA = sp.csc_matrix((len(idx_mut_only), len(idx_ppi)), dtype=np.float32)
    CB = sp.csc_matrix((len(idx_mut_only), len(idx_ppi_only)), dtype=np.float32)
    CC = sp.csc_matrix((len(idx_mut_only), len(idx_mut_only)), dtype=np.float32)

    print(' ==== ABC  ')
    ppi_total = sp.bmat([[AA, AB, AC], [BA, BB, BC], [CA, CB, CC]],
                        format='csc')
    # NOTE ppi_total in COO matrix -> csc matrix
    # ppi_total = ppi_total.tocsc()
    print(' ==== mut_total  ')
    mut_total = sp.bmat([[mutation_profile[:, idx_mut],
                          sp.csc_matrix((mutation_profile.shape[0],
                                         len(idx_ppi_only)), dtype=np.float32),
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
