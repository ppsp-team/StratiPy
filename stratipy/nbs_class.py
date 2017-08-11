import scipy.sparse as sp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
# NOTE see also 'from sklearn.neighbors.kde import KernelDensity'
matplotlib.rcParams.update({'font.size': 16})


# TODO creat new class for sparse matrix type

# TODO description, explication of class
class Ppi:
    """Class defining a PPI (Protein-Protein Interaction) network

    Parameters
    ----------
    network : sparse matrix
        PPI data matrix, called also 'adjacency matrix'

    gid : list, optional
        List of entrez gene ID in PPI
        The ID are in the same order as in network

    Attributes
    ----------
    deg : array, shape(network.shape[0], )
        Array of degree number of each protein

    deg_list : list
        List of degree number of each protein
    """
    def __init__(self, network, *args):
        self.ppi = network  # TODO try with other sparse matrices
        self.gid = args
        self.deg = np.squeeze(np.array(self.ppi.sum(axis=0)))
        self.deg_list = self.deg.tolist()

    def __repr__(self):
        return ('PPI: adjacency sparse matrix {} ; list of genes if exist'
                .format(self.ppi.shape))

    # TODO plot title
    def plot_spy(self):
        """Plot sparsity pattern of PPI
        """
        plt.figure(figsize=(16, 16))
        plt.spy(self.ppi, markersize=1)
        plt.show()

    def plot_degree(self):
        """Plot degree (connectivity) number of each protein
        """
        plt.figure(1, figsize=(16, 10))
        plt.plot(self.deg)
        plt.ylabel("Degree (number of neighboors in the PPI)")
        plt.xlabel("Genes (keys)")
        plt.show()

    def plot_degree_distribution(self):
        """Plot degree (connectivity) distribution of proteins
        """
        count = {}.fromkeys(set(self.deg_list), 0)
        for i in self.deg_list:
            count[i] += 1
        x_degree = list(count.keys())
        y_degree = list(count.values())
        plt.figure(1, figsize=(16, 10))
        plt.scatter(x_degree, y_degree, c='red', marker='o', s=50,
                    edgecolors='none')
        plt.gca().set_xscale('log')
        plt.gca().set_yscale('log')
        plt.title('Degree distribution')
        plt.ylabel("Number of nodes (frequency)")
        plt.xlabel("Degree (number of neighboors in the PPI)")
        plt.ylim([0.9, 10e3])  # TODO ymax, xmax
        plt.xlim([1, 10e2])
        plt.show()

    def plot_imshow(self):
        """Plot PPI (numpy array)
        """
        plt.imshow(self.ppi.todense(), cmap='Greys', interpolation='none')
        plt.show()


class Patient:
    """Class defining mutation profile of patients

    Parameters
    ----------
    mutation_profile : sparse matrix
        Mutation profile matrix (rows: patients, columns: genes). For each
        element, 1 = mutated gene or 0 = not mutated gene.

    gid : list, optional
        List of entrez gene ID in mutation profile.
        The ID are in the same order as in mutation profile matrix.

    pid : list, optional
        List of patients' code (ID) in mutation profile.

    Attributes
    ----------
    mut_per_patient : array
        Array of mutation number of each patient
    """
    # def __init__(self, mutation_profile, gene_id_patient, patient_id):
    #     self.mut = mutation_profile  # TODO other sp matrices
    #     self.gid = gene_id_patient
    #     self.pid = patient_id

    def __init__(self, mutation_profile, *args):
        self.mut = mutation_profile  # TODO other sp matrices
        self.gid = args
        self.pid = args
        self.mut_per_patient = np.squeeze(np.array(self.mut.sum(axis=1)))

    def plot_proba_distribution(self, bandwidth=0.2):
        """Plot probability distribution of mutation profile

        Histogram and Kernel Density Estimation (KDE) plot of the mutation
        profile superposed. KDE estimates the probabilitty density function
        using Guassian kernel option here.

        bandwidth : float, default: 0.2
            Smoothing parameter exhibiting a strong influence with

            bandwidth > 0.
        """
        # plt.figure(1,figsize=(16,10))
        M = self.mut_per_patient
        x_gauss = np.linspace(1, M.max(), 1000)
        kernel = gaussian_kde(M, bandwidth)

        plt.plot(x_gauss, kernel(x_gauss), 'r', linewidth=2)
        plt.hist(M, normed=True, alpha=0.4, bins=50, linewidth=0.5)
        plt.xlim([0, M.max()])
        plt.xlabel('Mutations')
        plt.ylabel('Probability')
        plt.title('Probability distribution of Mutation profile')
        plt.show()

# TODO several plots with dictionary
    def plot_mut_profile(self):
        plt.figure(1, figsize=(16, 13))
        plt.plot(np.squeeze(np.array(self.mut[0,:])))
        plt.xlim([0, self.mut.shape[1]])
        plt.show()

    def __repr__(self):
        return ('Mutation profile: sparse matrix {} ; list of genes'
                .format(self.mut.shape))
