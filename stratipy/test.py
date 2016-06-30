import importlib  # NOTE for python >= Python3.4
import load_data
import nbs
from nbs import Ppi, Patient
import filtering_diffusion
import clustering
import scipy.sparse as sp
import numpy as np
import time
import datetime

start = time.time()
end = time.time()
print("----- %.2f sec -----" % (end-start))
# ----------------------- load_data.py -----------------------
# load Hofree's dataset
(network, gene_id_ppi, patient_id, mutation_profile, gene_id_patient,
 gene_symbol_profile) = load_data.load_Hofree_data('data/')

# call PPI network class
# ppi_raw = Ppi(network, gene_id_ppi)
# call patients' mutation profiles class
# mut_raw = Patient(mutation_profile, gene_id_patient, patient_id)

# extract indexes of the genes in the adjacency matrix
idx_ppi, idx_mut, idx_ppi_only, idx_mut_only = load_data.classify_gene_index(
    network, gene_id_ppi, gene_id_patient)

# Extract the submatrices of references genes &
# zero-padding of the adjacency matrix
ppi_total, mut_total, ppi_filt, mut_filt = load_data.all_genes_in_submatrices(
    network, idx_ppi, idx_mut, idx_ppi_only, idx_mut_only, mutation_profile)

# ----------------------- filtering_diffusion.py -----------------------
# influence calcul on PPI
ppi_influence = (
    filtering_diffusion.calcul_ppi_influence(
        sp.eye(ppi_filt.shape[0]), ppi_filt,
        'data/', compute=False, overwrite=False, alpha=0.7, tol=10e-6))

# filtering
ppi_final, mut_final = filtering_diffusion.filter_ppi_patients(
    ppi_total, mut_total, ppi_filt, ppi_influence, ngh_max=11,
    keep_singletons=False, min_mutation=10, max_mutation=2000)

# diffusion of mutation profiles according to the PPI + QN
# 8s
mut_type, mut_propag = filtering_diffusion.propagation_profile(
    mut_final, ppi_filt, alpha=0.7, tol=10e-6, qn='median')

# ----------------------- decomposition.py -----------------------

# fit_transform and 'cd' version, Coordinate Descent solver (recommended)
# (gene_comp, patient_strat,
#  gene_comp_diff, patient_strat_diff,
#  gene_comp_mean_qn, patient_strat_mean_qn,
#  gene_comp_median_qn, patient_strat_median_qn) = decomposition.nmf_new(
#      mut_final, mut_diff, mut_mean_qn, mut_median_qn,
#      n_components=3, init='nndsvdar', random_state=0)

# ----------------------- clustering.py -----------------------
importlib.reload(clustering)
import clustering

genes_clustering, patients_clustering = (clustering.bootstrap(
    'data/', mut_type, mut_propag, ppi_final,
    alpha=0.7, tol=10e-6, ngh_max=11, min_mutation=10, max_mutation=2000,
    n_components=3, n_permutations=2,
    run_bootstrap=True, lambd=0, tol_nmf=1e-3))

distance_pat = clustering.concensus_clustering(patients_clustering)
distance_genes = clustering.concensus_clustering(genes_clustering)

from scipy.io import loadmat
bootstrap_data = loadmat('data/bootstrap-1000perm.mat')

genesClusteringNMF=bootstrap_data['genesClusteringNMF']
patientsClusteringNMF=bootstrap_data['patientsClusteringNMF']
    mut_type='raw',
    alpha=0,
    lambd=0,
genesClusteringNMFDiff=bootstrap_data['genesClusteringNMFDiff']
patientsClusteringNMFDiff=bootstrap_data['patientsClusteringNMFDiff']
    mut_type='diff',
    alpha=0.7,
    lambd=0,

genesClusteringNMFQDiff=bootstrap_data['genesClusteringNMFQDiff']
patientsClusteringNMFQDiff=bootstrap_data['patientsClusteringNMFQDiff']
    mut_type='mean_qn',
    alpha=0.7,
    lambd=0,
genesClusteringNMFQDiffMed=bootstrap_data['genesClusteringNMFQDiffMed']
patientsClusteringNMFQDiffMed=bootstrap_data['patientsClusteringNMFQDiffMed']
    mut_type='median_qn',
    alpha=0.7,
    lambd=0,
genesClusteringGNMF=bootstrap_data['genesClusteringGNMF']
patientsClusteringGNMF=bootstrap_data['patientsClusteringGNMF']
    mut_type='raw',
    alpha=0,
    lambd=0.7,
genesClusteringGNMFDiff=bootstrap_data['genesClusteringGNMFDiff']
patientsClusteringGNMFDiff=bootstrap_data['patientsClusteringGNMFDiff']
    mut_type='diff',
    alpha=0.7,
    lambd=0.7,
genesClusteringGNMFQDiff=bootstrap_data['genesClusteringGNMFQDiff']
patientsClusteringGNMFQDiff=bootstrap_data['patientsClusteringGNMFQDiff']
    mut_type='mean_qn',
    alpha=0.7,
    lambd=0.7,
genesClusteringGNMFQDiffMed=bootstrap_data['genesClusteringGNMFQDiffMed']
patientsClusteringGNMFQDiffMed=bootstrap_data['patientsClusteringGNMFQDiffMed']
    mut_type='median_qn',
    alpha=0.7,
    lambd=0.7,

dNMF = clustering.concensus_clustering_simple(patientsClusteringNMF)
dNMFDiff = clustering.concensus_clustering(dataNMFDiff)
dNMFQDiff = clustering.concensus_clustering(dataNMFQDiff)
dNMFQDiffMed = clustering.concensus_clustering(dataNMFQDiffMed)
dGNMF = clustering.concensus_clustering(dataGNMF)
dGNMFDiff = clustering.concensus_clustering(dataGNMFDiff)
dGNMFQDiff = clustering.concensus_clustering(dataGNMFQDiff)
dGNMFQDiffMed = clustering.concensus_clustering(dataGNMFQDiffMed)

importlib.reload(clustering)
import clustering

start = time.time()
distance_genes, distance_patients = clustering.consensus_clustering(
    'data/',
    genesClusteringNMFQDiff,
    patientsClusteringNMFQDiff,
    mut_type='mean_qn',
    alpha=0.7,
    lambd=0,
    tol=10e-6, ngh_max=11, min_mutation=10, max_mutation=2000,
    n_components=3, n_permutations=1000, tol_nmf=1e-3)
end = time.time()
print("---------- Clustering TOTAL = ", datetime.timedelta(seconds=end-start), "----------")

from scipy.io import loadmat
lire1 = loadmat('data/consensus_clustering/diff/nmf/bootstrap_alpha=0.7_tol=1e-05_ngh=11_minMut=10_maxMut=2000_comp=3_permut=1000_lambd=0_tolNMF=0.001.mat')
dNMFdiff_pat = lire1['distance_patients']
dNMFdiff_genes = lire1['distance_genes']
p1 = Ppi(dNMFdiff_pat

import matplotlib.pyplot as plt
plt.imshow(dNMFdiff_genes, cmap='Greys')
plt.show()









# ------------------------ dendrogram test
from scipy.io import loadmat
lire = loadmat('data/consensus_clustering/diff/gnmf/consensus_alpha=0.7_tol=1e-05_ngh=11_minMut=10_maxMut=2000_comp=3_permut=1000_lambd=1_tolNMF=0.001.mat')
dp_diff_gnmf = lire['distance_patients']
dg_diff_gnmf = lire['distance_genes']

lire = loadmat('data/consensus_clustering/diff/nmf/consensus_alpha=0.7_tol=1e-05_ngh=11_minMut=10_maxMut=2000_comp=3_permut=1000_lambd=0_tolNMF=0.001.mat')
dp_diff_nmf = lire['distance_patients']
dg_diff_nmf = lire['distance_genes']

lire = loadmat('data/consensus_clustering/mean_qn/gnmf/consensus_alpha=0.7_tol=1e-05_ngh=11_minMut=10_maxMut=2000_comp=3_permut=1000_lambd=1_tolNMF=0.001.mat')
dp_mean_gnmf = lire['distance_patients']
dg_mean_gnmf = lire['distance_genes']

lire = loadmat('data/consensus_clustering/mean_qn/nmf/consensus_alpha=0.7_tol=1e-05_ngh=11_minMut=10_maxMut=2000_comp=3_permut=1000_lambd=0_tolNMF=0.001.mat')
dp_mean_nmf = lire['distance_patients']
dg_mean_nmf = lire['distance_genes']


lire = loadmat('data/consensus_clustering/median_qn/gnmf/consensus_alpha=0.7_tol=1e-05_ngh=11_minMut=10_maxMut=2000_comp=3_permut=1000_lambd=1_tolNMF=0.001.mat')
dp_median_gnmf = lire['distance_patients']
dg_median_gnmf = lire['distance_genes']

lire = loadmat('data/consensus_clustering/median_qn/nmf/consensus_alpha=0.7_tol=1e-05_ngh=11_minMut=10_maxMut=2000_comp=3_permut=1000_lambd=0_tolNMF=0.001.mat')
dp_median_nmf = lire['distance_patients']
dg_median_nmf = lire['distance_genes']



import matplotlib.pyplot as plt
# plt.imshow(distance_patients, cmap='Greys')
# plt.show()

from scipy.cluster.hierarchy import linkage, fcluster
# link_patients = linkage(distance_patients)  # 5-6ms
# link_genes = linkage(distance_genes)  # long : 12min 52s

# clust_patients = fcluster(link_patients, 1)  # 1.09ms
# clust_genes = fcluster(link_genes, 1)  # 52 ms

lp_diff_gnmf = linkage(dp_diff_gnmf)
lp_diff_nmf = linkage(dp_diff_nmf)
lp_mean_gnmf = linkage(dp_mean_gnmf)
lp_mean_nmf = linkage(dp_mean_nmf)
lp_median_gnmf = linkage(dp_median_gnmf)
lp_median_nmf = linkage(dp_median_nmf)

# lg_diff_gnmf = linkage(dg_diff_gnmf)
# lg_diff_nmf = linkage(dg_diff_nmf)
# lg_mean_gnmf = linkage(dg_mean_gnmf)
# lg_mean_nmf = linkage(dg_mean_nmf)
# lg_median_gnmf = linkage(dg_median_gnmf)
# lg_median_nmf = linkage(dg_median_nmf)


import numpy as np
from scipy.cluster.hierarchy import dendrogram

# a = dendrogram(link_patients, count_sort='ascending')
# idx_patients=np.array(a['leaves'])
# plt.title('GNMF Median QN - consensus clustering PATIENTS - 1000 permutations')

plt.figure(1,figsize=(20,7))
plt.subplot(321)
a=dendrogram(lp_diff_nmf,count_sort='ascending');
idxNMFDiff=np.array(a['leaves'])
plt.title('NMF diff - consensus clustering PATIENTS - 1000 permutations')

plt.subplot(322)
a=dendrogram(lp_diff_gnmf,count_sort='ascending');
idxGNMFDiff=np.array(a['leaves'])
plt.title('GNMF diff - consensus clustering PATIENTS - 1000 permutations')

plt.subplot(323)
a=dendrogram(lp_mean_nmf,count_sort='ascending');
idxNMFQDiff=np.array(a['leaves'])
plt.title('NMF mean QN - consensus clustering PATIENTS - 1000 permutations')

plt.subplot(324)
a=dendrogram(lp_mean_gnmf,count_sort='ascending');
idxGNMFQDiff=np.array(a['leaves'])
plt.title('GNMF mean QN - consensus clustering PATIENTS - 1000 permutations')

plt.subplot(325)
a=dendrogram(lp_median_nmf,count_sort='ascending');
idxNMFQDiffMed=np.array(a['leaves'])
plt.title('NMF median QN - consensus clustering PATIENTS - 1000 permutations')

plt.subplot(326)
a=dendrogram(lp_median_gnmf,count_sort='ascending');
idxGNMFQDiffMed=np.array(a['leaves'])
plt.title('GNMF median QN - consensus clustering PATIENTS - 1000 permutations')
plt.tight_layout()
plt.show()



permutationsNum=1000
plt.figure(1,figsize=(10,15))
plt.subplot(321)
plt.imshow(dp_diff_nmf[idxNMFDiff,:][:,idxNMFDiff], cmap='Greys')
plt.title('NMF Diff - %s permutations'%permutationsNum)
plt.subplot(322)
plt.imshow(dp_diff_gnmf[idxGNMFDiff,:][:,idxGNMFDiff], cmap='Greys')
plt.title('GNMF Diff - %s permutations'%permutationsNum)
plt.subplot(323)
plt.imshow(dp_mean_nmf[idxNMFQDiff,:][:,idxNMFQDiff], cmap='Greys')
plt.title('NMF Mean QN Diff - %s permutations'%permutationsNum)
plt.subplot(324)
plt.imshow(dp_mean_gnmf[idxGNMFQDiff,:][:,idxGNMFQDiff], cmap='Greys')
plt.title('GNMF Mean QN Diff - %s permutations'%permutationsNum)
plt.subplot(325)
plt.imshow(dp_median_nmf[idxNMFQDiffMed,:][:,idxNMFQDiffMed], cmap='Greys')
plt.title('NMF Median QN Diff - %s permutations'%permutationsNum)
plt.subplot(326)
plt.imshow(dp_median_gnmf[idxGNMFQDiffMed,:][:,idxGNMFQDiffMed], cmap='Greys')
plt.title('GNMF Median QN Diff - %s permutations'%permutationsNum)
plt.show()















plt.imshow(distance_patients[idx_patients, :][:, idx_patients], cmap='Greys')
plt.show()
























# import sys
# sys.setrecursionlimit(8000)  # ATTENTION 1000 par defaut
# b = dendrogram(link_genes, count_sort='ascending')
# plt.title('GNMF Median QN - consensus clustering GENES - 1000 permutations')
# idx_genes=np.array(b['leaves'])
#
plt.imshow(distance_genes[idx_genes, :][:, idx_genes], cmap='Greys')
plt.show()

# --------------------------------------------------------------------
# --------------------------------------------------------------------
# ------------------------------ TO RUN ------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------
def all_functions(data_folder, compute=False, overwrite=Fals, alpha=0.7,
                  tol=10e-6, ngh_max=11, keep_singletons=False, min_mutation=10,
                  max_mutation=2000, qn=None,
                  n_components=3, n_permutations=2,
                  run_bootstrap=False, lambd=0, tol_nmf=1e-3):
    # ------------ load_data.py ------------
    (network, gene_id_ppi, patient_id, mutation_profile, gene_id_patient,
     gene_symbol_profile) = load_data.load_Hofree_data(data_folder)

    (idx_ppi, idx_mut, idx_ppi_only, idx_mut_only) = (
        load_data.classify_gene_index(
            network, gene_id_ppi, gene_id_patient))

    (ppi_total, mut_total, ppi_filt, mut_filt) = (
        load_data.all_genes_in_submatrices(
            network, idx_ppi, idx_mut, idx_ppi_only, idx_mut_only,
            mutation_profile))

    # ------------ filtering_diffusion.py ------------
    ppi_influence = (
        filtering_diffusion.calcul_ppi_influence(
            sp.eye(ppi_filt.shape[0]), ppi_filt,
            data_folder, compute, overwrite, alpha, tol))

    ppi_final, mut_final = filtering_diffusion.filter_ppi_patients(
        ppi_total, mut_total, ppi_filt, ppi_influence, ngh_max,
        keep_singletons, min_mutation, max_mutation)

    mut_type, mut_propag = filtering_diffusion.propagation_profile(
        mut_final, ppi_filt, alpha, tol, qn)

    # ------------ clustering.py ------------
    genes_clustering, patients_clustering = (clustering.bootstrap(
        data_folder, mut_type, mut_propag, ppi_final,
        alpha, tol, ngh_max, min_mutation, max_mutation,
        n_components, n_permutations,
        run_bootstrap, lambd, tol_nmf))

    distance_genes, distance_patients = clustering.consensus_clustering(
        data_folder, genes_clustering, patients_clustering, mut_type,
        alpha, tol, ngh_max, min_mutation, max_mutation,
        n_components, n_permutations, lambd, tol_nmf)

    return distance_genes, distance_patients


# ------------------------------ GO GO GO GO ------------------------------

start = time.time()

distance_pat, distance_genes = all_functions(data_folder='data/',
                                             compute=False,
                                             overwrite=False,
                                             alpha=0.7,
                                             tol=10e-6,
                                             ngh_max=11,
                                             keep_singletons=False,
                                             min_mutation=10,
                                             max_mutation=2000,
                                             qn='median',
                                             n_components=3,
                                             n_permutations=1000,
                                             run_bootstrap=True,
                                             lambd=0,
                                             tol_nmf=1e-3)


end = time.time()
print("---------- ALL = ", datetime.timedelta(seconds=end-start), "----------")











importlib.reload(load_data)
import load_data

importlib.reload(nbs)
import nbs
from nbs import Ppi, Patient

importlib.reload(filtering_diffusion)
import filtering_diffusion

importlib.reload(clustering)
import clustering


start = time.time()
#
end = time.time()
print("----- %s sec -----" % (end-start))
