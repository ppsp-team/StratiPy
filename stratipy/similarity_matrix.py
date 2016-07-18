from scipy.io import loadmat
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
import matplotlib


result_folder = '../data/result_Hofree/consensus_clustering/'

consensus_data = loadmat(result_folder + 'raw/nmf/consensus_alpha=0_tol=1e-05_ngh=11_minMut=10_maxMut=2000_comp=3_permut=1000_lambd=0_tolNMF=0.001.mat')
dNMF = consensus_data['distance_patients']
print('--- NMF ---')

consensus_data = loadmat(result_folder + 'diff/nmf/consensus_alpha=0.7_tol=1e-05_ngh=11_minMut=10_maxMut=2000_comp=3_permut=1000_lambd=0_tolNMF=0.001.mat')
dNMFDiff = consensus_data['distance_patients']
print('--- NMF diffused ---')

consensus_data = loadmat(result_folder + 'mean_qn/nmf/consensus_alpha=0.7_tol=1e-05_ngh=11_minMut=10_maxMut=2000_comp=3_permut=1000_lambd=0_tolNMF=0.001.mat')
dNMFQDiff = consensus_data['distance_patients']
print('--- NMF QN mean ---')

consensus_data = loadmat(result_folder + 'median_qn/nmf/consensus_alpha=0.7_tol=1e-05_ngh=11_minMut=10_maxMut=2000_comp=3_permut=1000_lambd=0_tolNMF=0.001.mat')
dNMFQDiffMed = consensus_data['distance_patients']
print('--- NMF QN median ---')

consensus_data = loadmat(result_folder + 'raw/gnmf/consensus_alpha=0_tol=1e-05_ngh=11_minMut=10_maxMut=2000_comp=3_permut=1000_lambd=1_tolNMF=0.001.mat')
dGNMF = consensus_data['distance_patients']
print('--- GNMF ---')

consensus_data = loadmat(result_folder + 'diff/gnmf/consensus_alpha=0.7_tol=1e-05_ngh=11_minMut=10_maxMut=2000_comp=3_permut=1000_lambd=1_tolNMF=0.001.mat')
dGNMFDiff = consensus_data['distance_patients']
print('--- GNMF diffused ---')

consensus_data = loadmat(result_folder + 'mean_qn/gnmf/consensus_alpha=0.7_tol=1e-05_ngh=11_minMut=10_maxMut=2000_comp=3_permut=1000_lambd=1_tolNMF=0.001.mat')
dGNMFQDiff = consensus_data['distance_patients']
print('--- GNMF QN mean ---')

consensus_data = loadmat(result_folder + 'median_qn/gnmf/consensus_alpha=0.7_tol=1e-05_ngh=11_minMut=10_maxMut=2000_comp=3_permut=1000_lambd=1_tolNMF=0.001.mat')
dGNMFQDiffMed = consensus_data['distance_patients']
print('--- GNMF QN median ---')


ZNMF = linkage(dNMF)
ZNMFDiff = linkage(dNMFDiff)
ZNMFQDiff = linkage(dNMFQDiff)
ZNMFQDiffMed = linkage(dNMFQDiffMed)
ZGNMF = linkage(dGNMF)
ZGNMFDiff = linkage(dGNMFDiff)
ZGNMFQDiff = linkage(dGNMFQDiff)
ZGNMFQDiffMed = linkage(dGNMFQDiffMed)

permutationsNum = 1000

print('...... Dendrogram plotting ......')

matplotlib.use('Agg')

from scipy.cluster.hierarchy import dendrogram
plt.suptitle("Dendrogram - cluster -", fontsize=20)
matplotlib.rcParams.update({'font.size': 10})
plt.figure(1,figsize=(16,16))

plt.subplot(811)
a=dendrogram(ZNMF,count_sort='ascending');
idxNMF=np.array(a['leaves'])
plt.title('NMF - %s permutations'%permutationsNum)
plt.subplot(812)
a=dendrogram(ZNMFDiff,count_sort='ascending');
idxNMFDiff=np.array(a['leaves'])
plt.title('NMF Diff - %s permutations'%permutationsNum)
plt.subplot(813)
a=dendrogram(ZNMFQDiff,count_sort='ascending');
idxNMFQDiff=np.array(a['leaves'])
plt.title('NMF Mean QN Diff - %s permutations'%permutationsNum)
plt.subplot(814)
a=dendrogram(ZNMFQDiffMed,count_sort='ascending');
idxNMFQDiffMed=np.array(a['leaves'])
plt.title('NMF Median QN Diff - %s permutations'%permutationsNum)
plt.subplot(815)
a=dendrogram(ZGNMF,count_sort='ascending');
idxGNMF=np.array(a['leaves'])
plt.title('GNMF - %s permutations'%permutationsNum)
plt.subplot(816)
a=dendrogram(ZGNMFDiff,count_sort='ascending');
idxGNMFDiff=np.array(a['leaves'])
plt.title('GNMF Diff - %s permutations'%permutationsNum)
plt.subplot(817)
a=dendrogram(ZGNMFQDiff,count_sort='ascending');
idxGNMFQDiff=np.array(a['leaves'])
plt.title('GNMF Mean QN Diff - %s permutations'%permutationsNum)
plt.subplot(818)
a=dendrogram(ZGNMFQDiffMed,count_sort='ascending');
idxGNMFQDiffMed=np.array(a['leaves'])
plt.title('GNMF Median QN Diff - %s permutations'%permutationsNum)
plt.tight_layout()
plt.savefig("dendrogram_cluster.png")




plt.figure(1,figsize=(10,15))
plt.suptitle("Similarity matrices - cluster -", fontsize=20)

plt.figure(1,figsize=(10,15))
plt.subplot(421)
plt.imshow(dNMF[idxNMF,:][:,idxNMF])
plt.title('NMF - %s permutations'%permutationsNum)
plt.subplot(422)
plt.imshow(dGNMF[idxGNMF,:][:,idxGNMF])
plt.title('GNMF - %s permutations'%permutationsNum)
plt.subplot(423)
plt.imshow(dNMFDiff[idxNMFDiff,:][:,idxNMFDiff])
plt.title('NMF Diff - %s permutations'%permutationsNum)
plt.subplot(424)
plt.imshow(dGNMFDiff[idxGNMFDiff,:][:,idxGNMFDiff])
plt.title('GNMF Diff - %s permutations'%permutationsNum)
plt.subplot(425)
plt.imshow(dNMFQDiff[idxNMFQDiff,:][:,idxNMFQDiff])
plt.title('NMF Mean QN Diff - %s permutations'%permutationsNum)
plt.subplot(426)
plt.imshow(dGNMFQDiff[idxGNMFQDiff,:][:,idxGNMFQDiff])
plt.title('GNMF Mean QN Diff - %s permutations'%permutationsNum)
plt.subplot(427)
plt.imshow(dNMFQDiffMed[idxNMFQDiffMed,:][:,idxNMFQDiffMed])
plt.title('NMF Median QN Diff - %s permutations'%permutationsNum)
plt.subplot(428)
plt.imshow(dGNMFQDiffMed[idxGNMFQDiffMed,:][:,idxGNMFQDiffMed])
plt.title('GNMF Median QN Diff - %s permutations'%permutationsNum)
plt.savefig("dendrogram_cluster.png")
