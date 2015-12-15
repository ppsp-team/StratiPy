# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:57:50 2015

@author: aman
"""

import scipy.sparse as sp
import pickle
import numpy
import random
import utils
from scipy.cluster.hierarchy import linkage, fcluster,dendrogram
import matplotlib.pyplot as plt


alpha = 0.7
mutations_min = 6
diff_thresh = 10**-6
nclust = 4
maxiter = 100
tolerance = 0.01
gamma = 0

f = open('../res/network.data','r')
data = pickle.load(f)
genes = data['genes']
f.close()

###patient x genes
###for unit testing
#mutations = numpy.random.random_integers(0,1,(100,10))
#thisgenes = random.sample(genes,10)
###

f = open('../res/OV.data','r')
mdata = pickle.load(f)
f.close()

print mdata.keys()
mutations = mdata['WUSM__IlluminaGA_DNASeq']
thisgenes = list(mutations.columns)
samples = numpy.array(list(mutations.index))
mutations = numpy.array(mutations)
mutations[mutations>1] = 1

toadd = [i for i in genes if i not in thisgenes]
tokeep_thisindex = [i for i,j in enumerate(thisgenes) if j in genes]
tokeep_geneindex = [i for i,j in enumerate(genes) if j in thisgenes]

minflag = (mutations.sum(axis=1) > mutations_min)
mutations = mutations[minflag,:]
samples = samples[minflag]
mutations_temp = numpy.zeros((len(mutations),len(genes)))
mutations_temp[:,tokeep_geneindex] = mutations[:,tokeep_thisindex]
mutations = sp.csr_matrix(mutations_temp)

mutation_smooth = utils.diffuse(mutations,data['adj'],alpha,diff_thresh)
mutation_smooth_norm = sp.csr_matrix(utils.quantile_normalization(numpy.array(mutation_smooth.todense())),shape=mutation_smooth.shape)

#U,V = utils.gnmf(mutation_smooth,data['knn'],nclust, gamma, maxiter, tolerance)
#labels = numpy.array(V.todense().argmax(axis=1))[:,0]

def gnmfsingle(X, W, nclust, gamma, maxiter, tolerance):
    U,V = utils.gnmf(X, W ,nclust, gamma, maxiter, tolerance)
    return numpy.array(V.todense().argmax(axis=1))[:,0]

cons = utils.consensus(gnmfsingle,mutation_smooth_norm, [data['knn'],nclust, gamma, maxiter, tolerance], bootstrap = 0.8,rep = 100)

######take from stratipy modules
zmatrix = linkage(cons,method='average')
clusters = fcluster(zmatrix,1)
dend = dendrogram(zmatrix,count_sort='ascending')

idx=numpy.array(dend['leaves'])
plt.imshow(cons[idx,:][:,idx])

plt.plot(range(1, len(zmatrix)+1), zmatrix[::-1, 2])
knee = numpy.diff(zmatrix[::-1, 2], 2)
plt.plot(range(2, len(zmatrix)), knee)

num_clust1 = knee.argmax() + 2
knee[knee.argmax()] = 0
num_clust2 = knee.argmax() + 2

plt.text(num_clust1, zmatrix[::-1, 2][num_clust1-1], 'possible\n<- knee point')