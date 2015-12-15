# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:58:37 2015

@author: aman
"""

import scipy.sparse as sp
import numpy
import pickle
import utils
import pandas

'''
To Add:
*Check if input network has directed edges or not
'''

alpha = 0.7
k = 11
thresh = 10**-6
topedges = 0.1 ##% of top edges to keep in network

##read network and keep top 10% edges by weight and put all weights as 1, to show whether there is a connection or not
net = numpy.loadtxt('../res/HumanNet.v1.join.txt',delimiter='\t',usecols = [0,1,23])
ranks = (-1*net[:,2]).argsort().argsort()
net = net[ranks < topedges*len(ranks),:]

##create list of genes
genes = list(set(net[:,0]).union(set(net[:,1])))
genes.sort()

##create adjacency matrix
indices = dict((j,i) for i,j in enumerate(genes))
net[:,0] = map(lambda x:indices[x],net[:,0])
net[:,1] = map(lambda x:indices[x],net[:,1])
mat = sp.csr_matrix((net[:,2],(net[:,0],net[:,1])),shape = (len(genes),len(genes)))
mat = mat + mat.T ### only if there are no reverse edges present, check this
mat = mat.tocsr() ## check if symmetric

##find influence
##borrowed and modified from  stratipy

raw = sp.dia_matrix((numpy.ones(len(genes)),[0]),shape = (len(genes),len(genes)))
influence = utils.diffuse(raw,mat,alpha,thresh)
influence = (influence < influence.T).multiply(influence) + (influence.T < influence).multiply(influence.T)
influence = influence.multiply(mat)  
##imo, this limits the influence to direct connections only i.e first degree and eliminates effect of a node on secondary connections
##This is effectively identifying best k direct connections of a node.
##Need to read this
##Vandin, F., Upfal, E., & Raphael, B. J. (2011). Algorithms for Detecting Significantly Mutated Pathways in Cancer. Journal of Computational Biology, 18(3), 507–522. http://doi.org/10.1089/cmb.2010.0265
##Vanunu O, Magger O, Ruppin E, Shlomi T, Sharan R (2010) Associating Genes and Protein Complexes with Disease via Network Propagation. PLoS Comput Biol 6(1): e1000641. doi:10.1371/journal.pcbi.1000641

##a bit slow solution, can be improved
vals = sp.find(influence)
vals = pandas.DataFrame({'i':vals[0],'j':vals[1],'v':vals[2]})
vals = vals.groupby('i').apply(lambda x:x.ix[x['v'].rank(method='max',ascending=False) <= k]).reset_index(drop=True)
vals['v'] = 1
knn = sp.csr_matrix((vals['v'],(vals['i'],vals['j'])),shape=mat.shape)
knn = mat.multiply(knn)
knn = knn + knn.T
knn.data[knn.data > 0] = 1

data = {'adj':mat,'knn':knn, 'genes':genes}
f = open('../res/network.data','w')
pickle.dump(data,f)
f.close()
