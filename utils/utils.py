# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 12:54:29 2015

@author: aman
"""
import numpy,pdb,copy
import scipy.sparse as sp

norm = lambda x: numpy.sqrt(x.multiply(x).sum())

def diffuse(raw,mat,alpha=0.7,thresh=10**-6):
    n = mat.shape[0]   
    sums = 1.0/mat.sum(axis=0)
    d = sp.dia_matrix((sums,[0]),shape = (n,n))
    mat_norm = mat.dot(d)
    x = raw
    diff = thresh + 1
    while diff>thresh:
        xn = alpha * (x.dot(mat_norm)) + (1-alpha) * raw
        diff = norm(x-xn)
        x = xn
        print diff      
    return x

def quantile_normalization(anarray):
    A = anarray.T
    AA = numpy.zeros_like(A)
    I = numpy.argsort(A,axis=0)
    AA[I,numpy.arange(A.shape[1])] = numpy.mean(A[I,numpy.arange(A.shape[1])],axis=1)[:,numpy.newaxis]
    return AA.T
        
def gnmf(X, W, nclust, gamma = 0, maxiter = 100, tolerance = .001):
# X is patients X genes
# X = UV
# obj = norm(X - U * V)^2 + trace(U.T * L * U)
# X = sp.lil_matrix(numpy.random.random((100,3))) * sp.lil_matrix(numpy.random.random((10,3))).T
# W = sp.csr_matrix(numpy.array(range(10000)).reshape(100,100))
    X = X.T
    D = sp.dia_matrix((W.sum(axis=0),[0]),shape = W.shape)
    eps =  10**-14
    ngenes, numpyatients = X.shape
    U = sp.lil_matrix(numpy.random.random((ngenes,nclust)))
    V = sp.lil_matrix(numpy.random.random((numpyatients,nclust)))
    
    for i in xrange(maxiter):
        obj = norm(X - U*V.T)
#        print "Iteration :",i,"gamma :",gamma,"Objective :",obj,"norm approx :",norm(U*V.T),"norm :",norm(X)
    
        un = X * V + gamma * W * U
        ud = U * V.T * V + gamma * D * U
        un.data[un.data < eps] = eps
        ud.data += eps
        U = U.multiply(un/ud)
        U.data = numpy.nan_to_num(U.data)
        
        vn = X.T * U
        vd = V * U.T * U
        vn.data[vn.data < eps] = eps
        vd.data += eps
        V = V.multiply(vn/vd)    
        V.data = numpy.nan_to_num(V.data)
        
        nf = U.sum(axis=0)
        U = U * sp.dia_matrix((1/nf,[0]),shape = (nclust,nclust))
        V = V * sp.dia_matrix((nf,[0]),shape = (nclust,nclust))
        
        if obj<tolerance:
            print 'converged'
            return (U,V)
            
    print 'Not converged'
    print "Iteration :",i,"gamma :",gamma,"Objective :",obj,"norm approx :",norm(U*V.T),"norm :",norm(X)
    return (U,V)
    
def consensus(func,X,fargs,**kwargs):
    ###X is patient X genes
    bootstrap = kwargs.pop('bootstrap',0.8)
    rep = kwargs.pop('rep',100)
    X = X.todense()
    nsamples, ngenes = X.shape
    similarity = numpy.zeros((nsamples,nsamples))
    indicator = numpy.zeros((nsamples,nsamples))
    for i in range(rep):
	print 'Repetition :',i
        thisgenes = numpy.random.random_integers(0,1,int(bootstrap*ngenes)).astype(bool)
        thissamples = numpy.random.random_integers(0,1,int(bootstrap*nsamples)).astype(bool)
        thisX = copy.deepcopy(X)
        thisX[:,~thisgenes] = 0
        thisX = thisX[thissamples,:]
        thisX = sp.csr_matrix(thisX,shape = thisX.shape)
        labels = func(thisX,*fargs,**kwargs)
        data = numpy.vstack([numpy.array(range(len(X)))[thissamples],labels]).T
        for j in set(labels):
            this = data[data[:,1]==j,0]
            similarity[numpy.ix_(this,this)] += 1
        indicator[numpy.ix_(thissamples,thissamples)] += 1
    con = 1.0*similarity/indicator
    return numpy.nan_to_num(con)
    

    
