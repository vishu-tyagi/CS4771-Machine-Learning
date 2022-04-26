import random
import numpy as np

import scipy
from scipy.spatial.distance import cdist 

from sklearn.neighbors import kneighbors_graph


def kmeans(X, k, iters=10, return_means=False, initialize=False, initial_means=None):
    '''Groups the points in X into k clusters using the K-means algorithm.

    Parameters
    ----------
    X : (m x n) data matrix
    k: number of clusters
    iters: number of iterations to run k-means loop

    Returns
    -------
    y: (m x 1) cluster assignment for each point in X
    '''
    # number of samples
    m = X.shape[0]
    
    # dimension
    n = X.shape[1]
    
    # initialize cluster means
    if initialize:
        if initial_means is None:
            raise ValueError('Enter initial means')

        means = np.array(initial_means)

    if not initialize:
        if k < m+1:
            # without replacement
            idx = random.sample(population=range(m), k=k)
        else:
            # with replacement
            idx = random.choice(population=range(m), k=k)
    
        means = X[idx]
    
    # list to indicate cluster assignments
    clusters = []

    # to keep track of number of iterations passed
    n_iters = 0
    
    while n_iters < iters:
        # calculate distance of each sample point from each cluster mean
        distances = cdist(X, means ,'euclidean')
        
        # assign each sample point the closest cluster depending upon their
        # distance from the cluster means
        clusters = np.array([np.argmin(i) for i in distances])
        
        # update the cluster means after assigning new clusters
        new_means = []
        for c in range(k):
            new_means.append(X[clusters == c].mean(axis=0))
        
        new_means = np.vstack(new_means)
        
        # stopping condition
        if np.array_equal(new_means, means):
            means = new_means
            break
        else:
            means = new_means
            n_iters += 1
            
        if n_iters == iters:
            print('Maximum number of iterations reached')
        
    distances = cdist(X, means ,'euclidean') 
    clusters = np.array([np.argmin(i) for i in distances])
    
    if return_means:
        return clusters, means

    return clusters

def spectral_kmeans(X, k, n_neighbors, lowest_r, iters=10, return_means=False, initialize=False, initial_means=None):
    n_samples = X.shape[0]

    G = kneighbors_graph(X=X, n_neighbors=n_neighbors, include_self=True).toarray()

    # edge matrix
    W = np.eye(n_samples)

    for i in range(n_samples):
        for j in range(i+1, n_samples):
            if G[i, j] == 1 or G[j, i] == 1:
                W[i, j] = 1
                W[j, i] = 1

    # diagonal
    D = np.diag(W.sum(axis=1)) 

    # Laplacian
    L = D - W

    assert((L == L.T).all())

    evalues, evectors = scipy.linalg.eigh(L)

    # sort eigenvalues 
    idx = range(n_samples)
    sortedIdx = list(list(zip(*sorted(zip(idx, evalues.tolist()), key=lambda x: x[1])))[0])

    # lowest r eigenvectors
    V = evectors[:, sortedIdx][:,0:lowest_r]

    if return_means:
        preds, means = kmeans(X=V, k=k, iters=iters, return_means=return_means)

        return preds, means

    preds = kmeans(X=V, k=k, iters=iters, return_means=return_means)

    return preds







