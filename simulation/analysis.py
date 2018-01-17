from scipy import cluster
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(23)
X = np.random.randn(50, 4)
X
Z = cluster.hierarchy.ward(X)
Z

fig = plt.figure(figsize=(10, 10))
P = dendrogram(Z, count_sort='ascending')
plt.show()
cutree = cluster.hierarchy.cut_tree(Z, n_clusters=4)
type(cutree)
cutree
cutlist = cutree.tolist()
cutlist[10]
