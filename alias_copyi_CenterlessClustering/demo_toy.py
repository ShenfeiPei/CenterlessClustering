import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from alias_copyi_CenterlessClustering import KSUMS, KSUMSX
from alias_copyi_CenterlessClustering.Public import Funs, Gfuns, Mfuns
from alias_copyi_CenterlessClustering import demoapi
from alias_copyi_CenterlessClustering import KSUMS

name = "toy_out"

X, y_true, NN, NND, knn_time = demoapi.load_toy(name)
c_true = len(np.unique(y_true)) - 1

# ksums
obj = KSUMS(NN.astype(np.int32), NND, c_true, debug=False, max_dd=-1)
obj.opt(rep=50, ITER=100, our_init=1)
Y = obj.y_pre
# times = obj.times
plt.scatter(X[:, 0], X[:, 1], c=Y[0])
plt.show()

# k-means
y_km = KMeans(n_clusters=c_true).fit(X).labels_
plt.scatter(X[:, 0], X[:, 1], c=y_km)
plt.show()
