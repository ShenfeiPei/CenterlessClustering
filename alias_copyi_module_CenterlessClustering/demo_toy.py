import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from alias_copyi_module_CenterlessClustering import KSUMS
# from KSUMS2 import KSUMS

def demo():
    name = "toy_out"
    current_path = os.path.dirname(__file__)
    data_full_name = os.path.join(current_path, f"data/{name}.mat")
    data = sio.loadmat(data_full_name)
    X, y_true, NN, NND, knn_time = data["X"], data["y_true"].reshape(-1), data["NN"], data["NND"], data["knn_time"][0][0]
    y_true[-5:] = 0
    c_true = len(np.unique(y_true))
    print(X.shape, c_true)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)   
    ax2 = fig.add_subplot(122)

    # ksums
    obj = KSUMS(NN.astype(np.int32), NND, c_true, debug=False, max_dd=-1)
    obj.opt(rep=50, ITER=100, our_init=1)
    Y = obj.y_pre
    # times = obj.times
    ax2.scatter(X[:, 0], X[:, 1], c=Y[0])
    ax2.set_xlabel("k-sums")

    # k-means
    y_km = KMeans(n_clusters=c_true).fit(X).labels_
    ax1.scatter(X[:, 0], X[:, 1], c=y_km)
    ax1.set_xlabel("k-means")

    plt.show()

if __name__ == "__main__":
    demo()
