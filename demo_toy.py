import numpy as np
import scipy.io as sio
import funs as Ifuns, funs_graph as Gfuns, funs_metric as Mfuns
from KSUMS2 import KSUMS

name = "toy_out" # D1_autoknn
data = sio.loadmat(f"data/{name}.mat")
X, y_true, NN, NND, knn_time = data["X"], data["y_true"].reshape(-1), data["NN"], data["NND"], data["knn_time"][0][0]
c_true = len(np.unique(y_true))
print(X.shape, c_true)

# ksums
obj = KSUMS(NN.astype(np.int32), NND, c_true, debug=False, max_dd=-1)
obj.opt(rep=50, ITER=100, our_init=1)
Y = obj.y_pre
times = obj.times

pre = Mfuns.multi_precision(y_true, Y)
rec = Mfuns.multi_recall(y_true, Y)
f1 = 2 * pre * rec / (pre + rec)
print(f"{np.mean(f1):.3f}, {np.mean(times) + knn_time:.3f}")

if name == "toy_out":
    c_true = 3
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    y_km = KMeans(n_clusters=c_true).fit(X).labels_

    fig = plt.figure()
    ax1 = fig.add_subplot(121)    # The big subplot
    ax2 = fig.add_subplot(122)
    ax1.scatter(X[:, 0], X[:, 1], c=y_km)
    ax1.set_xlabel("k-means")
    ax2.scatter(X[:, 0], X[:, 1], c=Y[0])
    ax2.set_xlabel("k-sums")
    plt.show()