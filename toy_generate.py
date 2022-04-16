import time
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.transforms as mt
from sklearn.neighbors import KDTree
import itertools

def turdata2(N, tur1, tur2):
    mu = np.repeat(0, 2)
    sig = np.array([[tur1, 0], [0, tur2]])
    x = np.random.multivariate_normal(mu, sig, N)
    return x

def outlier(N1, N2):
    x1 = turdata2(N1, 0.03, 0.003) + np.array([1, 0.5])
    x2 = turdata2(N1, 0.03, 0.003) + np.array([1, 1])
    x3 = turdata2(N1, 0.03, 0.003) + np.array([1, 1.5])
    x4 = turdata2(N2, 0.001, 0.001) + np.array([3, 1.5])

    y1 = np.repeat(0, N1)
    y2 = np.repeat(1, N1)
    y3 = np.repeat(2, N1)
    y4 = np.repeat(3, N2)

    y = np.concatenate((y1, y2, y3, y4), axis=0).reshape(-1)
    X = np.concatenate((x1, x2, x3, x4), axis=0)
    return X, y


def grid_x(r_num, c_num, num_per_clu=10, noise=0.02):
    c_true = r_num * c_num
    N = num_per_clu * c_true

    Cen = np.meshgrid(np.arange(r_num), np.arange(c_num))
    Cen2 = np.stack(Cen, -1)
    Cen3 = Cen2.reshape(-1, 2)

    Cen4 = np.repeat(Cen3, repeats=num_per_clu, axis=0)

    mean = [0, 0]
    cov = np.eye(2) * noise
    tur_data = np.random.multivariate_normal(mean, cov, size=N)

    X = Cen4 + tur_data
    y = np.repeat(np.arange(c_true), repeats=num_per_clu)

    return X, y


def kdtree(X, knn):
    t1 = time.time()
    tree = KDTree(X, leaf_size=40)
    NND, NN = tree.query(X, k=knn)
    t2 = time.time()
    return NN, NND, t2 - t1

def save(X, y, NN, NND, knn, t, name):
    sio.savemat(name, {"X": X, "y_true": y, "NND": NND, "NN": NN, "knn": knn, "knn_time": t})


def gen_grid(rc, num, knn):
    r_num = rc[0]
    c_num = rc[1]

    c_true = int(r_num*c_num)

    num_per_clu = int(num/c_true)

    print(int(num/1000), num_per_clu, int(c_true/1000))
    X, y = grid_x(r_num, c_num, num_per_clu=num_per_clu, noise=0.03)

    NN, NND, t = kdtree(X, knn)
    return X, y, NN, NND, t

# version 2
rcs = list([[50, 100], [100, 100], [100, 200]])
nums = list([100000, 200000, 300000])
cou = 0
for num in nums:
    for rc in rcs:
        c_true = int(rc[0] * rc[1])
        knn = int(num / c_true * 1.2)
        cou += 1
        X, y, NN, NND, t = gen_grid(rc, num, knn)
        save(X, y, NN, NND, knn, t, f"./toydata/D{cou}_autoknn.mat")


# name = "D1"
# data = sio.loadmat(f"./toydata/{name}.mat")
# X, y_true = data["X"], data["y_true"].reshape(-1)
# ind = np.logical_and(X[:, 0] < 10, X[:, 1] < 10)
# X = X[ind]
# y = y_true[ind]

# plt.figure(figsize=(6.4, 4.8))
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.xticks([])
# plt.yticks([])
# plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
# plt.margins(0.05, 0.05)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.savefig(f"./toydata/{name}.png", dpi = 300, bbox_inches=mt.Bbox([[-0.1, -0.1], [6.5, 4.9]]))


# X, y = outlier(20, 5)
# NN, NND, t = kdtree(X, knn)
# save(X, y, NN, NND, knn, t, "./toydata/toy_out.mat")


