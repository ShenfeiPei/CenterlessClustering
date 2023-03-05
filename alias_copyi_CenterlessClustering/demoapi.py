import os
import time
import numpy as np
import scipy.io as sio
from alias_copyi_CenterlessClustering.Public import Funs, Gfuns, Mfuns

def load_data(data_name):
    current_path = os.path.dirname(__file__)
    data_full_name = os.path.join(current_path, f"data/{data_name}.mat")
    X, y_true, N, dim, c_true = Funs.load_mat(data_full_name)
    X = X.astype(np.float64)

    knn = int(N/c_true * 1.2)
    t_start = time.time()
    NN, NND = Gfuns.knn_f(X, knn)
    t_end = time.time()
    t1 = t_end - t_start
    return X, y_true, c_true, NN, NND, t1

def load_toy(name):
    current_path = os.path.dirname(__file__)
    data_full_name = os.path.join(current_path, f"data/{name}.mat")
    data = sio.loadmat(data_full_name)
    X, y_true, NN, NND, knn_time = data["X"], data["y_true"].reshape(-1), data["NN"], data["NND"], data["knn_time"][0][0]
    return X, y_true, NN, NND, knn_time
