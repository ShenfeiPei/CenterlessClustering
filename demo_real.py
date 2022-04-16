import os
import sys
import time
import numpy as np
import scipy.io as sio
import funs as Ifuns, funs_graph as Gfuns, funs_metric as Mfuns
import toy_generate
import matplotlib.pyplot as plt

from KSUMS2 import KSUMS
from KSUMSX_eigen import KSUMSX

data_name = "FaceV5"
X, y_true, N, dim, c_true = Ifuns.load_mat("data/"+data_name)
X = X.astype(np.float64)

# ksums
knn = int(N/c_true * 1.2)
t_start = time.time()
NN, NND = Gfuns.knn_f(X, knn)
t_end = time.time()
t1 = t_end - t_start

obj = KSUMS(NN.astype(np.int32), NND, c_true, debug=False, max_dd=-1)
obj.opt(rep=50, ITER=100, our_init=1)

acc = Mfuns.multi_accuracy(y_true, obj.y_pre)
nmi = Mfuns.multi_nmi(y_true, obj.y_pre)
ari = Mfuns.multi_ari(y_true, obj.y_pre)
print(f"{np.mean(acc):.3f}, {np.mean(nmi):.3f}, {np.mean(ari):.3f}, {np.mean(obj.times) + t1:.3f}")

# ksumsx
obj = KSUMSX(X, c_true, debug=False)
init_Y = Ifuns.initialY("random", X.shape[0], c_true, rep=50, X=X)
obj.opt(init_Y, b=200, ITER=100)

acc = Mfuns.multi_accuracy(y_true, obj.y_pre)
nmi = Mfuns.multi_nmi(y_true, obj.y_pre)
ari = Mfuns.multi_ari(y_true, obj.y_pre)
print(f"{np.mean(acc):.3f}, {np.mean(nmi):.3f}, {np.mean(ari):.3f}, {np.mean(obj.times):.3f}")
