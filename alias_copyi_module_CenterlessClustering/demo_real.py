import os
import time
import numpy as np

from alias_copyi_module_CenterlessClustering import KSUMS, KSUMSX
from alias_copyi_module_CenterlessClustering.Public import Funs, Gfuns, Mfuns
# from KSUMS2 import KSUMS
# from KSUMSX_eigen import KSUMSX
# from Public import Mfuns, Gfuns, Funs

def demo_real():
    data_name = "FaceV5"
    current_path = os.path.dirname(__file__)
    data_full_name = os.path.join(current_path, f"data/{data_name}.mat")
    X, y_true, N, dim, c_true = Funs.load_mat(data_full_name)
    X = X.astype(np.float64)

    # ksums
    knn = int(N/c_true * 1.2)
    t_start = time.time()
    NN, NND = Gfuns.knn_f(X, knn)
    t_end = time.time()
    t1 = t_end - t_start

    obj = KSUMS(NN.astype(np.int32), NND, c_true, debug=False, max_dd=-1)
    obj.opt(rep=50, ITER=100, our_init=1)
    t2 = obj.times

    acc = Mfuns.multi_accuracy(y_true, obj.y_pre)
    nmi = Mfuns.multi_nmi(y_true, obj.y_pre)
    ari = Mfuns.multi_ari(y_true, obj.y_pre)
    print(f"{np.mean(acc):.3f}, {np.mean(nmi):.3f}, {np.mean(ari):.3f}, {np.mean(t1 + t2):.3f}")
    # paper: 0.956, 0.986, 0.931, 0.295


    # ksumsx
    obj = KSUMSX(X, c_true, debug=False)
    init_Y = Funs.initialY("random", X.shape[0], c_true, rep=50, X=X)
    obj.opt(init_Y, b=200, ITER=100)

    acc = Mfuns.multi_accuracy(y_true, obj.y_pre)
    nmi = Mfuns.multi_nmi(y_true, obj.y_pre)
    ari = Mfuns.multi_ari(y_true, obj.y_pre)
    print(f"{np.mean(acc):.3f}, {np.mean(nmi):.3f}, {np.mean(ari):.3f}, {np.mean(obj.times):.3f}")
    # paper: 0.963, 0.986, 0.915, 0.254

if __name__ == "__main__":
    demo_real()
