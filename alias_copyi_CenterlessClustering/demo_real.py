import os
import time
import numpy as np
import sys
sys.path.append("/home/pei/CenterlessClustering/CC_code")
from alias_copyi_CenterlessClustering import KSUMS, KSUMSX
from alias_copyi_CenterlessClustering.Public import Funs, Gfuns, Mfuns
from alias_copyi_CenterlessClustering import demoapi

# k-sums
# print("Table 4")
data_name = "FaceV5"
X, y_true, c_true, NN, NND, t1 = demoapi.load_data(data_name)

obj = KSUMS(NN.astype(np.int32), NND, c_true, debug=False, max_dd=-1)
obj.opt(rep=10, ITER=100, our_init=1)
t2 = obj.times

acc = Mfuns.multi_accuracy(y_true, obj.y_pre)
nmi = Mfuns.multi_nmi(y_true, obj.y_pre)
ari = Mfuns.multi_ari(y_true, obj.y_pre)
print(f"{data_name}: {np.mean(acc):.3f}, {np.mean(nmi):.3f}, {np.mean(ari):.3f}, {np.mean(t1 + t2):.3f}")
# paper: 0.956, 0.986, 0.931, 0.295


# k-sums-x
# print("Table 5")
X, y_true, c_true, NN, NND, t1 = demoapi.load_data(data_name)
obj = KSUMSX(X, c_true, debug=False)
init_Y = Funs.initialY("random", X.shape[0], c_true, rep=10, X=X)
obj.opt(init_Y, b=200, ITER=100)

acc = Mfuns.multi_accuracy(y_true, obj.y_pre)
nmi = Mfuns.multi_nmi(y_true, obj.y_pre)
ari = Mfuns.multi_ari(y_true, obj.y_pre)
print(f"{np.mean(acc):.3f}, {np.mean(nmi):.3f}, {np.mean(ari):.3f}, {np.mean(obj.times):.3f}")
# paper: 0.963, 0.986, 0.915, 0.254
