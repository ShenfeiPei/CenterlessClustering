import numpy as np
import sys
sys.path.append("/home/pei/KSUMS_Journal/KSUMS2_code/")
from KSUMSX_eigen import KSUMSX

X = np.zeros((3,3), dtype=np.float64)
mod = KSUMSX(X, 3, 1)