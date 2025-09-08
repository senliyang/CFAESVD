# -*- coding: utf-8 -*-
"""

"""
import numpy as np
from sklearn import metrics
np.random.seed(2024)

def kernel_normalized(k):
    #理想核矩阵的归一化
    n = len(k)

    
    k = np.abs(k)
    index_nozeros = k.nonzero()

    min_value = min(k[index_nozeros])
    k[np.where(k==0)] = min_value
    
    diag = np.resize(np.diagonal(k), [n,1])**0.5
    k_nor = k/(np.dot(diag,diag.T))
    return k_nor
            
    


