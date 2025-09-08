# encoding=utf-8
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from scipy import sparse as sp

pca_dim = 90
pca_maxiter = 200

def vector_to_diagonal(vector):
    """
    
    
    :param vector:
    :return:
    """
    if (isinstance(vector, np.ndarray) and vector.ndim == 1) or \
            isinstance(vector, list):
        length = len(vector)
        diag_matrix = np.zeros((length, length))
        np.fill_diagonal(diag_matrix, vector)
        return diag_matrix
    return None


interMatrix = pd.read_csv('MD_A.csv', sep=',', header=0, index_col=0).values

interMatrix = interMatrix.astype('float')
U, S, VT = svds(sp.csr_matrix(interMatrix), k=pca_dim, maxiter=pca_maxiter)
S = vector_to_diagonal(S)

print('microbe vector representation shape:')
print(U.shape)
print('Singular value matrix：')
print(np.sum(S, axis=0))
print('disease vector representation shape:')
print(VT.T.shape)

np.savetxt('SVD_microbe_feature.csv', U, delimiter=',')
np.savetxt('SVD_disease_feature.csv', VT.T, delimiter=',')
