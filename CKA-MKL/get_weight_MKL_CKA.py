# -*- coding: utf-8 -*-
"""
@author: Administrator
"""
import numpy as np
from cal_W_MKL import get_P , get_q ,get_CKA_Wi
from cal_W_cos import get_WW
from Moreau_Brota import get_Med,get_mu
from cvxopt  import matrix
from kernel import  kernel_normalized
def load_kernel_from_file(file_path):
    """"""
    
    return np.loadtxt(file_path)
def get_n_weight(k_train_list ,ideal_kernel ,lambd):
    '''
    input：
        k_train_list
    output：
        weight weight
    '''
    n = len(k_train_list)
    Wij = np.zeros([n,n])
    for i in range(n):#Wij 
        for j in range(i,n):
            Wij[i][j] = get_WW(k_train_list[i],k_train_list[j])
            Wij[j][i] = Wij[i][j]
    D = Wij.sum(axis = 0)
    Dii = np.zeros([n,n])
    for i in range(n):
        Dii[i][i] = D[i]
    L = Dii - Wij
    L=abs(L)
    M = get_Med( get_P(k_train_list) ) #归一化
    P = M + lambd*L
    a = get_mu ( get_q(k_train_list , ideal_kernel) )
    q = -1*a
    G = -1 * np.identity(n)
    h = np.zeros([n,1])
    A = np.ones([1,n])
    b=matrix(1.)
    return get_CKA_Wi(P , q , G , h , A , b)

from Tool import get_train_label
data = np.loadtxt(r'data/MD_A.txt')
n_microbe = len(data)  #1177
n_disease = len(data[0])       #134
cv = 5
np.random.seed(2024)
cv_index = np.random.randint(cv, size=n_microbe*n_disease)
# 打印cv_index的分布
print("cv_index分布:", np.bincount(cv_index))
k_train_list = [
        load_kernel_from_file('data/disease/sem_DS.txt'),
        load_kernel_from_file('data/disease/cos_DS.txt'),
        load_kernel_from_file('data/disease/Ga_DS.txt')
]
for cv_i in range(cv):
    y_train, y_label = get_train_label(data, cv_index, cv_i)
    print(f"cv_i: {cv_i}, y_train非零元素数目: {np.count_nonzero(y_train)}")
    side_ideal_kernel = np.dot(y_train.T, y_train)
    side_k_nor = kernel_normalized(side_ideal_kernel) 
    weights = get_n_weight(k_train_list, side_k_nor, 0.8)
    k_s = np.zeros([n_disease, n_disease])
    for i in range(len(k_train_list)):
        k_s = k_s + weights[i] * k_train_list[i]
    np.savetxt('../../data/disease/CKA-disease.txt', k_s, delimiter=' ')
k_train_list = [
    load_kernel_from_file('data/microbe/fun_MS.txt'),
    load_kernel_from_file('data/microbe/cos_MS.txt'),
    load_kernel_from_file('data/microbe/Ga_MS.txt')
 ]
for cv_i in range(cv):
         y_train, y_label = get_train_label(data, cv_index, cv_i)
         print(f"cv_i: {cv_i}, y_train非零元素数目: {np.count_nonzero(y_train)}")
         side_ideal_kernel = np.dot(y_train, y_train.T)
         side_k_nor = kernel_normalized(side_ideal_kernel)
         weights = get_n_weight(k_train_list, side_k_nor, 0.8)
         k_s = np.zeros([n_microbe, n_microbe])
         for i in range(len(k_train_list)):
             k_s = k_s + weights[i] * k_train_list[i]
         np.savetxt('data/microbe/CKA-microbe.txt', k_s, delimiter=' ')

