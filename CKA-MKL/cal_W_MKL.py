# -*- coding: utf-8 -*-
"""

min 1/2 XT*P*X + qT*X
    Gx <= h
    Ax = b
"""
import numpy as np
from cvxopt  import solvers,matrix

np.random.seed(2024)
def get_CKA_Wi(P , q , G , h , A , b):
    '''
    数值如下：除了b都是matrix'
    l = 6
    P = get_P(train_x_k_list , gamma_list)
    q = get_q(train_x_k_list , train_y)
    G = np.identity(l)
    h = np.zeros([l,1])
    A = np.ones([1,l])
    b=matrix(1.)
    '''
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    sol = solvers.qp(P,q,G,h,A,b)
    print(sol['x'])
    return (sol['x'])
    
    
    
    
def get_trace(a , b):
    '''
    计算<a , b>F
    Trace(a.T*b)
    '''
    # print(f'Calculating trace of matrices with shapes: a={a.shape}, b={b.shape}')

    return np.trace(np.dot(a.T , b))
    
def get_P(train_x_k_list ):
    '''
    获得P矩阵
    input:train_x_list 
    output: P矩阵
    '''
    l = len(train_x_k_list)
    n = len(train_x_k_list[0])#n行
    #计算Un Un = In - (1/n)ln*ln.T
    In = np.identity(n)
    ln = np.ones([n,1])
    Un = In - (1/n) * np.dot(ln,ln.T) 
    #计算P
    P = np.zeros([l,l])
    for i in range(l):
        for j in range(l):
            P[i,j] = get_trace( np.dot(np.dot(Un , train_x_k_list[i]),Un) , np.dot(np.dot(Un , train_x_k_list[j]),Un.T))
            #P[j,i] = P[i,j]#对称矩阵
    return P

def get_q(train_x_k_list , ideal_kernel):
    '''
    input:
        train_x_k_list 
    output:
        train_y 标签
    '''
    l = len(train_x_k_list)
    n = len(train_x_k_list[0])
    #计算Un Un = In - (1/n)ln*ln.T
    In = np.identity(n)
    ln = np.ones([n,1])
    Un = In - (1/n) * np.dot(ln,ln.T)
    #计算Ki 理想核
    Ki = ideal_kernel
    #计算a
    a = np.zeros([l,1])
    for i in range(l):
        a[i,0] = get_trace(np.dot(np.dot(Un , train_x_k_list[i]),Un) , Ki )
    return a





