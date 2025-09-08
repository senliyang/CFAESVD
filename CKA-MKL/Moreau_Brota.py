1# -*- coding: utf-8 -*-
"""
"""


import numpy as np
np.random.seed(2024)
def getMB( np_array):
    """
     Moreau_Brota
    input:
        np_array 
    output:
        Broto_array
    """
    Max = np_array.max(axis=0)#求平均值
    Min = np_array.min(axis=0)#求方差
    Broto_array=(np_array-Min)/(Max-Min)#归一化后的数组#广播#点成*#乘积dot
    return Broto_array#两种归一化方法

def getMB_double( train , test):
    """
    input:
        train
        test
    output:
        MB_ed train
        MB_ed test
    """
    n = len(train)
    train_test = np.vstack( [train , test] )#垂直拼接
    Max = train_test.max(axis=0)#求平均值
    Min = train_test.min(axis=0)#求方差
    Broto_array=(train_test-Min)/(Max-Min)#归一化后的数组#广播#点成*#乘积dot
    return Broto_array[0:n,:] , Broto_array[n: ,:]

def get_z(array):
    '''
    z-score 归一化算法
    x* = （ x- 均值 ） / 标准差
    '''
    AVE = array.mean(axis=0)
    STD = array.std(axis=0)
    return (array - AVE) / STD

def get_Med(array):
    '''
    对参数归一化
    M(i,j)/[M(j,j)**0.5  *  M(i,i)**0.5]
    '''
    l = len(array)
    re = np.zeros([l,l])
    for i in range(l):
        for j in range(l):
            re[i][j] = array[i][j] / ((array[i][i]**0.5)*(array[j][j]**0.5))
    return re

def get_mu(array):
    s = 0
    for i in range(len(array)):
        s = s + array[i]**2
    return array/(s**0.5)
    
    

