# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:44:39 2021

@author: qianyuqing
"""
import scipy.io as scio
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
np.random.seed(2024)
# def get_train_label( y , cv_index , cv_i ):
#     n_microbe = len(y)
#     n_disease = len(y[0])
#     y = y.reshape( n_microbe*n_disease )
#     y_label = y[cv_index==cv_i]
#     y[np.where(cv_index==cv_i)]=0
#     y_train = y
#     y_train = y_train.reshape( [n_microbe,n_disease] )
#     return y_train , y_label
def get_train_label(data, cv_index, cv_i):
    """
    根据交叉验证的索引分配训练标签
    """
    y_train = np.zeros_like(data)
    y_label = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if cv_index[i * data.shape[1] + j] == cv_i:
                y_label[i, j] = data[i, j]
            else:
                y_train[i, j] = data[i, j]
    return y_train, y_label

def get_pre( y_pre , cv_index , cv_i ):
    n_drug = len(y_pre)
    n_effect = len(y_pre[0])
    y_pre = y_pre.reshape( n_drug*n_effect )
    pre = y_pre[ cv_index==cv_i ]
    return pre

def get_auc_aupr( y_true , y_pre ):
    auc_score = roc_auc_score(y_true, y_pre)
    aupr_score = average_precision_score(y_true, y_pre)
    return auc_score,aupr_score


def wknkn( y , Similar_1 , Similar_2 , knn , miu ):
    n = len(y)
    m = len(y[0])
    
    y_d = np.zeros([n,m])
    y_t = np.zeros([m,n])
    
    index = np.argsort(Similar_1 , 1)
    index = index[:,::-1]
    index = index[:,:knn]
    
    for d in range(n):
        w_i = np.zeros([1,knn])
        for ii in range(knn):
            w_i[0,ii] = (miu**ii)*Similar_1[d,index[d,ii]]
        #normalization term
        z_d = 1/sum(Similar_1[d,index[d,:]])
        
        y_d[d,:] = z_d*(np.dot(w_i,y[index[d,:],:]))
        
    index = np.argsort(Similar_2 , 1)
    index = index[:,::-1]
    index = index[:,:knn]
    
    for t in range(m):
        w_i = np.zeros([1,knn])
        for ii in range(knn):
            w_i[0,ii] = (miu**ii)*Similar_2[t,index[t,ii]]
        #normalization term
        z_t = 1/sum(Similar_2[t,index[t,:]])
        
        y_t[t,:] = z_t*(np.dot(w_i,y.T[index[t,:],:]))
        
    y_dt = (y_d+y_t.T)/2
    f_new = np.fmax(y,y_dt)
    return f_new
    
def getACC(y_true, y_pre ):
    """
    y_true是n*1维度数组
    y_pre在模型计算后是1*n维数组
        这里y_pre.resize(len(y_pre),1)已经经过变换！！！
    """
    length = len(y_true)
    y_pre.resize(len(y_pre),1)
    right = np.count_nonzero(y_true==y_pre)
    ACC = round(right / length , 4)
    return ACC
    
def my_confusion_matrix(y_true, y_pre):
    """
    y_true:numpy矩阵
    获得con矩阵
    TP  FP
    TN  FN
    """
    y_pre.resize(len(y_pre),1)
    return confusion_matrix(y_true,y_pre)


def getMCC(con):
    TP=con[0,0]
    FN=con[0,1]
    FP=con[1,0]
    TN=con[1,1]
    MCC=(TP*TN-FP*FN)/((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5
    MCC=round(MCC,4)
    return MCC
    
def getSN(con):
    TP=con[0,0]
    FN=con[0,1]
    SN=TP/(TP+FN)
    SN=round(SN*100,4)
    return SN

def getSP(con):
    FP=con[1,0]
    TN=con[1,1]
    SP=TN/(TN+FP)
    SP=round(SP*100,4)
    return SP
    
    
            























