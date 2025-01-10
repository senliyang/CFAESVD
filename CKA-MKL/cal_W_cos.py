# -*- coding: utf-8 -*-
"""


"""
import numpy as np
np.random.seed(2024)


def get_WW(t1 , t2):
    '''
    
    '''
    fenzi = np.trace(np.dot(t1,t2))
    fenmu = ((np.trace(np.dot(t1 , t1)))*(np.trace(np.dot(t2 , t2))))**0.5
    return round(fenzi / fenmu , 4)
    
    
    
    
    
    
    
    
    