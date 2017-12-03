# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 00:38:00 2017

@author: cdonnat
"""
import numpy as np
import pandas as pd

def compute_correlation(data_log,Delta_t,plot=False):
    ''' Compute the empricial correation matrices from data of size 30 days
    '''
    cor={}
    K=int(np.floor(data_log.shape[0]*1.0/Delta_t))
    print K
    for t in range(K):
        a=K*t
        if t<K-1:
            b=K*(t+1)
            cor[t]=(data_log.iloc[a:b,:]).corr(method='pearson')
        else:
            cor[t]=(data_log.iloc[a:,:]).corr(method='pearson')
        cor[t].fillna(0)
    if plot==True:
        fig2=plt.figure()
        sns.heatmap(cor[0])
        plt.show()
    return cor

def stats_correlations(cor):
    C=np.triu_indices(cor[0].shape[0],k=1,m=cor[0].shape[0])
    mean_Corr=[None]*(len(cor))
    for t in cor.keys():
        mean_Corr[t]=cor[t].iloc[C[0],C[1]].mean()
    return mean_Corr
        
def threshold_correlation(cor,threshold):
    for t in cor.keys():
        return true