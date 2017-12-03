# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 00:01:14 2017

@author: cdonnat
"""

### Defines different regimes

def separate_regimes(data_log,alpha=2,thres=0.7):
    std_returns=data_log.std()
    mean_returns=data_log.mean()
    data_log2=data_log.copy()
    data_log3=data_log2.applymap(lambda x: (x-mean_returns[x.name])<alpha*std_returns[x.name])
    t=785-np.sum(data_log3,0)
    nn={}
    for u in data_log.columns:
        ind=list(np.where(data_log3[u]==False)[0])
        corr_u=(data_log.iloc[ind,:]).corr()
        corr_u=corr_u[u]
        corr_u.sort_values(inplace=True)
        nn[u]=list(corr_u.loc[corr_u>thres].index)
        nn[u].remove(u)
        corr_u=(data_log.iloc[ind,:]).corr()
        corr_u=corr_u[u]
        corr_u.sort_values(inplace=True)
        
    return nn
        
        
        