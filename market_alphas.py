# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 23:31:11 2017

@author: cdonnat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys
from regression import *



def compute_weights(stock_data,Volume_data):
    value_data=Volume_data*stock_data
    #print(value_data)
    #Volume_data=pd.DataFrame(Volume_data.mean())
    #Volume_data.columns=['vol']
    #Volume_data/=Volume_data.sum()
    #Volume_data=Volume_data.fillna(0)
    #value_data=stock_data.apply(lambda x: Volume_data.T[x.name]*x,axis=0)
    
    value_data=pd.DataFrame(value_data.mean())
    value_data.columns=['w']
    #value_data=value_data.merge(Volume_data,right_index=True,left_index=True)
    #weights=value_data['val']*value_data['vol']
    #weights=pd.DataFrame(mean_val,index=mean_val.index,columns=['weight'])
    weights=value_data
    weights=weights.fillna(1)
    weights=weights*1.0/weights.sum()
    weights=weights.fillna(0)
    return weights

def compute_market_alpha(stock_data,stock_returns,Volume_data,plot=True):
    weights=compute_weights(stock_data,Volume_data)
    W=pd.concat([weights]*stock_returns.shape[0],1)
    W=W.T
    W.index=stock_returns.index
    r_m=stock_returns*W
    r_m=r_m.iloc[:,:-1]
    r_m=r_m.sum(1)
    r_m1=r_m.diff()
    ind=(np.abs(r_m1-r_m1.mean())>5*r_m1.std())
    indd,=np.where(ind==True)
    if len(indd)>0:
        r_m[indd[-1]]=r_m[indd[-1]-1]
        indd=indd[:-1]
        r_m[indd]=0.5*(np.array(r_m[indd-1].tolist())+np.array(r_m[indd+1].tolist()))
    if plot==True:
        r_m.plot()
    
    return r_m
    
