# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 00:35:49 2017

@author: cdonnat
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error

def split_OIS(data,k=5,size_blocks=1):
    if size_blocks==1:
        #test=data.index.tolist()
        test=range(data.shape[0])
        np.random.shuffle(test)
        K=int(np.floor(data.shape[0]/k))
        test_ind={}
        for t in range(k):
            if t<k-1:
                test_ind[t]=test[K*t:K*(t+1)]
            else:
                test_ind[k]=test[K*t:]
    else:
        test=np.arange(0,data.shape[0],size_blocks)
        end=test[-1]
        np.random.shuffle(test)
        K=int(np.floor(len(test)/k))
        test_ind={}
        for t in range(k):
                
            l=[]
            if t==k-1:
                end=len(test)
            else:
                end=K*(t+1)
            for tt in test[K*t:end]:
                add=tt+np.arange(size_blocks)
                if add[-1]>data.shape[0]-1:
                    add=np.arange(tt,data.shape[0])
                l+=list(add)
     
            test_ind[t]=l
            #data.index[l]
            
        return test_ind
        
def regress_stock_against_market(stock_return,r_m,K=5):
    test_split=split_OIS(stock_return,k=K,size_blocks=5)
    MSE=pd.DataFrame(np.zeros((K,stock_return.shape[1])),index=range(K),columns=stock_return.columns)
    R2=pd.DataFrame(np.zeros((K,stock_return.shape[1])),index=range(K),columns=stock_return.columns)
    coeff=pd.DataFrame(np.zeros((K,stock_return.shape[1])),index=range(K),columns=stock_return.columns)
    intercept=pd.DataFrame(np.zeros((K,stock_return.shape[1])),index=range(K),columns=stock_return.columns)
        
    for k in range(K):
        model=LinearRegression()
        test_set=np.array([False]*stock_return.shape[0])
        test_set[test_split[k]]=True
        X=r_m.as_matrix()[~test_set]
        X_test=r_m.as_matrix()[test_set]
        for u in stock_return.columns:
            Y=np.array(stock_return[u].tolist())[~test_set]
            Y_test=np.array(stock_return[u].tolist())[test_set]
            Y[np.isnan(Y)]=0
            Y_test[np.isnan(Y_test)]=0
            model.fit(X.reshape([-1,1]),Y.reshape([-1,1]))
            coeff.loc[k,u]=model.coef_[0][0]
            intercept.loc[k,u]=model.intercept_[0]
            R2.loc[k,u]=model.score(X.reshape([-1,1]),Y.reshape([-1,1]))
            pred=model.predict(X_test.reshape([-1,1]))
            MSE.loc[k,u]=mean_squared_error(Y_test, pred)
    return coeff,intercept,MSE,R2