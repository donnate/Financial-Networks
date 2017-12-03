# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 00:20:21 2017

@author: cdonnat
"""

import numpy as np
import pandas as pd
):
    s=stock_residuals.std()
    m=stock_residuals.median()
    outliers=stock_residuals.apply(lambda x: (np.abs(x-m[x.name])>10*s[x.name]),0)
    outliers_ind=np.where(outliers)
    for i in range(10):
        #fig,(ax1,ax2)=plt.subplots(2,1,sharex=False,sharey=True);
        print('here')
        plt.figure()
        plt.scatter(range(stock_residuals.shape[0]),stock_residuals.iloc[:,outliers_ind[1][i]])
        ind=np.where(outliers_ind[1]==outliers_ind[1][i])
        stock=stock_residuals.columns[outliers_ind[1][i]]
        x=outliers_ind[0][ind]
        print(x)
        
        plt.scatter(new_stock_data.index[x],stock_residuals.loc[new_stock_data.index[x],stock],c='red')
        plt.title(stock)
        plt.show()
        print(stock)
        plt.figure()
        plt.plot(new_stock_data[stock]);
        plt.scatter(new_stock_data.index[x],new_stock_data.loc[stock_residuals.index[x],stock],c='red')
        print('here')
        plt.title(stock)
        plt.show()
        print(i)
        
    nb_outlying_stocks=stock_residuals.columns[np.unique(outliers_ind[1])]
    days=stock_residuals.index[np.unique(outliers_ind[0])]
    
    


### Clustering algorith