# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 07:41:34 2017

@author: cdonnat
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys

def treat_missing_data(stock_data,verbose=True):
    ''' treat missing values in dataset
    INPUT:
    ==================================================
    stock_data: dataset with missing values
    OUTUT:
    ==================================================
    new_stock_data dataset
    
    '''
    ### Check who has missing values
    nb_missing_values={}
    nb_stocks=stock_data.shape[1]
    for i in range(nb_stocks):
            company=stock_data.columns[i]
            nb_missing_values[company]=np.sum(pd.isnull(stock_data.ix[:,i]))
    if verbose==True:
        plt.figure()
        plt.hist([v for k,v in nb_missing_values.iteritems()])
        plt.title('Histogram for the missing values (per stock)')
        plt.show()
    
    ### Get rid of the stocks which have too much missing values:
    eta=150
    eliminate_stocks=[k for k in nb_missing_values.keys() if nb_missing_values[k]>eta ]
    ### Check which days are missing

    day=np.sum(stock_data.isnull(),axis=1)
    day_missing_values={k:day[k] for k in stock_data.index}
    if verbose==True:
        plt.figure()
        plt.hist([v for k,v in day_missing_values.iteritems()])
        plt.title('Histogram for the missing values (per day)')
        plt.show()
    ### Some companies also don't start at the beginning (drop them)
    late_starts = []
    nb_stocks=stock_data.shape[1]
    for i in range(nb_stocks):
            company=stock_data.columns.values[i]
            if np.sum(pd.isnull(stock_data.ix[0:10,i])) > 20:
                late_starts.append (company)
    
    new_stock_data = stock_data.drop (eliminate_stocks, axis = 1)
    new_stock_data = new_stock_data.apply(pd.Series.interpolate)
    new_stock_data = new_stock_data.bfill ()
    
    ##Sanity check
    nb_stocks=new_stock_data.shape[1]
    for i in range(nb_stocks):
            if np.sum(pd.isnull(new_stock_data.ix[:,i])) > 0:
                print "whoops"
    return new_stock_data,nb_missing_values,day_missing_values
                
def logTransform (ts):
        
    return (np.log(ts)).diff()

def transform_stock(new_stock_data):
    collection= (np.log(new_stock_data)).diff()
    collection=collection.iloc[1:,:]
    collection.fillna(0)
    return collection


def ht(x,thres):
    if np.abs(x)>thres:
        return x
    else:
        return 0

def cap(x,thres):
    if x>thres:
        return thres
    elif x<-thres:
        return -thres
    else:
        return x
if __name__=='__main__':
    ###load data

    path='/Users/cdonnat/Dropbox/Financial Networks/data/'
    sys.path.append(path)
    stock_data=pickle.load(open(path+'stock_data.pkl','rb'))
    new_stock_data,nb_missing_values,day_missing_values=treat_missing_data(stock_data,verbose=False)
    data_log=transform_stock(new_stock_data)
    verbose=True
    if verbose==True:
        plt.figure()
        plt.hist(data_log.as_matrix().reshape([-1,1]))
        plt.title('Histogram for the log returns')
        plt.show()
    ##
    ### Cap some outliers
    thres=5*(pd.concat([data_log.std()]*data_log.shape[0],1)).T
    thres.index=data_log.index
    serious_outliers=((np.abs(data_log-data_log.mean())>thres))
    f,(ax1,ax2)=plt.subplots(1,2)
    ax1.hist(serious_outliers.sum(0))
    ax1.set_title('Distribution of outlier number accross stocks')
    ax2.hist(serious_outliers.sum(1))
    ax2.set_title('Distribution of outlier number accross days')
    plt.show()
    #data_log=data_log.applymap(lambda x: cap(x,thres))
    if verbose==True:
        plt.figure()
        plt.hist(data_log.as_matrix().reshape([-1,1]))
        plt.title('Histogram for the log returns')
        plt.show()
    
    