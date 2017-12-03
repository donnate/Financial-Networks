# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 00:09:21 2017

@author: cdonnat
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys
import quandl
from utils_preprocessing import *
path='/Users/cdonnat/Dropbox/Financial Networks/scripts/scripts_Claire/'

API_key="VT765CPhSPde_JKHeGT4"
quandl.ApiConfig.api_key = 'VT765CPhSPde_JKHeGT4'
num_key={}
it=0
volume_data={}
year_starts=[str(y)+'-01-01' for y in range(2005,2016)]
year_starts2=[str(y)+'-01-10' for y in range(2005,2016)]
year_ends=[str(y)+'-12-31' for y in range(2005,2016)]
stocks_y={}
stock_data={}
Stock_data={}
year=range(2005,2016)
nb_missing_values={}
day_missing_values={}
Data_log={}
for i in range(len(year_starts)):
    data = quandl.get_table('WIKI/PRICES', date = { 'gte': year_starts[i], 'lte': year_starts2[i] })
    stocks_y[year[i]]=list(set(data['ticker']))
    it=0
    stock_data[year[i]]={}
    volume_data={}
    for key in stocks_y[year[i]]:
        if it<100000:
            name="WIKI/"+str(key)
            print name
            #data=quandl.get(name,returns='np')
            try:
                data=quandl.get(name,authtoken=API_key, trim_start=year_starts[i], trim_end=year_ends[i],collapse="daily", returns='np')
                volume_data[key]=pd.Series(data['Adj. Volume'], index=data.index)  
                stock_data[year[i]][key]=data['Adj. Close']
            except:
                    print "whoops: problem with stock ", key
            it+=1
        else:
            break
    Volume_data=pd.DataFrame(volume_data,columns=volume_data.keys())
    Volume_data.to_pickle(path+"data/volume_data"+str(year[i])+".pkl") 
    Stock_data[year[i]]=pd.DataFrame(stock_data[year[i]],columns=stock_data[year[i]].keys())
    Stock_data[year[i]].to_pickle(path+"data/stock_data"+str(year[i])+".pkl") 
    new_stock_data,nb_missing_values[year[i]],day_missing_values[year[i]]=treat_missing_data(Stock_data[year[i]],verbose=True)
    data_log=transform_stock(new_stock_data)
    thres=8*(pd.concat([data_log.std()]*data_log.shape[0],1)).T
    thres.index=data_log.index
    serious_outliers=(np.abs(data_log-data_log.mean())>thres)
    sign=(data_log-data_log.mean()>0)
    serious_outliers2=(data_log.mean()+sign*thres-data_log)*serious_outliers
    data_log=data_log+serious_outliers2 
    Data_log[year[i]]=data_log
    data_log.to_pickle(path+"data/data_log"+str(year[i])+".pkl") 
    
    
    
    