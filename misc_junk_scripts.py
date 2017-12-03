# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 23:48:50 2017

@author: cdonnat
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys
import quandl

path='/Users/cdonnat/Dropbox/Financial Networks/data/'

API_key="VT765CPhSPde_JKHeGT4"
num_key={}
it=0
volume_data={}
for key in stock_data.columns:
    if it<100000:
        name="WIKI/"+str(key)
        print name
        num_key[it]=key
        #data=quandl.get(name,returns='np')
        try:
            data=quandl.get(name,authtoken=API_key, trim_start='2012-01-01', trim_end='2015-01-08',collapse="daily", returns='np')
            volume_data[key]=pd.Series(data['Adj. Volume'], index=data.index)    
        except:
                print "whoops: problem with stock ", key
        it+=1
    else:
        break
Volume_data=pd.DataFrame(volume_data,columns=volume_datakeys())
Volume_data.to_pickle(path+"volume_data.pkl") 