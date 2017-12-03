# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 07:41:24 2017

@author: cdonnat
"""


import numpy as np
import pandas as pd
from scipy.stats import norm

def compute_cdf(stock_res):
    cdf_dict={}
    for u in stock_res.columns:
        vec=pd.DataFrame(stock_res[u].copy())
        vec['rank']=vec.rank()
        vec['cdf']=vec['rank']/(vec['rank'].max()+1.0)
        cdf_dict[u]=vec
    return cdf_dict


def cdf(u,x,cdf_dict):
    if u not in cdf_dict.keys():
        print 'u not in cdf_dict'
        return np.nan
    else:
        vec=cdf_dict[u]
        value=vec.loc[vec[u]<=x,'cdf']
        if len(value)>0:
            return np.max(value)
        else:
            return 1

def cdfm1(u,x,cdf_dict):
    if u not in cdf_dict.keys():
        print 'u not in cdf_dict'
        return np.nan
    else:
        vec=cdf_dict[u]
        value=vec.loc[vec['cdf']<=x,'cdf']
        if len(value)>0:
            return vec.loc[np.argmax(value),u]
        else:
            return vec[u].min()
            
            
def transformation(stock_residuals,cdf_dict):
    cdf_df=pd.DataFrame(np.zeros(stock_residuals.shape),index=stock_residuals.index, columns=stock_residuals.columns)
    transf_df=pd.DataFrame(np.zeros(stock_residuals.shape),index=stock_residuals.index, columns=stock_residuals.columns)
        
    for u in stock_residuals.columns:
        cdf_df[u]=cdf_dict[u]['cdf']
        transf_df[u]=norm.ppf(cdf_df[u])
    return cdf_df,transf_df

def MC_sample(transf_df,cdf_dict,cor_transf=[],M=[]):
    if len(cor_transf)==0:
        cor_transf=transf_df.corr()
    if len(M)==0:
        M=transf_df.mean()
    z=np.random.multivariate_normal(M,cor_transf)
    v=norm.cdf(z)
    ind=transf_df.columns
    vm1=[cdfm1(ind[i],v[i],cdf_dict) for i in range(len(v))]
    return vm1
    
def evaluate_autocorrelation(stock_returns,N,l):
    diff=stock_returns+stock_returns.diff(l)
    R2=pd.DataFrame(np.zeros((2,stock_returns.shape[1])),columns=stock_returns.columns,index=['beta','R2'])
    diff=diff.fillna(0)
    plt.figure()
    bunch=np.random.choice(stock_returns.columns,N)
    cmap=plt.get_cmap('gnuplot')
    colors=[cmap(i) for i in np.linspace(0,1,N)]
    it_b=0
    for b in bunch:
        
        plt.scatter(diff[b],stock_returns[b],c=colors[it_b],label=b)
        it_b+=1
    plt.legend(loc='upper right')
    model=LinearRegression()
    for u in stock_returns.columns:
        model.fit(diff[u].as_matrix().reshape([-1,1]),stock_returns[u].as_matrix().reshape([-1,1]))
        R2[u][0]=model.coef_[0][0]
        R2[u][1]=model.score(diff[u].as_matrix().reshape([-1,1]),stock_returns[u].as_matrix().reshape([-1,1]))
    return R2
    
def permute_blocks(cdf_dict):
    return True #evaluate_autocorrelastion(stock_returns,N,l)

#if __name__=="__main__":
#    bunch=np.random.choice(stock_returns.columns, 10)
#    plt.figure()
#    x=np.linspace(0,1,500);
#    for b in bunch:
#        plt.plot(x,[cdf(u,xx,cdf_dict) for xx in x],label=b)
#    plt.legend(loc="upper left")
#    plt.title("cdf")
#    plt.show()
