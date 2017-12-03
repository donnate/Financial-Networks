# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 23:29:31 2017

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
from correlation import *
from market_alphas import *
from regression import *
from extract_sectors import *


path='/Users/cdonnat/Dropbox/Financial Networks/data/'
sys.path.append(path)
stock_data=pickle.load(open(path+'stock_data.pkl','rb'))
Volume_data=pickle.load(open(path+'volume_data.pkl','rb'))
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
thres=5*(data_log.as_matrix().reshape([-1,1])).std()
data_log=data_log.applymap(lambda x: cap(x,thres))
if verbose==True:
        plt.figure()
        plt.hist(data_log.as_matrix().reshape([-1,1]))
        plt.title('Histogram for the log returns')
        plt.show()



### threshold
stock_returns=data_log.copy()
r_m=compute_market_alpha(stock_data,stock_returns,Volume_data,plot=True)
r_m.plot()
from datetime import datetime
import matplotlib.dates as mdates
colors=['indianred','lightblue','coral','maroon','gold','orange','royalblue','mediumseagreen','violet','grey']
bunch=np.random.choice(data_log.columns,10)
fig, ax = plt.subplots(figsize=(8,6))
sns.set(style='ticks')
sns.set_context("paper", font_scale=1.3)
plt.plot_date(x=data_log.index, y=r_m,c='black',label='Market',fmt='-')
it_b=0
for b in bunch:
    plt.plot_date(x=data_log.index, y=data_log[b],c=colors[it_b],label=b,fmt='-')
    it_b+=1
plt.legend(loc='upper left')
plt.show()
coeff,intercept,MSE,R2=regress_stock_against_market(data_log,r_m,K=5)
## Check stuff out
plt.hist(R2.mean(0))
plt.hist(MSE.mean(0))
plt.hist(coeff.mean(0))
plt.figure()
plt.scatter(coeff.mean(0),R2.mean(0))
plt.xlabel('coeff')
plt.ylabel('R2')
plt.title('coefficient of the regression against the market vs R2 score')

betas=pd.DataFrame(coeff.mean(0)).T
betas=pd.concat([betas]*data_log.shape[0], ignore_index=True)
betas.index=r_m.index
alphas=pd.DataFrame(intercept.mean(0)).T
alphas=pd.concat([alphas]*data_log.shape[0], ignore_index=True)
alphas.index=r_m.index
r_m_df=pd.concat([r_m]*data_log.shape[1], axis=1)
r_m_df.columns=betas.columns
market_effect=betas*r_m_df+alphas
stock_residuals=stock_returns-market_effect


### Compute a few statistics on the residualized returns:
plt.hist(stock_residuals.mean(0),bins=50)
plt.hist(stock_residuals.var(0),bins=50)

fig, ax = plt.subplots(figsize=(8,6))
sns.set(style='ticks')
sns.set_context("paper", font_scale=1.3)
it_b=0
for b in bunch:
    plt.plot_date(x=data_log.index, y=stock_residuals[b],c=colors[it_b],label=b,fmt='-')
    it_b+=1
plt.legend(loc='upper left')
plt.show()

cor_returns=stock_returns.corr()
cor_res=stock_residuals.corr()

plt.figure()
plt.imshow(cor_returns,cmap='hot')
plt.colorbar()

ind_x,ind_y=np.tril_indices(cor_returns.shape[0])
#mask = np.ones(cor_returns.shape,dtype='bool')
#mask[np.tril_indices(cor_returns.shape[0])] = False
plt.imshow(cor_res )
#cor_ret_flattened=[cor_returns.iloc[i,j] for i,j in ind_x,ind_y]
Betas=pd.DataFrame(betas.iloc[0,:])
Betas.columns=['beta']
Betas=Betas.merge(CompanyInfo.T,right_index=True,left_index=True)
Alphas=pd.DataFrame(alphas.iloc[0,:])
Alphas.columns=['alpha']
CompanySector,CompanyIndustry,CompanyInfo=load_sectors(path)

Betas=Betas.merge(Alphas,right_index=True,left_index=True)
sec=pd.DataFrame(np.unique(Betas['Sector']))
sec['key_sector']=sec.index
ind=pd.DataFrame(np.unique(Betas['Industry']))
ind['key_industry']=ind.index
ind_beta=Betas.index
Betas=Betas.merge(sec,left_on='Sector',right_on=0)
Betas=Betas.merge(ind,left_on='Industry',right_on=0)
del Betas['0_y']
del Betas['0_x']
Betas.index=ind_beta
ind=pd.DataFrame(np.unique(Betas['Industry']))

x=np.linspace(0,1,sec.shape[0])
cmap=plt.get_cmap('plasma')
colors_sec={}
it=0
for u in sec[0]:
    colors_sec[u]=cmap(it*1.0/sec.shape[0])
    it+=1
Betas=Betas.merge(R2_mean,right_index=True,left_index=True)
label_sec=[Betas.loc[u,'Sector'] for u in Betas.index]
label_colors=[colors_sec[Betas.loc[u,'Sector']] for u in Betas.index]
plt.figure()
ax = plt.subplot(111)
for s in sec[0]:
    ind_s=(Betas['Sector']==s)
    try:
        col=colors_sec[s]
    except:
        colors_sec[s]='black'
        col=colors_sec[s]
    plt.scatter(Betas.loc[ind_s,'beta'],Betas.loc[ind_s,'R2'],c=col,label=s)
plt.xlabel('coeff')
plt.ylabel('R2')
plt.title('coefficient of the regression against the market vs R2 score')
#plt.legend(loc='upper left')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
import networkx as nx
G=nx.relabel_nodes(G,{k:list(cor_res.index)[k] for k in range(cor_res.shape[0])})
G=nx.from_numpy_matrix(cor_res.applymap(lambda x:  x==0).as_matrix())


thres_array=np.sort([0.5,0.4,0.2,0.175,0.15,0.125,0.1,0.075,0.05])
degree=pd.DataFrame(np.zeros((Betas.shape[0],len(thres_array))),index=Betas.index,columns=thres_array)
consistency=pd.DataFrame(np.zeros((Betas.shape[0],len(thres_array))),index=Betas.index,columns=thres_array)
diversity=pd.DataFrame(np.zeros((Betas.shape[0],len(thres_array))),index=Betas.index,columns=thres_array)
cor_res=stock_residuals.corr()
for thres in thres_array:
    print thres
    cor_res=cor_res.applymap(lambda x: x*(np.abs(x)>thres))
    print np.min(np.min(np.abs(cor_res))),np.max(np.max(np.abs(cor_res)))
    ## Check diag
    sector_consistency_temp=sector_consistence(Betas,cor_res)
    consistency[thres]=sector_consistency_temp['consistency']
    diversity[thres]=sector_consistency_temp['diversity']
    degree[thres]=sector_consistency_temp['degree']
    #plt.hist(sector_consistency_temp['diversity'])
    #plt.hist(sector_consistency_temp['consistency'],bins=30)
    #plt.hist(sector_consistency_temp['degree'],bins=30)
    
### Most consistent stocks per threshold\
l={}
overlap=pd.DataFrame(np.zeros((Betas.shape[0],len(thres_array))),index=Betas.index,columns=thres_array)
for thres in thres_array:
    l[thres]=consistency[thres][consistency[thres].argsort()>Betas.shape[0]-200]
    overlap.loc[l.index,thres]+=1
plt.figure()
for d in [0,1,2,4,5,10]:    
    plt.plot(thres_array,[np.sum(degree[t]==0) for t in thres_array],label=d)
plt.xtitle('threshold for the correlation')
plt.ytitle('proportion of nodes with given degree')
plt.title('Evolution of the degree with the threshold value')
plt.legend(loc='upper left')
plt.show()


#### Decomposition in the eigenvalue sapce
import pygsp
Gg = pygsp.graphs.Graph(cor_res)
Gg.compute_fourier_basis()
D=np.zeros(Gg.N)
D[:5]=Gg.e[:5]
D=np.diag(D)
Alt=Gg.U*D*U.T
#### Definition of a sufficient statisitic
### Distances between matrices


check_weights


test_weights=weights.merge(pd.DataFrame(CompanyInfo.CUR_MKT_CAP), right_index=True,left_index=True, how='left')
test_weights=test_weights.fillna(0)
test_weights.CUR_MKT_CAP[test_weights.CUR_MKT_CAP=="#N/A Invalid Security"]=0
test_weights.CUR_MKT_CAP=pd.to_numeric(test_weights.CUR_MKT_CAP)
test_weights/=np.sum(test_weights)
test_weights=test_weights.sort_values(by='CUR_MKT_CAP')
plt.plot(test_weights.CUR_MKT_CAP,test_weights.w)



