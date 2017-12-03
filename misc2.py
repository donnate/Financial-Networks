# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 07:51:43 2017

@author: cdonnat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys
import quandl
path='/Users/cdonnat/Dropbox/Financial Networks/scripts/scripts_Claire/'
sys.path.append(path)

from utils_preprocessing import *
from correlation import *
from market_alphas import *
from regression import *
from extract_sectors import *

years=range(1995,2016,1)
R_m={}
 
path='/Users/cdonnat/Dropbox/'
from tree_attempt import *
import scipy as sc
years=range(1995,2000,1)
B=200


import networkx as nx



A={}
G={}
Boot={}
cor_res={}
edges={}
nodes={}
Stop_year=2016
path='/Users/cdonnat/Dropbox/'
path_sectors='/Users/cdonnat/Dropbox/Financial Networks/data/'
tree={}
first_pass=True
import networkx as nx
CompanyInfo=load_sectors2(path_sectors,filename='book_CompanyInfo.csv')
if first_pass==True:
    for y in range(1995,Stop_year):
        stock_data=pickle.load(open(path+"data/stock_data"+str(y)+'.pkl','rb'))
        stock_residuals=pickle.load(open(path+'data/stock_res'+str(y)+'.pkl','rb'))
        tree[y]=pickle.load(open(path+"data/tree_B2_"+str(y)+'.pkl','rb'))
        nodes[y]=pd.DataFrame.from_csv(path+"data/nodes_B2_"+str(y)+'.csv')
        edges[y]=pd.DataFrame.from_csv(path+"data/edges_B2_"+str(y)+'.csv')
        edges[y]['weight']=pd.to_numeric(edges[y]['weight'])
        G[y]=nx.from_pandas_dataframe(edges[y],source='Source',target='Target',edge_attr='weight')
        print(G[y]['AGN'])
        A[y]=nx.adjacency_matrix(G[y]).todense()
        A[y]=pd.DataFrame(A[y],index=G[y].nodes(),columns=G[y].nodes())
    
else:
    for y in range(2009,2016):
        print(y)
        stock_data=pickle.load(open(path+"data/stock_data"+str(y)+'.pkl','rb'))
        stock_residuals=pickle.load(open(path+'data/stock_res'+str(y)+'.pkl','rb'))
        if stock_residuals.shape[0]>stock_residuals.shape[1]:
             stock_residuals=stock_residuals.T
        Volume_data=pickle.load(open(path+'data/volume_data'+str(y)+'.pkl','rb'))
        weights=compute_weights(stock_data,Volume_data)
        edges[y]=pd.DataFrame.from_csv('/Users/cdonnat/Dropbox/data/edges_B_'+str(y)+'.csv')
        edges[y]['weight']=pd.to_numeric(edges[y]['weight'])
        nodes[y]=pd.DataFrame.from_csv('/Users/cdonnat/Dropbox/data/nodes_B_'+str(y)+'.csv')
        cor_res[y]=pickle.load(open('/Users/cdonnat/Dropbox/data/cor_res'+str(y)+'.pkl','rb'))
        Boot[y]=pickle.load(open('/Users/cdonnat/Dropbox/data/Boot'+str(y)+'.pkl','rb'))
        Boot[y]+=cor_res[y]
        Boot[y]*=1.0/(Boot[y].iloc[0,0])
        
        tree[y],count,weights2=agg_clustering(Boot[y],weights,stock_residuals,stepsize=0.05,verbose=False)
        pickle.dump(tree[y],open('/Users/cdonnat/Dropbox/data/tree_B2_'+str(y)+'.pkl','wb'))
        e,n=edges_tree(tree[y],CompanyInfo,weights2,name_edges='edges_B2_'+str(y),name_nodes='nodes_B2_'+str(y),\
                          path='/Users/cdonnat/Dropbox/data/')
        e['weight']=pd.to_numeric(e['weight'])
        edges[y]=e
        nodes[y]=n
        G[y]=nx.from_pandas_dataframe(e,source='Source',target='Target',edge_attr='weight')
        #print(G[y]['AGN'])
        A[y]=nx.adjacency_matrix(G[y]).todense()
        A[y]=pd.DataFrame(A[y],index=G[y].nodes(),columns=G[y].nodes())
        CompanyInfo,pb= extend_label_nodes(A,nodes,y,path_sectors='/Users/cdonnat/Dropbox/Financial Networks/data/')
        e,n=edges_tree(tree[y],CompanyInfo,weights2,name_edges='edges_B2_'+str(y),name_nodes='nodes_B2_'+str(y),\
                          path='/Users/cdonnat/Dropbox/data/')
        e['weight']=pd.to_numeric(e['weight'])
        edges[y]=e
        nodes[y]=n

path_sectors='/Users/cdonnat/Dropbox/Financial Networks/data/'



import operator

def extend_label_nodes(A,nodes,y=1995,path_sectors='/Users/cdonnat/Dropbox/Financial Networks/data/'):
    CompanyInfo=load_sectors2(path_sectors,filename='book_CompanyInfo.csv')
    pb=[]
    Sectors_lev={}
    for xx in np.linspace(0,2,int(2.0/0.05)):
        list_nn=nodes[y].Id.apply(lambda x: True if str(x)[:8]=='lev_'+str(1-xx)[:4] else False)
        list_nn=(nodes[y].loc[list_nn])['Id']
        #print(list_nn)
           
        if len(list_nn)>0:
            for n in list_nn:
                neighbors=A[y].loc[n,A[y][n]>=float(str(1-xx)[:4])]
                #print(neighbors)
                #print(1-xx)
                #print(n+ 'has '+str(len(neighbors))+' neighbors')
                
                sectors={k:0 for k in np.unique(CompanyInfo.Sector_key)}
                for nn in neighbors.index:
                    #print nn
                    try: 
                        sectors[CompanyInfo.loc[nn,'Sector_key']]+=1
                    except:
                        pass
                sectors=pd.DataFrame.from_dict(sectors,orient='index')
                sectors.columns=['count']
                sectors=sectors.sort_values(by='count',ascending=False)
                nodes[y].loc[n,'keys_s']=sectors.index[0]
                CompanyInfo.loc[n,'Sector_key']=sectors.index[0]
                Sectors_lev[n]=sectors
                if sectors.iloc[0,0]==sectors.iloc[1,0]:
                    pb.append((n,sectors))
    return CompanyInfo,pb

def cluster_tree(tree):
    return true


from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score

      


def cluster_at(n,y,tree):
    #C=A[y][n][A[y][n]>lev]
    k=float(n[4:8])
    lv=float(n[4:8])+np.min(np.array([kk -float(n[4:8]) for kk in tree[y].keys() if kk>float(n[4:8])]))
    neighbors=pd.DataFrame(tree[y][lv][float(n[9:])])
    neighbors.columns=['Id']
    #if len(not_candidate_supernodes)==0
    list_nn=neighbors.Id.apply(lambda x: True if (str(x)[:4]=='lev_') else False)
    C=neighbors.loc[~list_nn]
    CC=[c[0] for c in C.values]
    list_nn=(neighbors.loc[list_nn])['Id']
    list_nn_s=list_nn.apply(lambda x: True if float(x[4:8])>=float(n[4:8]) else False)
    list_nn=list(list_nn[list_nn_s])

    if len(list_nn)==0:
        return CC,[]
    else:
        print(list_nn)
        print 'CC is:', CC
        list_nn2=[l for l in list_nn]
        for nn in list_nn:
            res=cluster_at(nn,y,tree)
            CC+=list(res[0])
            list_nn2+=list(res[1])
        return CC,list_nn2

def cluster_lev(lev,tree,y):
    list_nn=nodes[y].Id.apply(lambda x: True if (str(x)[:4]=='lev_') else False)
    real_nodes=nodes[y].Id.loc[~list_nn]
    N=len(real_nodes)
    list_nn=(nodes[y].loc[list_nn])['Id']
    list_nn_s=list_nn.apply(lambda x: True if float(x[4:8])>lev else False)
    list_nn=list_nn[list_nn_s]
    list_nn=np.sort(list_nn)
    C={}
    heads={}
    k=0
    Clusters=pd.DataFrame(np.zeros((N,N)),index=real_nodes,columns=real_nodes)
    while len(list_nn)>0:
        n=list_nn[0]
        C[k],list_nn_taken=cluster_at(n,y,tree)
        heads[k]=n
        list_nn=np.setdiff1d(list_nn[1:], list_nn_taken)
        Clusters.loc[C[k],C[k]]=1
        k+=1
    return Clusters,C,heads


def compare_clustering(tree,lev,start=1995,end=2000):
    Clusters={}
    C={}
    heads={}
    nodes_total=[]
    for y in np.arange(start, end):
        Clusters[y],C[y],heads[y]=cluster_lev(lev,tree,y)
        nodes_total+=[c for c in Clusters[y].index]
    nodes_total=list(set(nodes_total))
    for y in np.arange(start, end):
        diff=np.setdiff1d(nodes_total,[c for c in Clusters[y].index])
        Clusters[y]=pd.concat([Clusters[y],\
                              pd.DataFrame(np.zeros((Clusters[y].shape[0],len(diff))),\
                              index=Clusters[y].index,columns=diff)],1)
        Clusters[y]=pd.concat([Clusters[y],\
                              pd.DataFrame(np.zeros((len(diff),Clusters[y].shape[1])),\
                              columns=Clusters[y].columns,index=diff) ])                  
    dist=pd.DataFrame(np.zeros((len(Clusters),len(Clusters))),index=np.arange(start, end),columns=np.arange(start, end))
    for y in np.arange(start, end-1):
        print y
        for yy in np.arange(y, end):
            print yy
            temp=(Clusters[yy].add(Clusters[y],fill_value=0)).applymap(lambda x: 1 if x!=0 else 0)
            dist.loc[y,yy]=np.linalg.norm(Clusters[yy]-Clusters[y])/np.linalg.norm(temp)
    return dist+dist.T,Clusters,heads,C


for lev in [0.1,0.3,0.4,0.5,0.7,0.8,0.9,0.05]:
    print lev
    dist,Clusters,heads,c=compare_clustering(tree,lev,start=1995,end=2016)
    dist.to_csv('/Users/cdonnat/Dropbox/data/dist2_'+str(lev)+'.csv')
    #pickle.dump(Clusters,open('/Users/cdonnat/Dropbox/data/clusters2_'+str(lev)+'.pkl'))
    pickle.dump(c,open('/Users/cdonnat/Dropbox/data/C2_'+str(lev)+'.pkl','wb'))
    pickle.dump(heads,open('/Users/cdonnat/Dropbox/data/heads2_'+str(lev)+'.pkl','wb'))
