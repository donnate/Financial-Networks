# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 06:44:47 2017

@author: cdonnat
"""
import networkx as nx
import numpy as np
import pandas as pd


def agg_clustering(cor_res,weights,stock_residuals,stepsize=0.01, verbose=True):
    stock_bis_res=stock_residuals.copy()
    x=np.linspace(0,2.0,int(2.0/stepsize))
    np.fill_diagonal(cor_res.as_matrix(),0)
    tree={}
    print(cor_res.iloc[0,0])
    count={i:[] for i in cor_res.columns}
    weights2=weights.copy()
    cor_res_new=cor_res.copy()
    
    if verbose: print cor_res_new.shape
    for xx in x:
        if verbose: print(1-xx)
        ind=np.where(cor_res_new>1-xx)
        print ind
        tree[1-xx]={}
        it_tree=0
        to_delete=[]
        mapping={}
        for it in range(len(ind[0])):
            i=ind[0][it]
            j=ind[1][it]
            ii=cor_res_new.columns[ind[0][it]]
            jj=cor_res_new.columns[ind[1][it]]
            if i>j:
                pass
            else:
                if len(count[ii])==0 and len(count[jj])==0:
                    tree[1-xx][it_tree]=[ii,jj]
                    count[ii]=[((1-xx),it_tree)]
                    count[jj]=[((1-xx),it_tree)]
                    mapping[it_tree]=it_tree
                    it_tree+=1
                    
                else:
                    #print (len(count[ii]),len(count[jj]))
                    if len(count[ii])>0 and len(count[jj])==0:
                        #try:
                            u=mapping[count[ii][0][1]]
                            tree[1-xx][u].append(jj)
                            count[jj].append(((1-xx),count[ii][0][1]))
                            #tt_tree=count[ii][0][1]
                        #except:
                            #tree[1-xx][it_tree].append(jj)
                            
                            #it_tree+=1
                    elif len(count[ii])==0 and len(count[jj])>0:
                        #try:
                            v=mapping[count[jj][0][1]]
                            tree[1-xx][v].append(ii)
                        #except:
                            #tree[1-xx][it_tree].append(ii)
                            #it_tree+=1
                            count[ii].append(((1-xx),count[jj][0][1]))
                    else:
                        u=mapping[count[ii][0][1]]
                        v=mapping[count[jj][0][1]]
                        tree[1-xx][u]+=tree[1-xx][v]
                        tree[1-xx][u]=list(set(tree[1-xx][u]))
                        to_delete.append(tree[1-xx][v])
                        if u!=v:
                            mapping[v]=u
                            for key,val in mapping.iteritems():
                                if val==v:
                                    mapping[key]=u
                            del tree[1-xx][v]
                    
        ## merge the clusters
                    
                    
        for k in tree[1-xx].keys():
            count.update({'lev_'+str(1-xx)[:4]+'_'+str(k):[]})
            stock_bis_res['lev_'+str(1-xx)[:4]+'_'+str(k)]=pd.DataFrame(1.0/float(weights.loc[tree[1-xx][k]].sum())*(stock_bis_res[tree[1-xx][k]]).as_matrix().dot(weights.loc[tree[1-xx][k]]),index=stock_bis_res.index)
            weights.loc['lev_'+str(1-xx)[:4]+'_'+str(k)]=float((weights.loc[tree[1-xx][k]]).sum())
            weights2.loc['lev_'+str(1-xx)[:4]+'_'+str(k)]=float((weights.loc[tree[1-xx][k]]).sum())
            if verbose: print 'lev_'+str(1-xx)[:4]+'_'+str(k),float((weights.loc[tree[1-xx][k]]).sum())
            stock_bis_res=stock_bis_res.drop(tree[1-xx][k],axis=1)
            
            weights.loc[tree[1-xx][k]]=0
        if verbose: print(np.sum(weights))
        weights/=float(np.sum(weights))
        cor_res_new=stock_bis_res.corr()
        if verbose: print cor_res_new.shape
        np.fill_diagonal(cor_res_new.as_matrix(),0)
    return tree,count,weights2

def edges_tree(tree,CompanyInfo0,weights,name_edges='edges',name_nodes='nodes',path='/Users/cdonnat/Dropbox/data/'):
    edges=[]
    CompanyInfo=CompanyInfo0[['Sector','Industry','Sector_key','Industry_key']]
    for xx in tree.keys():
        for grp_key in tree[xx].keys():
            for v in tree[xx][grp_key]:
                w=float(weights.loc[v]/weights.loc['lev_'+str(xx)[:4]+'_'+str(grp_key)])
                edges+=[['lev_'+str(xx)[:4]+'_'+str(grp_key),v,float(w+(1-w)*xx)]]
            for u in range(1,len(tree[xx][grp_key])):
                for v in range(u):
                    edges+=[[tree[xx][grp_key][u],tree[xx][grp_key][v],xx]]
    edges=pd.DataFrame(np.array(edges)[1:,:],columns=['Source','Target','weight'])
  #  edges['Target']=edges.apply(lambda x: 'lev_'+x['Target'][5:10] if (x['Target'][:5]=='level') else x['Target'],1 )    
   # edges['Source']=edges['Source'].apply(lambda x: 'lev_'+x[5:10] if (x[:5]=='level') else x )    
    edges.to_csv(path+name_edges+'.csv')
    nodes=pd.DataFrame(np.array(list(set(list(np.unique(edges['Source']))+list(np.unique(edges['Target']))))))
    nodes.columns=['Id']
    nodes['index']=range(nodes.shape[0])
    nodes['category']=nodes['Id'].apply(lambda x: float(x[4:7]) if (x[:3]=='lev') else 2.0 )
    nodes=nodes.merge(CompanyInfo,right_index=True,left_on='Id',how='left')
    nodes.columns=['Id', 'index', 'category', 'Sector', 'Industry', 'keys_s',
       'keys_ind']
    nodes['keys_s'][np.isnan(nodes['keys_s'])]=20
    nodes['keys_s'][(nodes['Sector']=='n/a')]=20
    nodes['keys_ind'][np.isnan(nodes['keys_ind'])]=500
    nodes['keys_ind'][(nodes['Industry']=='n/a')]=500
    #nodes['Id']=nodes.apply(lambda x: 'lev_'+(x[6:10]) if (x[:5]=='level') else x)
    nodes['keys_s']=nodes.apply(lambda x: float(x['Id'][6:10]) if (x['Id'][:5]=='level') else x['keys_s'],1 )
    nodes['keys_ind']=nodes.apply(lambda x: float(x['Id'][6:10]) if (x['Id'][:5]=='level') else x['keys_ind'],1 )    
    nodes.to_csv(path+name_nodes+'.csv')
    nodes.index=nodes['Id']
    return edges,nodes
    
    
def bin_values(test):
    nodes['keys_s2']=nodes['keys_s'].copy()
    for xx in range(0,1,1.0/stepsize):
        for grp_key in tree[1-xx].keys():
                grp=tree[1-xx][grp_key]
                test=nodes.apply(lambda x :  x['keys_s'] if (x['Id'] in list(grp)) else -1, axis=1)
                count={v:0 for v in np.unique(test)}
                for v in np.unique(test):
                    count[v]=np.sum(test==v)
                try:
                    del count[-1.0]
                except:
                    pass
                count=pd.DataFrame.from_dict(count,orient='index')
                count=count.sort_values(by=0,ascending=False)

                
    return true

def investigate_graph(tree,edges,x=[]):
    plt.figure()
    if len(x)==0:
        x=np.linspace(0,1,100)
    plt.scatter( [xx for xx in tree.keys()], [np.mean([len(g) for g in tree[xx].values()]) for xx in tree.keys()],label='avg size grouping',c='blue')
    plt.scatter( [xx for xx in tree.keys()],[len(tree[xx]) for xx in tree.keys()],c='black',label='nb groups added')
    plt.xlabel('Correlation Threshold')
    plt.legend(loc='upper right')
    plt.show()
    G=nx.from_pandas_dataframe(edges,source='Source',target='Target')
    print 'nb connected components:',nx.number_connected_components(G)
    for u in nx.connected_component_subgraphs(G):
        print 'size cmpt: \t'+str(u.number_of_edges())+'\n'
    i=0
    for u in nx.connected_component_subgraphs(G):
        i+=1
        if i>1:
            print 'cmpt'+str(i)+': \t'+u.nodes()+'\n'
    industry_diversity={}
    for xx in tree.keys():
        industry_diversity[xx]={}
        for grp_key in tree[xx].keys():
            grp=tree[xx][grp_key]
            industry_diversity[xx][grp_key]=[np.sum(nodes.loc[tree[xx][grp_key],'keys_s']==ind) for ind in np.unique(nodes['keys_s'])]
    div=[]    
    for xx in industry_diversity.keys():
        for jj in industry_diversity[xx].keys():
            div.append(industry_diversity[xx][jj]+[xx])
    nodes['keys_s2']=nodes['keys_s'].copy()
    nodes_bis=[]
    for xx in np.sort(list(tree.keys()))[::-1]:
        for grp_key in tree[xx].keys():
            nd='lev_'+str(1-xx)[:4]+'_'+str(grp_key)
            ind=pd.DataFrame([np.sum(nodes.loc[tree[xx][grp_key],'keys_s']==ind) for ind in np.unique(nodes['keys_s'])],index=np.unique(nodes['keys_s']),columns=['c'])            
            ind=ind.sort_values(by='c',ascending=False)
            if (ind.iloc[1]==0).bool() &(ind.index[0]==20 ):
                nodes.loc[nd,'keys_s2']=20
            elif (ind.iloc[1]>0).bool() &(ind.index[0]==20 ):
                if (ind.iloc[2]==ind.iloc[1]).bool():
                    nodes_bis.append((nd,ind))
                nodes.loc[nd,'keys_s2']=float(ind.index[1])
            else:
                if (ind.iloc[0]==ind.iloc[1]).bool():
                    nodes_bis.append((nd,ind))
                nodes.loc[nd,'keys_s2']=float(ind.index[0])
                    
                
            
            
    div=pd.DataFrame(np.array(div),columns= [nodes.loc[nodes['keys_s']==j,'Sector'][0] for j in np.unique(nodes['keys_s'])]+['corr'])
    nb_activated=div.apply(lambda x: np.sum([xx!=0 for xx in x[:-2]])/np.sum([xx!=0 for xx in x[:-2]]),axis=1)
    nb_activated2=div.iloc[:,:-1]    
    nb_activated2=nb_activated2.apply(lambda x: x/np.sum(x), axis=1)
    plt.hist(nb_activated2.iloc[:,12])
    for u in nb_activated2.columns:
            plt.figure()
            plt.hist(nb_activated2[u])
            plt.title(u)
            plt.show()
    
    
    nb_activated3=div.iloc[:,:-2]    
    nb_activated3=nb_activated3.apply(lambda x: x/np.sum(x), axis=1)
    #plt.hist(nb_activated3.iloc[:,12])
    for u in nb_activated3.columns:
            plt.figure()
            plt.hist(nb_activated3[u])
            plt.title(u)
            plt.show()
            
            
 
 
        