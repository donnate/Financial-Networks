# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 22:13:18 2017

@author: cdonnat
"""
import numpy as np
import pandas as pd

def sector_consistence(betas,cor_res,nb_sectors=14):
  
    sector_consistency=pd.DataFrame(np.zeros((betas.shape[0],3)),index=betas.index)
    sector_consistency.columns=['diversity','consistency','degree']
    for s in betas.index:
        neighbors=list(cor_res.index[np.where(cor_res[s]!=0)])
        neighbors.remove(s)
        if len(neighbors)>0:
            neighbor_sectors=betas['Sector'][neighbors]
            sector_consistency.loc[s,'degree']=len(neighbors)
            sector_consistency.loc[s,'diversity']=len(np.unique(neighbor_sectors))*1.0/nb_sectors
            sector_consistency.loc[s,'consistency']=np.sum(neighbor_sectors==betas['Sector'][s])*1.0/len(neighbors)
        
    return sector_consistency