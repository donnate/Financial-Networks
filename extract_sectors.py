# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 19:17:00 2017

@author: cdonnat
"""

import numpy as np
import pandas as pd
import csv 


path='/Users/cdonnat/Dropbox/Financial Networks/data/'

#filename='companylist.csv'
filename='book_CompanyInfo.csv'
def load_sectors(path,filename='book_CompanyInfo.csv'):
    
    CompanyInfo={}
    CompanySector={}
    CompanyIndustry={}
    line=0
    with open(path+filename, 'rb') as f:
        reader = csv.reader(f)
        data= list(reader)
        for i in range(len(data)):
            CompanyInfo[data[i][0]]=[data[i][6],data[i][7]]
            CompanySector[data[i][0]]=[data[i][6]]
            CompanyIndustry[data[i][0]]=[data[i][7]]
    
    CompanySector=pd.DataFrame.from_dict(CompanySector)
    CompanySector.index=['Sector']
    CompanyIndustry=pd.DataFrame.from_dict(CompanyIndustry)
    CompanyIndustry.index=['Industry']
    CompanyInfo=pd.DataFrame.from_dict(CompanyInfo)
    CompanyInfo.index=['Sector','Industry']
    return CompanySector,CompanyIndustry,CompanyInfo



def load_sectors2(path,filename='book_CompanyInfo.csv'):
    CompanyInfo=pd.DataFrame.from_csv(path+filename, sep=",",header=0)
    cols=CompanyInfo.columns.values
    cols[2:5]=['Sector','Industry','Industry_subgroup']
    CompanyInfo.columns=cols
    index_name=CompanyInfo.index
    
    sectors=pd.DataFrame(np.unique(CompanyInfo.Sector),columns=['Sector'])
    sectors['Sector_key']=sectors.index
    CompanyInfo=CompanyInfo.merge(sectors,right_on='Sector',left_on='Sector',how='left')
    
    industry=pd.DataFrame(np.unique(CompanyInfo.Industry),columns=['Industry'])
    industry['Industry_key']=industry.index
    CompanyInfo=CompanyInfo.merge(industry,right_on='Industry',left_on='Industry',how='left')
    
    
    industry_subgroup=pd.DataFrame(np.unique(CompanyInfo.Industry_subgroup),columns=['Industry_subgroup'])
    industry_subgroup['Industry_subgroup_key']=industry_subgroup.index
    CompanyInfo=CompanyInfo.merge(industry_subgroup,right_on='Industry_subgroup',left_on='Industry_subgroup',how='left')
    CompanyInfo.index=index_name
    return CompanyInfo



def deal_with_nans_sectors(CompanyInfo,betas):
    get_nan=CompanyInfo.applymap(lambda x: x=='n/a')
    where_nan=get_nan.sum(0)
    one_miss=list(np.where((where_nan==1))[0])
    two_miss=list(np.where((where_nan==2))[0])
    return CompanyInfo.iloc[:,(where_nan==0)]
#list_Sectors=set([v for (k,v) in CompanySector.iteritems()])
#list_Industries=set([v for (k,v) in CompanyIndustry.iteritems()])
#company_labels={}
#for (k,v) in CompanySector.iteritems():
#    company_labels[k]=[k,v,CompanyIndustry[k]]
#company_labels.to_csv('company_labels.csv)
