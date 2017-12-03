# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 00:54:13 2017

@author: cdonnat
"""
R2={}
for l in range(1,40,2):
    R2[l]=evaluate_autocorrelation(stock_returns,N,l)
mapsector={np.unique(CompanyInfo.loc["Sector"])[u]:u for u in range(len(np.unique(CompanyInfo.loc["Sector"])))}
mapsector={np.unique(CompanyInfo.loc["Industry"])[u]:u for u in range(len(np.unique(CompanyInfo.loc["Industry"])))}
 mapsector=pd.DataFrame.from_dict({np.unique(CompanyInfo.loc["Sector"])[u]:u for u in range(len(np.unique(CompanyInfo.loc["Sector"])))},orient='index');mapsector.columns=['keys']
mapind=pd.DataFrame.from_dict({np.unique(CompanyInfo.loc["Industry"])[u]:u for u in range(len(np.unique(CompanyInfo.loc["Industry"])))},orient='index');mapind.columns=['keys']

CompanyInfo=(CompanyInfo.T).merge(mapsector,right_index=True,left_on='Sector')
CompanyInfo=(CompanyInfo).merge(mapind,right_index=True,left_on='Industry')
#paths=mapind.copy()
#paths.columns=['child']
#paths['parent']=np.zeros(paths.shape[0])
paths=[]
for i in mapind['keys']:
    l=CompanyInfo.loc[CompanyInfo['keys_y']==i,'keys_x']
    print l
    for j in np.unique(l):
        w=np.sum(l==j)*1.0/len(l)
        paths+=[[i,j,w]]
paths=pd.DataFrame(paths,columns=['child','parent','weight'])
path='/Users/cdonnat/Dropbox/Financial Networks/data/'
paths.to_csv(path+'path.csv')
    #paths['parent'][i]=CompanyInfo.loc[CompanyInfo['keys_y']==i,'keys_x'][0]
    