# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 08:17:09 2017

@author: cdonnat
"""

returns_diff=stock_data.diff(1)
returns_diff=returns_diff.fillna(0)

### cap returns:
cap_up=data_log.mean(0)+2*stock_data.std(0)
cap_down=stock_data.mean(0)-2*stock_data.std(0)
stock_data=stock_data.apply(lambda x: np.max(cap_down[x.name],x) )
stock_data2=stock_data.fillna(0)
stock_data2=stock_data2.apply(lambda x: (x-cap_down[x.name])*(x>cap_down[x.name])+cap_down[x.name],axis=0)
stock_data2=stock_data2.apply(lambda x: (x-cap_up[x.name])*(x<cap_up[x.name])+cap_up[x.name],axis=0)




plt.figure(1)
for i in range(1,10):
    for j in range(i):
        plt.figure()
        plt.scatter(np.sqrt(1+data_log.iloc[:,i]),np.sqrt(1+data_log.iloc[:,j]),c=data_log.iloc[:,200],cmap='hot')
        plt.show()
plt.subplot(211)
plt.plot(t, s1)
plt.subplot(212)
plt.plot(t, 2*s1)


