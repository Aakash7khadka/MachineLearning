
# coding: utf-8

# # Assignment 2
# 
# Before working on this assignment please read these instructions fully. In the submission area, you will notice that you can click the link to **Preview the Grading** for each step of the assignment. This is the criteria that will be used for peer grading. Please familiarize yourself with the criteria before beginning the assignment.
# 
# An NOAA dataset has been stored in the file `data/C2A2_data/BinnedCsvs_d400/fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89.csv`. This is the dataset to use for this assignment. Note: The data for this assignment comes from a subset of The National Centers for Environmental Information (NCEI) [Daily Global Historical Climatology Network](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/readme.txt) (GHCN-Daily). The GHCN-Daily is comprised of daily climate records from thousands of land surface stations across the globe.
# 
# Each row in the assignment datafile corresponds to a single observation.
# 
# The following variables are provided to you:
# 
# * **id** : station identification code
# * **date** : date in YYYY-MM-DD format (e.g. 2012-01-24 = January 24, 2012)
# * **element** : indicator of element type
#     * TMAX : Maximum temperature (tenths of degrees C)
#     * TMIN : Minimum temperature (tenths of degrees C)
# * **value** : data value for element (tenths of degrees C)
# 
# For this assignment, you must:
# 
# 1. Read the documentation and familiarize yourself with the dataset, then write some python code which returns a line graph of the record high and record low temperatures by day of the year over the period 2005-2014. The area between the record high and record low temperatures for each day should be shaded.
# 2. Overlay a scatter of the 2015 data for any points (highs and lows) for which the ten year record (2005-2014) record high or record low was broken in 2015.
# 3. Watch out for leap days (i.e. February 29th), it is reasonable to remove these points from the dataset for the purpose of this visualization.
# 4. Make the visual nice! Leverage principles from the first module in this course when developing your solution. Consider issues such as legends, labels, and chart junk.
# 
# The data you have been given is near **Ann Arbor, Michigan, United States**, and the stations the data comes from are shown on the map below.

# In[1]:

import matplotlib.pyplot as plt
import mplleaflet
import pandas as pd

def leaflet_plot_stations(binsize, hashid):

    df = pd.read_csv('data/C2A2_data/BinSize_d{}.csv'.format(binsize))

    station_locations_by_hash = df[df['hash'] == hashid]

    lons = station_locations_by_hash['LONGITUDE'].tolist()
    lats = station_locations_by_hash['LATITUDE'].tolist()

    plt.figure(figsize=(8,8))

    plt.scatter(lons, lats, c='r', alpha=0.7, s=200)

    return mplleaflet.display()

leaflet_plot_stations(400,'fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89')


# In[61]:

import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('data/C2A2_data/BinnedCsvs_d400/fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89.csv',parse_dates=['Date'])
data.set_index('Date',inplace=True)
data.sort_index(inplace=True)
data=data[~((data.index.month==2)&(data.index.day==29))]
data


# In[62]:

data['Data_Value']=data['Data_Value']/10
data2015=data.loc['2015-01-01':].copy()
data=data.loc[:'2014-12-31'].copy()
data.reset_index(inplace=True)
data['Month']=data['Date'].dt.month
data['Day']=data['Date'].dt.day

data['Month-Day']=data['Date'].dt.strftime('%m-%d')


mini=data[data.Element=='TMIN'].copy()
maxi=data[data.Element=='TMAX'].copy()


# In[47]:

mini.tail()


# In[63]:


mini=mini.groupby('Month-Day').Data_Value.agg({'TEMP':min}).reset_index()
maxi=maxi.groupby('Month-Day').Data_Value.agg({'TEMP':max}).reset_index()
mini.head()


# In[49]:

record_max=maxi.TEMP.max()
record_min=mini.TEMP.min()
print(record_max,record_min)


# In[93]:

# data2015.reset_index(inplace=True)
ma=data2015[data2015['Element']=='TMAX'].reset_index()
mi=data2015[data2015['Element']=='TMIN'].reset_index()
ma.head()


# In[94]:

ma=ma.groupby('Date')['Data_Value'].agg({'MAX':max}).reset_index()
mi=mi.groupby('Date')['Data_Value'].agg({'MIN':min}).reset_index()


# In[102]:

scatt_min=mi['MIN']<mini['TEMP']

scatt_max=ma['MAX']>maxi['TEMP']
scatt_max=ma[scatt_max]
scatt_min=mi[scatt_min]


# In[105]:

scatt_max.head()


# In[117]:

import numpy as np
plt.figure(figsize=(14,10))
y1=mini['TEMP']
y2=maxi['TEMP']
plt.plot(y1,color='#ADD8E6',label='Minimum Temperature')
plt.plot(y2,label='Maximum Temperature')
plt.legend()
plt.plot(scatt_max.index,scatt_max['MAX'],'.',color='red')
plt.plot(scatt_min.index,scatt_min['MIN'],'.',color='red')

tick=['Jan','Feb','Mar','Apr','May','June','July','Aug','Sep','Oct','Nov','Dec']
plt.fill_between(x,mini['TEMP'],maxi['TEMP'],color='grey',alpha=0.2)
plt.xticks(np.arange(0,365+30.4166666667,32.9513888889),tick)
plt.ylabel('Temperature in Celsius',fontsize=15)
plt.title(' Days in 2015 that broke a record high or low for 2005-2014',fontsize=15)
plt.show()


# In[127]:


plt.savefig('Downloads\hello.png')

