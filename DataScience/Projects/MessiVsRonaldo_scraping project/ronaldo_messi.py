#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
ron_url='https://en.wikipedia.org/wiki/Cristiano_Ronaldo'
ronaldo=pd.read_html(ron_url,na_values='—')


# In[2]:


ronaldo_goals=ronaldo[5].copy()
ronaldo_goals


# In[3]:


mes_url='https://en.wikipedia.org/wiki/Lionel_Messi'

messi=pd.read_html(mes_url,na_values='—')


# In[5]:


messi_goals=messi[4].copy()
messi_goals


# In[6]:


ronaldo_goals.columns=[col[1] for col in ronaldo_goals.columns]
messi_goals.columns=[col[1] for col in messi_goals.columns]


# In[7]:


ronaldo_goals.drop('Apps',axis=1,inplace=True)
messi_goals.drop('Apps',axis=1,inplace=True)

ronaldo_goals['Season']=ronaldo_goals['Season'].str.replace(r'\[.*]','')
messi_goals['Season']=messi_goals['Season'].str.replace(r'\[.*]','')


# In[8]:


ronaldo_goals=ronaldo_goals.set_index('Season').drop('Total').drop('Career total')
messi_goals=messi_goals.set_index('Season').drop('Total').drop('Career total')

ronaldo_goals=ronaldo_goals.iloc[:,-1]
messi_goals=messi_goals.iloc[:,-1]


# In[9]:


ronaldo_goals=ronaldo_goals.groupby(ronaldo_goals.index).sum()
ronaldo_goals=ronaldo_goals.iloc[1:]


# In[21]:


messi_goals=messi_goals.groupby(messi_goals.index).sum()


# In[27]:


plt.style.available


# In[28]:


plt.style.use('seaborn-colorblind')


# In[41]:


# plt.figure(figsize=(14,10))
# ronaldo_goals.plot(label='Cristiano Ronaldo')
# messi_goals.plot(color='#d31a63',label='Lionel Messi')
# plt.legend()
# plt.ylabel('Goals',fontsize=12)
# plt.title('Club Career goals of Messi and Ronaldo (2003/04 - 2019/20) seasons',fontsize=15)


# In[44]:


plt.figure(figsize=(18,10))
sns.lineplot(ronaldo_goals.index,ronaldo_goals,label='Cristiano Ronaldo')
sns.lineplot(messi_goals.index,messi_goals,label='Lionel Messi')
plt.legend()
plt.ylabel('Goals',fontsize=12)
plt.title('Club Career goals of Messi and Ronaldo (2003/04 - 2019/20) seasons',fontsize=15)


# In[15]:


plt.show()

