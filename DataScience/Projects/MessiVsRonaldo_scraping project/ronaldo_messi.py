#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
ron_url='https://en.wikipedia.org/wiki/Cristiano_Ronaldo'
ronaldo=pd.read_html(ron_url,na_values='—')
# print(ronaldo[7])

# In[2]:

#determine the table where the goals are situated i.e 7 currently
ronaldo=ronaldo[7].copy()


# In[3]:


mes_url='https://en.wikipedia.org/wiki/Lionel_Messi'

messi=pd.read_html(mes_url,na_values='—')


# In[5]:


messi=messi[5].copy()


# In[6]:


ronaldo.columns=[col[1] for col in ronaldo.columns]
messi.columns=[col[1] for col in messi.columns]


# In[7]:


print(ronaldo)

ronaldo['Season']=ronaldo['Season'].str.replace(r'\[.*]','', regex= True)
messi['Season']=messi['Season'].str.replace(r'\[.*\]','', regex= True)

# 2004/05 to 2021/22[When both were playing for a full season in European Top 5 league]
actual_range = [str(x)+'–'+str(x+1)[-2:] for x in range(2004, 2022)]
print(actual_range)

ronaldo = ronaldo[ronaldo['Season'].isin(actual_range)]
messi = messi[messi['Season'].isin(actual_range)]



# In[8]:


ronaldo=ronaldo.set_index('Season')
messi=messi.set_index('Season')


ronaldo=ronaldo.iloc[:,-2:]
messi=messi.iloc[:,-2:]

# print(ronaldo)
# In[9]:

# print(ronaldo_goals)
# combining the multiple rows for same season
ronaldo=ronaldo.groupby(ronaldo.index).sum()
print(ronaldo)
# ronaldo_goals=ronaldo_goals.iloc[1:]
# print(ronaldo_goals)

# In[21]:


# print(messi_goals.replace(r'\[\d+\]','',regex=True))
# messi_goals = messi_goals.replace(r'\[\d+\]','',regex=True)
messi=messi.groupby(messi.index).sum()

print(messi)


# print(messi_goals.reindex(actual_range))

# In[27]:


print(plt.style.available)


# In[28]:


plt.style.use('seaborn-v0_8-colorblind')


# ronaldo_goals.name = 'ronaldo_goals'
# messi_goals.name = 'messi_goals'
# In[44]:

plt.figure(figsize=(18,10))

sns.lineplot(x=ronaldo.index,y=ronaldo['Goals'],label='Cristiano Ronaldo', dashes=(2,2))
sns.lineplot(x=messi.index,y=messi['Goals'],label='Lionel Messi')
plt.xticks(rotation=45)
plt.legend()
plt.ylabel('Goals',fontsize=12)
plt.title('Club Career goals of Messi and Ronaldo (2003/04 - 2021/22) seasons',fontsize=15)


# In[15]:


plt.show()
ronaldo.columns = ['ronaldo_apps', 'ronaldo_goals']
messi.columns = ['messi_apps', 'messi_goals']
print(ronaldo)
combined_df = pd.concat([ronaldo, messi], axis=1)
print(combined_df)
