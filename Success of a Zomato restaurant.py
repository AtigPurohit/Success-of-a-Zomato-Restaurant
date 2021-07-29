#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dataset = pd.read_csv("C:\\Users\\itsth\\Documents\\MachineLearning\\Data\\zomato.csv")


# In[3]:


dataset.head()


# In[4]:


dataset.dtypes


# In[5]:


dataset.shape


# In[6]:


dataset.isnull().sum()


# In[7]:


features_na = []
for features in dataset.columns:
    if(dataset[features].isnull().sum()>1):
        features_na.append(features)
features_na


# In[8]:


for features in dataset.columns:
    if(dataset[features].isnull().sum()>1):
        p = dataset[features].isnull().sum()/len(dataset)*100
        p=np.round(p)
        print("{} has {}% missing values".format(features,p))


# In[9]:


dataset.info()


# In[10]:


dataset['approx_cost(for two people)'].dtype


# In[11]:


dataset['approx_cost(for two people)'].isnull()


# In[12]:


dataset[dataset['approx_cost(for two people)'].isnull()]


# In[13]:


dataset['approx_cost(for two people)'].unique()


# In[14]:


dataset['approx_cost(for two people)'].astype(str).apply(lambda x:x.replace(',',''))


# In[15]:


def remove_comma(x):
    return x.replace(',','')


# In[16]:


dataset['approx_cost(for two people)'] = dataset['approx_cost(for two people)'].astype(str).apply(remove_comma)


# In[17]:


dataset['approx_cost(for two people)']= dataset['approx_cost(for two people)'].astype(float)


# In[18]:


dataset['approx_cost(for two people)'].dtype


# In[19]:


dataset['rate'].dtype


# In[20]:


dataset['rate'].unique()


# In[21]:


def split(x):
    return x.split('/')[0]


# In[22]:


dataset['rate'] = dataset['rate'].astype(str).apply(split)


# In[23]:


dataset['rate'].replace('-',0,inplace=True)
dataset['rate'].replace('NEW',0,inplace=True)


# In[24]:


dataset['rate'] = dataset['rate'].astype(float)


# In[25]:


dataset['rate'].dtype


# In[26]:


plt.figure(figsize=(20,12))
dataset['rest_type'].value_counts().nlargest(20).plot.bar(color='red')


# In[27]:


def mark(x):
    if x in ('Quick Bites', 'Casual Dining'):
        return "Quick Bites/Casual Dining"
    else:
        return "Others"


# In[28]:


dataset['Top_types']=dataset['rest_type'].apply(mark)


# In[29]:


dataset.head()


# In[30]:


pip install plotly


# In[31]:


import plotly.express as px


# In[32]:


values = dataset['Top_types'].value_counts().values


# In[33]:


labels = dataset['Top_types'].value_counts().index


# In[34]:


fig = px.pie(dataset,names=labels,values=values)
fig.show()


# In[35]:


dataset.columns


# In[36]:


dataset.dtypes


# In[37]:


rest = dataset.groupby('name').agg({'votes':'sum','url':'count','approx_cost(for two people)':'mean','rate':'mean'}).reset_index()


# In[38]:


rest


# In[39]:


rest.columns = ['name', 'total_votes', 'total_unities','avg_approx_cost','mean_rating']


# In[40]:


rest.head()


# In[42]:


rest['votes_per_unity'] = rest['total_votes']/rest['total_unities']


# In[43]:


rest.head()


# In[44]:


popular = rest.sort_values(by='total_unities',ascending=False)
popular


# In[46]:


popular.shape


# In[48]:


popular['name'].nunique()


# In[63]:


import seaborn as sns
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(20,15))

sns.barplot(x='total_votes',y='name',data = popular.sort_values(by='total_votes',ascending=False).query('total_votes>0').head(5),ax=ax1,palette="plasma")
ax1.set_title('Top 5 most voted restaurants')

sns.barplot(x='total_votes',y='name',data = popular.sort_values(by='total_votes',ascending=False).query('total_votes>0').tail(5),ax=ax2,palette="plasma")
ax2.set_title('Least 5 most voted restaurants')


# In[ ]:




