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


dataset['approx_cost(for two people)'].astype(str).apply(remove_comma)


# In[ ]:




