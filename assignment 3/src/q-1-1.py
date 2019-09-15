#!/usr/bin/env python
# coding: utf-8

# # import Statements

# In[84]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')

import random
from pprint import pprint


# In[85]:


df = pd.read_csv("intrusion_detection/data.csv")
# df = df.rename(columns={"left": "label"})


# In[86]:


# df.head()


# # Training and Validation data split

# In[2]:


def train_validate_split(df,test_size):
    if isinstance(test_size,float):
        test_size = round(test_size*len(df))
        
    indices=df.index.tolist()
    validate_indices = random.sample(population=indices,k=test_size)
    
    validate_df=df.loc[validate_indices]
    train_df=df.drop(validate_indices)
    return train_df,validate_df


# In[90]:


# random.seed(0)
# train_df,validate_df=train_validate_split(df,test_size=0.2)


# In[136]:


data = df.values
data=data[:,:-1]
label=df.values
label=label[:,-1]
# validate_df.head()
name_list=list(df)

# print(data.shape,label.shape)

mean_list=[]
new_values=data.transpose()
# print(new_values.shape,data.shape)
# for x in new_values:
#     mn=np.mean(x,axis=0)
#     mean_list.append(mn)

# print(mean_list)
X = df.loc[:,'duration':'dst_host_srv_rerror_rate']

# for i in range(len(data)):
X.loc[:,'duration':'dst_host_srv_rerror_rate']=(X.loc[:,'duration':'dst_host_srv_rerror_rate'] - X.loc[:,'duration':'dst_host_srv_rerror_rate'].mean())/X.loc[:,'duration':'dst_host_srv_rerror_rate'].std()
    
# print(X.shape)
# Q=X.transpose()
# print(Q.shape)
# Q=np.matmul(X,Q)
# print(Q.shape)
covarience=np.cov(np.array(X.T))
# print(covarience.shape)

eigen_values,eigen_vectors=np.linalg.eig(covarience)
# print(eigen_values)

max_val=[]
for i in range(len(eigen_values)):
    temp=eigen_values[i]/sum(eigen_values)
    max_val.append((temp,i))
    
max_val=sorted(max_val)
max_val.reverse()
# print(max_val)
newdata_list=[]
temp=0

for x in max_val:
    if temp >= (.9):
        break
    temp+=x[0]
    newdata_list.append(x[1])

# print(type(data),type(new_data))
new_data=eigen_vectors[:,newdata_list]    
# print()
new_data=np.dot(data,new_data)
print(new_data)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




