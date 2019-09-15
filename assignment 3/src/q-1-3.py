#!/usr/bin/env python
# coding: utf-8

# # import Statements

# In[19]:


import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')

import random
from pprint import pprint


# In[20]:


df = pd.read_csv("intrusion_detection/data.csv")
# df = df.rename(columns={"left": "label"})


# # Training and Validation data split

# In[21]:


def train_validate_split(df,test_size):
    if isinstance(test_size,float):
        test_size = round(test_size*len(df))
        
    indices=df.index.tolist()
    validate_indices = random.sample(population=indices,k=test_size)
    
    validate_df=df.loc[validate_indices]
    train_df=df.drop(validate_indices)
    return train_df,validate_df


# In[22]:


# random.seed(0)
# train_df,validate_df=train_validate_split(df,test_size=0.2)


# In[23]:


data = df.values
data=data[:,:-1]
label=df.values
label=label[:,-1]
# validate_df.head()
name_list=list(df)

# print(data.shape,label.shape)

mean_list=[]
new_values=data.transpose()

X = df.loc[:,'duration':'dst_host_srv_rerror_rate']

X.loc[:,'duration':'dst_host_srv_rerror_rate']=(X.loc[:,'duration':'dst_host_srv_rerror_rate'] - X.loc[:,'duration':'dst_host_srv_rerror_rate'].mean())/X.loc[:,'duration':'dst_host_srv_rerror_rate'].std()

covarience=np.cov(np.array(X.T))
# print(covarience.shape)

eigen_values,eigen_vectors=np.linalg.eig(covarience)
# print(eigen_values)

max_val=[]
for i in range(len(eigen_values)):
    temp=eigen_values[i]/sum(eigen_values)
    max_val.append((temp,i))
    
# print(max_val)
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

# print(newdata_list)
# print(type(data),type(new_data))
new_data=eigen_vectors[:,newdata_list]    
# print(new_data)
new_data=np.dot(X,new_data)
# print(new_data)


# In[24]:


gmm = GaussianMixture(n_components=5)
array=gmm.fit_predict(new_data)

data2=df.values
list1=[]
list2=[]
list3=[]
list4=[]
list5=[]

for x in range(len(array)):
    if array[x]==0:
        list1.append(data2[x])
    elif array[x]==1:
        list2.append(data2[x])
    elif array[x]==2:
        list3.append(data2[x])
    elif array[x]==3:
        list4.append(data2[x])
    elif array[x]==4:
        list5.append(data2[x])
        
y=np.array(list1)
uniq,count=np.unique(y[:,-1],return_counts=True)
catagory1=uniq[np.argmax(count)]
accuracy1=max(count)/np.sum(count)
print(catagory1,accuracy1)

y=np.array(list2)
uniq,count=np.unique(y[:,-1],return_counts=True)
catagory2=uniq[np.argmax(count)]
accuracy2=max(count)/np.sum(count)
print(catagory2,accuracy2)

y=np.array(list3)
uniq,count=np.unique(y[:,-1],return_counts=True)
catagory3=uniq[np.argmax(count)]
accuracy3=max(count)/np.sum(count)
print(catagory3,accuracy3)

y=np.array(list4)
uniq,count=np.unique(y[:,-1],return_counts=True)
catagory4=uniq[np.argmax(count)]
accuracy4=max(count)/np.sum(count)
print(catagory4,accuracy4)

y=np.array(list5)
uniq,count=np.unique(y[:,-1],return_counts=True)
catagory5=uniq[np.argmax(count)]
accuracy5=max(count)/np.sum(count)
print(catagory5,accuracy5)

final_acc=(accuracy1+accuracy2+accuracy3+accuracy4+accuracy5)/5

print("final_acc  : ",final_acc )

