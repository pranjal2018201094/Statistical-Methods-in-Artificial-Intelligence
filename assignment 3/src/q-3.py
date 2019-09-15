#!/usr/bin/env python
# coding: utf-8

# In[234]:


import numpy as np
import math
import pandas as pd
from scipy.optimize import fmin_tnc
import matplotlib.pyplot as plt
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')

import random


# In[235]:


df = pd.read_csv("wine-quality/data.csv")
# df.values


# In[236]:


def train_validate_split(df,test_size):
    if isinstance(test_size,float):
        test_size = round(test_size*len(df))
        
    indices=df.index.tolist()
    validate_indices = random.sample(population=indices,k=test_size)
    
    validate_df=df.loc[validate_indices]
    train_df=df.drop(validate_indices)
    return train_df,validate_df


# In[237]:


# random.seed(0)
train_df,validate_df=train_validate_split(df,0.2)


# In[238]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# In[239]:


data=train_df

# print(df.shape)
initial_theta = [0,0]
alpha = 0.1
iterations = 1000

X = data.iloc[:,:-1]
y = data.iloc[:,-1]

X=(X-np.mean(X))/np.std(X)

X_validate=validate_df.iloc[:,:-1]
y_validate=validate_df.values[:,-1]
# X_validate=(X_validate-np.mean(X))/np.std(X)

X_validate=np.c_[np.ones((X_validate.shape[0], 1)), X_validate]

X = np.c_[np.ones((X.shape[0], 1)), X]
y = y[:, np.newaxis]

####################################################################

uniq=np.unique(y)
# print(uniq)
prediction_list=[]
for p in uniq:

#     print(p)
    y_new=np.copy(y)
#     print(y_new.T)
    for x in range(len(y_new)):
        if y_new[x]==p:
            y_new[x]=1
        else:
            y_new[x]=0
            
#     print(y_new.T)

    theta = np.zeros((X.shape[1], 1))

    z=np.dot(X,theta)

    h = sigmoid(z)

    for i in range(1000):
        gradient = np.dot(X.T, (h - y_new)) / y_new.shape[0]
        lr = 0.01
        theta -= lr * gradient
        z = np.dot(X, theta)
        h = sigmoid(z)

    prdctn=sigmoid(np.dot(X_validate, theta))
    p = [q[0] for q in prdctn]
    prediction_list.append(p)
#     print(prdctn.shape)
    
prediction_list=np.array(prediction_list)
predicted_classes = np.argmax(prediction_list,axis=0)
predicted_classes = [uniq[p] for p in predicted_classes]
# print(predicted_classes)
# print(prediction_list.shape)

# max_values = np.max(prediction_list,axis=0)
# print(max_values)

true=0
total=0
# print(y_validate,predicted_classes)

for x in range(len(y_validate)):
    if y_validate[x]==predicted_classes[x]:
        true+=1
    total+=1

print("accuracy : ",(true/total))


# In[ ]:




