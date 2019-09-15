#!/usr/bin/env python
# coding: utf-8

# In[72]:


import numpy as np
import math
import pandas as pd
from scipy.optimize import fmin_tnc
import matplotlib.pyplot as plt
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')

import random


# In[73]:


df = pd.read_csv("AdmissionDataset/data.csv")


# In[124]:


def train_validate_split(df,test_size):
    if isinstance(test_size,float):
        test_size = round(test_size*len(df))
        
    indices=df.index.tolist()
    validate_indices = random.sample(population=indices,k=test_size)
    
    validate_df=df.loc[validate_indices]
    train_df=df.drop(validate_indices)
    return train_df,validate_df


# In[125]:


train_df,validate_df=train_validate_split(df,0.2)


# In[126]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# In[127]:


data=train_df

initial_theta = [0,0]
alpha = 0.1
iterations = 1000

X = data.iloc[:,1:-1]
y = data.iloc[:, -1]

X=(X-np.mean(X))/np.std(X)

X_validate=validate_df.iloc[:,1:-1]
y_validate=validate_df.values[:,-1]
X_validate=(X_validate-np.mean(X))/np.std(X)

X_validate=np.c_[np.ones((X_validate.shape[0], 1)), X_validate]

X = np.c_[np.ones((X.shape[0], 1)), X]
y = y[:, np.newaxis]
theta = np.zeros((X.shape[1], 1))

z=np.dot(X,theta)

h = sigmoid(z)

for i in range(1000):
    gradient = np.dot(X.T, (h - y)) / y.shape[0]
    lr = 0.01
    theta -= lr * gradient
    z = np.dot(X, theta)
    h = sigmoid(z)
    
prdctn=sigmoid(np.dot(X_validate, theta))
final_prediction=[]
for i in range(len(prdctn)):
#       print(y[i],prdctn[i])
    if prdctn[i]>=(.5):
        final_prediction.append(1)
    else:
        final_prediction.append(0)
        
# print(final_prediction)
true=0
total=0
# print(y_validate)
y_validation=[]
for x in range(len(y_validate)):
    if y_validate[x]>=(.5):
        y_validation.append(1)
    else:
        y_validation.append(0)
        
# print(y_validation)
for x in range(len(y_validation)):
    if y_validation[x]==final_prediction[x]:
        true+=1
    total+=1

print("accuracy : ",(true/total))
