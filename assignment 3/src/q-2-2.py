#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import math
import pandas as pd
from scipy.optimize import fmin_tnc
import matplotlib.pyplot as plt
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')

import random


# In[16]:


df = pd.read_csv("AdmissionDataset/data.csv")


# In[17]:


def train_validate_split(df,test_size):
    if isinstance(test_size,float):
        test_size = round(test_size*len(df))
        
    indices=df.index.tolist()
    validate_indices = random.sample(population=indices,k=test_size)
    
    validate_df=df.loc[validate_indices]
    train_df=df.drop(validate_indices)
    return train_df,validate_df


# In[18]:


train_df,validate_df=train_validate_split(df,0.2)


# In[19]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# In[20]:


def logistic():
    data=train_df
    
    initial_theta = [0,0]
    alpha = 0.1
    iterations = 1000

    X = data.iloc[:,1:-1]
    y = data.iloc[:, -1]

    X=(X-np.mean(X))/np.std(X)

    X_validate=validate_df.iloc[:,1:-1]
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

    prdctn=sigmoid(np.dot(X_validate,theta))
    t_p=0
    t_n=0
    f_n=0
    f_p=0
    for i in range(len(prdctn)):
    #       print(y[i],prdctn[i])
        if y[i]>=(.5):
            if prdctn[i]>=(.5):
                t_p+=1
            else:
                f_n+=1
        else:
            if prdctn[i]<(.5):
                t_n+=1
            else:
                f_p+=1
    total=t_p+t_n+f_n+f_p
    print("Accuracy Logistic :",((t_p+t_n)/total))

# print(final_prediction)


# In[21]:


def distances(train,test):
    distance=0
    for i in range(len(train)-1):
        distance=distance+pow((train[i]-test[i]),2)
#     print(distance)
    return distance


# In[22]:


def find_min_k(training_data,test_sample,k):
#     print(training_data[0],test_sample,k)
    all_dist=dict()
    for x in range(len(training_data)):
#         print(x)
        all_dist[x]=(distances(training_data[x],test_sample)) 
    
    lists=(sorted(all_dist.items(), key = lambda kv:(kv[1], kv[0])))
    k_least=[]
    for i in range(k):
        k_least.append(lists[i])
    return k_least


# In[23]:


def checking(training_data,validation_data,k):
    k_least=find_min_k(training_data,validation_data,k)
#     print(k_least)
    uniq=np.unique(training_data[:,-1])
    uniq.sort()
    types=[]
    for i in range(len(uniq)):
        types.append(0)
    for x in k_least:
        for i in range(len(uniq)):
            if training_data[x[0],-1]==uniq[i]:
                types[i]+=1
            
    indx=types.index(max(types))
    predicted=uniq[indx]
    return predicted
#     print(k_least)


# In[24]:


def knn():
    
#     training_df,validation_df=train_validate_split(df,.2)
    train_data = train_df.iloc[:,1:]
    training_data=train_data.values
    for x in range(len(training_data)):
        if training_data[x][-1]>=(0.5):
            training_data[x][-1]=1
        else:
            training_data[x][-1]=0
    
    validate_data = validate_df.iloc[:,1:]
    validation_data=validate_data.values
    for x in range(len(validation_data)):
        if validation_data[x][-1]>=(0.5):
            validation_data[x][-1]=1
        else:
            validation_data[x][-1]=0

    true=0
    for i in validation_data:
        predicted=checking(training_data,i,6)
        if predicted==i[-1]:
            true+=1
    total=len(validation_data)
    accuracy=true/total

#     print("Iris")
    print("Accuracy Knn:",(accuracy))


logistic()

knn()


