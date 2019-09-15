#!/usr/bin/env python
# coding: utf-8

# In[8]:

import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import operator
import pprint
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches


# # train-validate Split

# In[9]:


def train_validate_split(df,test_size):
    if isinstance(test_size,float):
        test_size = round(test_size*len(df))
        
    indices=df.index.tolist()
    validate_indices = random.sample(population=indices,k=test_size)
    
    validate_df=df.loc[validate_indices]
    train_df=df.drop(validate_indices)
    return train_df,validate_df


# # calculation distances

# In[10]:


def distances(train,test):
    distance=0
    for i in range(len(train)-1):
        distance=distance+pow((train[i]-test[i]),2)
#     print(distance)
    return distance


# In[11]:


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


# In[12]:


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


# # Remaining...................................

# In[26]:


def iris(test_file):
    df=pd.read_csv("Iris.csv")
    data=df.values
    random.seed(0)
    training_df,validation_df=train_validate_split(df,.2)
    training_data=training_df.values
    
    validation_data=validation_df.values
#     df_test=pd.read_csv(test_file)
#     validation_data=df_test.values
    
    tp=0
    tn=0
    fp=0
    fn=0
    for i in validation_data:
        predicted=checking(training_data,i,6)
        if predicted==i[-1]:
            if i[-1]=="Iris-setosa":
                tp+=1
            else:
                tn+=1
        else:
            if i[-1]=="Iris-setosa":
                fp+=1
            else:
                fn+=1


    accuracy1=(tp+tn)/(tp+tn+fn+fp)
    if tp!=0:
        precision1=tp/(tp+fp)
        recall1=tp/(tp+fn)
        f1_score1=2*(recall1 * precision1) / (recall1 + precision1)

    else:
        precision1=0
        recall1=0
        f1_score1=0
    
    tp=0
    tn=0
    fp=0
    fn=0
    for i in validation_data:
        predicted=checking(training_data,i,6)
        if predicted==i[-1]:
            if i[-1]=="Iris-virginica":
                tp+=1
            else:
                tn+=1
        else:
            if i[-1]=="Iris-virginica":
                fp+=1
            else:
                fn+=1


    accuracy2=(tp+tn)/(tp+tn+fn+fp)
    if tp!=0:
        precision2=tp/(tp+fp)
        recall2=tp/(tp+fn)
        f1_score2=2*(recall2 * precision2) / (recall2 + precision2)

    else:
        precision2=0
        recall2=0
        f1_score2=0
    
        tp=0
    tn=0
    fp=0
    fn=0
    for i in validation_data:
        predicted=checking(training_data,i,6)
        if predicted==i[-1]:
            if i[-1]=="Iris-versicolor":
                tp+=1
            else:
                tn+=1
        else:
            if i[-1]=="Iris-versicolor":
                fp+=1
            else:
                fn+=1


    accuracy3=(tp+tn)/(tp+tn+fn+fp)
    if tp!=0:
        precision3=tp/(tp+fp)
        recall3=tp/(tp+fn)
        f1_score3=2*(recall3 * precision3) / (recall3 + precision3)

    else:
        precision3=0
        recall3=0
        f1_score3=0
    
#     true=0
#     for i in validation_data:
#         predicted=checking(training_data,i,6)
#         if predicted==i[-1]:
#             true+=1
#     total=len(validation_data)
#     accuracy=true/total

    print("Iris")
    print("accuracy :",(accuracy1+accuracy2+accuracy3)/3)
    print("precision :",(precision1+precision2+precision3)/3)
    print("recall :",(recall1+recall2+recall3)/3)
    print("f1_score :",(f1_score1+f1_score2+f1_score3)/3)


# In[27]:


def robo1(test_file):
    df=pd.read_csv("Robot1",delimiter=" ")
    new_arr=df.values
    random.seed(0)
    training_df,validation_df=train_validate_split(df,.2)
    training_data=training_df.values
    
    validation_data=validation_df.values
#     df_test=pd.read_csv(test_file)
#     validation_data=df_test.values
    
    training_data=training_data[:,1:len(training_data[0])]
    validation_data=validation_data[:,1:len(validation_data[0])]
    training_data[:,[-1,0]] = training_data[:,[0,-1]]
    validation_data[:,[-1,0]] = validation_data[:,[0,-1]]
    training_data=training_data[:,2:len(training_data[0])]
    validation_data=validation_data[:,2:len(validation_data[0])]
    
    tp=0
    tn=0
    fp=0
    fn=0
    for i in validation_data:
        predicted=checking(training_data,i,6)
        if predicted==i[-1]:
            if i[-1]==0:
                tp+=1
            else:
                tn+=1
        else:
            if i[-1]==1:
                fp+=1
            else:
                fn+=1


    accuracy=(tp+tn)/(tp+tn+fn+fp)
    if tp!=0:
        precision=tp/(tp+fp)
        recall=tp/(tp+fn)
        f1_score=2*(recall * precision) / (recall + precision)

    else:
        precision=0
        recall=0
        f1_score=0

    print()
    print("robot 1")
    print("accuracy :",accuracy)
    print("precision :",precision)
    print("recall :",recall)
    print("f1_score :",f1_score)


# In[28]:


def robo2(test_file):
    df=pd.read_csv("Robot2",delimiter=" ")
    new_arr=df.values
    random.seed(0)
    training_df,validation_df=train_validate_split(df,.2)
    training_data=training_df.values
    
    validation_data=validation_df.values
#     df_test=pd.read_csv(test_file)
#     validation_data=df_test.values
    
    training_data=training_data[:,1:len(training_data[0])]
    validation_data=validation_data[:,1:len(validation_data[0])]
    training_data[:,[-1,0]] = training_data[:,[0,-1]]
    validation_data[:,[-1,0]] = validation_data[:,[0,-1]]
    training_data=training_data[:,2:len(training_data[0])]
    validation_data=validation_data[:,2:len(validation_data[0])]
    
    tp=0
    tn=0
    fp=0
    fn=0
    for i in validation_data:
        predicted=checking(training_data,i,6)
        if predicted==i[-1]:
            if i[-1]==0:
                tp+=1
            else:
                tn+=1
        else:
            if i[-1]==1:
                fp+=1
            else:
                fn+=1


    accuracy=(tp+tn)/(tp+tn+fn+fp)
    if tp!=0:
        precision=tp/(tp+fp)
        recall=tp/(tp+fn)
        f1_score=2*(recall * precision) / (recall + precision)

    else:
        precision=0
        recall=0
        f1_score=0

    print()
    print("robot 2")
    print("accuracy :",accuracy)
    print("precision :",precision)
    print("recall :",recall)
    print("f1_score :",f1_score)


# In[29]:


def main(test_file):
    test_file=[]
    iris(test_file)
    robo1(test_file)
    robo2(test_file)


# In[30]:

test_file=sys.argv[1]
main(test_file)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




