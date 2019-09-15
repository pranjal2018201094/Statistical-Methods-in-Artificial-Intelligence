#!/usr/bin/env python
# coding: utf-8

# In[234]:

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

# In[235]:


def train_validate_split(df,test_size):
    if isinstance(test_size,float):
        test_size = round(test_size*len(df))
        
    indices=df.index.tolist()
    validate_indices = random.sample(population=indices,k=test_size)
    
    validate_df=df.loc[validate_indices]
    train_df=df.drop(validate_indices)
    return train_df,validate_df



def get_subtable(training_data):    
    true_data=[]
    false_data=[]
    for i in training_data:
        if i[-1]==1:
            true_data.append(list(i))
        elif i[-1]==0:
            false_data.append(i)
    true_dat=np.array(true_data,dtype=object)
    false_dat=np.array(false_data,dtype=object)
    return true_dat,false_dat


# In[243]:


def prob_of_col_catag(col,data,value):
    count=0
    tfs=len(data)
    for x in data:
        if x[col]==value:
            count+=1
    prob=count/tfs
    return prob


# In[244]:


def prob_of_col_num(col,data,value):
    n=len(data)
    mean=data[:,col].mean()
#     print(mean)
#     variance = sum([pow(x-mn,2) for x in data[:,col]])/float(len(data[:,col])-1)
#     print(variance)
#     stdev=math.sqrt(variance)
#     print(std_dev)
    stdev=data[:,col].std()
#     exponent=math.exp(-1*(pow(value-mn,2)/(2*pow(std_dev,2))))
#     prob=(1/(math.sqrt(2*math.pi)*std_dev))*exponent
    exponent = math.exp(-(math.pow(value-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
#     print(prob)
#     return prob


# In[245]:


# prob_of_col_num(1,training_data,validation_data[5,1])


# In[255]:


def calc(validation,training_data,true_table,false_table,inv_ind):
    prob_true=1
    prob_false=1
    for x in inv_ind:
#         print(x)
        if inv_ind[x]==10:
            prob=prob_of_col_num(x,true_table,validation[x])
        elif inv_ind[x]==20:
            prob=prob_of_col_catag(x,true_table,validation[x])
#         print(prob)
        prob_true=prob_true*prob
    prob_true=prob_true*len(true_table)/(len(true_table)+len(false_table))    
#     print(" ")
    for x in inv_ind:
        if inv_ind[x]==10:
            prob=prob_of_col_num(x,false_table,validation[x])
        elif inv_ind[x]==20:
            prob=prob_of_col_catag(x,false_table,validation[x])
        prob_false=prob_false*prob
    prob_false=prob_false*len(false_table)/(len(true_table)+len(false_table))

#     print(prob_true,prob_false)
    
    if prob_true>prob_false:
        prediction=1
    else:
        prediction=0
    return prediction


# In[ ]:





# In[258]:


def main(test_file):
    # random.seed(0)
    df=pd.read_csv("data.csv", header = None)
    df = df[1:]
    new_arr=df.values
    # trimmed_matrix = new_arr[:,1:len(new_arr[0])]
    training_df,validation_df=train_validate_split(df,.2)
    training_data=training_df.values
    
    validation_data=validation_df.values
#     test_df=pd.read_csv(test_file, header = None)
#     validation_data=test_df.values
    
    training_data[:,[-1,10]] = training_data[:,[10,-1]]
    validation_data[:,[-1,10]] = validation_data[:,[10,-1]]
#     print(validation_data.shape)

    inv_ind={1:10,2:10,3:10,4:10,5:10,6:10,7:20,8:10,9:20,10:20,11:20,12:20}
    
    true_table,false_table=get_subtable(training_data)
    # print(validation_data.shape)
    true=0
    false=0
    for x in validation_data:
        prediction=calc(x,training_data,true_table,false_table,inv_ind)
        if prediction==x[-1]:
            true+=1
        else:
            false+=1
    print(true,false)
    print(true/(false+true))


# In[259]:


test_file=sys.argv[1]
main(test_file)


# In[ ]:




