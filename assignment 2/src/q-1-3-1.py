#!/usr/bin/env python
# coding: utf-8

# In[197]:

import sys
import math
import pandas as pd
import numpy as np
import random
import operator
import pandas as pd
from sklearn.metrics import r2_score


# In[203]:


def train_validate_split(df,test_size):
    if isinstance(test_size,float):
        test_size = round(test_size*len(df))
        
    indices=df.index.tolist()
    validate_indices = random.sample(population=indices,k=test_size)
    
    validate_df=df.loc[validate_indices]
    train_df=df.drop(validate_indices)
#     print(train_df,validate_df)
    return train_df,validate_df


# In[ ]:





# In[204]:


def predict(data,theta):
#     print(data.shape,theta.shape)
    predicted=np.matmul(data,theta)
    return predicted


# In[214]:


def main(test_file):
    df=pd.read_csv("AdmissionDataset/data.csv")
    del df['Serial No.']
    
#     random.seed(0)
    training_df,validation_df=train_validate_split(df,.2)
    training_data=training_df.values
    label_train=training_data[:,-1]
    training_data=training_data[:,:-1]
    
#     print(training_data)
    length=len(training_data)
    training_data=training_data.transpose()
    a=np.ones(length)
    training_data=np.insert(training_data, 0,a,0)
#     print(training_data)

    
    training_data=training_data.transpose()
    training_data_traspose=training_data.transpose()
#     print(label_train.shape)
#     print(training_data.shape,training_data_traspose.shape)
    
    one=np.matmul(training_data_traspose, training_data)
    two=np.linalg.inv(one)
    three=np.matmul(two,training_data_traspose)
    theta=np.matmul(three,label_train)
#     print(theta)
    
#     df_test=pd.read_csv("AdmissionDataset/data.csv")
#     del df['Serial No.']
#     validation_data=df_test.values

    validation_data=validation_df.values
    label_validate=validation_data[:,-1]
    validation_data=validation_data[:,:-1]
    
    length2=len(validation_data)
    validation_data=validation_data.transpose()
#     print(training_data)
    b=np.ones(length2)
    validation_data=np.insert(validation_data, 0,b,0)
    validation_data=validation_data.transpose()
#     print(validation_data.shape)

#     theta_tra=theta.transpose()
    predicted=predict(validation_data,theta)
#     print(predicted.shape,label_validate.shape)
    print(r2_score(label_validate,predicted))
    


# In[217]:


test_file=sys.argv[1]
main(test_file)
