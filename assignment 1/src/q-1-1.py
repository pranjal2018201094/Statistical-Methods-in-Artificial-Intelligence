
#get_ipython().run_line_magic('matplotlib', 'inline')

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import operator
import pprint

df=pd.read_csv("train.csv")



def train_validate_split(df,test_size):
    if isinstance(test_size,float):
        test_size = round(test_size*len(df))
        
    indices=df.index.tolist()
    validate_indices = random.sample(population=indices,k=test_size)
    
    validate_df=df.loc[validate_indices]
    train_df=df.drop(validate_indices)
    return train_df,validate_df


# random.seed(0)
train_df,validate_df=train_validate_split(df,test_size=0.2)


# print(len(df))
# print(len(train_df))
# print(len(validate_df))
validate_df.head()
name_list=list(df)
column_list=[5,7,8,9]
diction=dict()
diction["salary"]=9
diction["sales"]=8
diction["promotion_last_5years"]=7
diction["Work_accident"]=5


training_data=train_df.values
validate_data=validate_df.values
tr=0
fl=0
for x in validate_data:
    if x[6]==1:
        tr+=1
    else:
        fl+=1
# print(tr)
# print(fl)
# print(training_data[1:10,:])
# temp=np.unique(training_data[:,6])
# print(temp)
# print(validate_data[:1000,6])




def is_pure(data):
    unique_vals=np.unique(data[:,6])
    if len(unique_vals)==1:
        return unique_vals[0]
    return -1



def classify_data(training_data):
    label_col=training_data[:6]
    unique_classes,unique_class_count=np.unique(label_col, return_counts=True)
    index=unique_class_count.argmax()
    classification=unique_classes[index]
    return classification


def entropy_cal(training_data,column_no):
    unique_vals,unique_val_count=np.unique(training_data[:,column_no],return_counts=True)
    total_len=len(training_data)
    entropy_sum=0
    dic=dict()
    for j in range(len(unique_vals)):
        true=0
        false=0
        for i in training_data:
            if (i[6]==1)and(unique_vals[j]==i[column_no]):
                true+=1
            elif (i[6]==0)and(unique_vals[j]==i[column_no]):
                false+=1

        p_true=true/(true+false)
        p_false=false/(true+false)
        total=unique_val_count[j]
#         print(total)
        if p_true and p_false:
            entropy=(-1)*(p_true*(math.log2(p_true))+p_false*(math.log2(p_false)))
            dic[unique_vals[j]]=entropy
#         print(entropy)
            entropy_weighted=(entropy*total)/total_len
            entropy_sum+=entropy_weighted
#     print(entropy_sum)
    return (entropy_sum,dic)


def overall_entropy(training_data):
    true=0
    false=0
    for i in training_data[:,6]:
        if i==1:
            true=true+1
        elif i==0:
            false=false+1
    p_true=true/(true+false)
    p_false=false/(true+false)
    entropy_of_training_data=(-1)*(p_true*(math.log2(p_true))+p_false*(math.log2(p_false)))
#     print(entropy_of_training_data)
    
    return entropy_of_training_data


def information_gain(training_data,col_list):
    entropy_of_training_data=overall_entropy(training_data);
    
    entropy_of_unique_vals=dict()
    entropies_list=dict()
    for i in col_list:
        temp,entropy_of_unique_vals[i]=entropy_cal(training_data,i)
        entropies_list[i]=temp

    sorted_x = sorted(entropies_list.items(), key=lambda x: x[1])
    
#     print()
    selected=sorted_x[0][0]    
    gain=sorted_x[0][1]

#     info_gain=entropy_of_training_data-gain
#     print()
#     print(entropy_of_unique_vals)
#     print(info_gain)
    return selected


def get_subtable(training_data,node,value):    
    new_data=[]
    for i in training_data:
        if i[node]==value:
            new_data.append(list(i))
    
    data=np.array(new_data,dtype=object)
#     print("ppppppppp")
#     print(data)
#     print("ppppppppp")
    return data


def decision_tree(column_list,training_data,tree=None):
    
    
    column = information_gain(training_data,column_list)
#     print("...")
#     print(column)
#     print("...")
    unique_vals = np.unique(training_data[:,column])
    
     #Create an empty dictionary to create tree    
    if tree is None:                    
        tree={}
        tree[name_list[column]] = {}
    column_list.remove(column)   
    for value in unique_vals:
        new_data=get_subtable(training_data,column,value)
#         print("new_data")
#         print(new_data[0][0])
        
        n = is_pure(new_data)
        if n!=-1:            
            tree[name_list[column]][value]=n
        elif len(column_list)==0:
            zeroes=0
            ones=0
            for i in new_data:
                if i[6]==0:
                    zeroes+=1
                else:
                    ones+=1
            if ones>zeroes:
                tree[name_list[column]][value]=1
            else:
                tree[name_list[column]][value]=0
        else:
            tree[name_list[column]][value]=decision_tree(column_list,new_data)
#     print(tree)        
    column_list.append(column)
    return tree

tree=decision_tree(column_list,training_data)

# pprint.pprint(tree)

def predict(tree,inst):
    try:
        for x in tree.keys():
            col=diction[x]

    #         print(nodes)
            value = inst[col]
#             print(x,value)
            tree = tree[x][value]
    #         print(tree)
            prediction = 0

            if type(tree) is dict:
                prediction = predict(tree,inst)
            else:
                prediction = tree
                break;                            
    #     print(prediction)
        return prediction
    except:
        return 0

# In[1028]:

tp=0
tn=0
fp=0
fn=0
for i in validate_data:
    if(predict(tree,i) == i[6]):
        if i[6]==0:
            tp+=1
        else:
            tn+=1
    else:
        if i[6]==1:
            fp+=1
        else:
            fn+=1
        
# print(tn)
# print(tp)
# print(fn)
# print(fp)
accuracy=(tp+tn)/(tp+tn+fn+fp)
if tp!=0:
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    f1_score=2*(recall * precision) / (recall + precision)

else:
    precision=0
    recall=0
    f1_score=0
    
print("accuracy :",accuracy)
print("precision :",precision)
print("recall :",recall)
print("f1_score :",f1_score)



