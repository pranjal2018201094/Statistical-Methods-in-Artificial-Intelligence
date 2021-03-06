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


random.seed(0)
train_df,validate_df=train_validate_split(df,test_size=0.2)

train_data=train_df.values
validate_data=validate_df.values

name_list=list(df)
column_list=[0,1,2,3,4,5,7,8,9]


dictionary_for_type=dict()
dictionary_for_type[0]=1
dictionary_for_type[1]=1
dictionary_for_type[2]=1
dictionary_for_type[3]=1
dictionary_for_type[4]=1
# dictionary_for_type["left"]=6
dictionary_for_type[5]=2
dictionary_for_type[7]=2
dictionary_for_type[8]=2
dictionary_for_type[9]=2

dictionary_for_col_no=dict()
dictionary_for_col_no["salary"]=9
dictionary_for_col_no["sales"]=8
dictionary_for_col_no["promotion_last_5years"]=7
dictionary_for_col_no["Work_accident"]=5
# dictionary_for_col_no["left"]=6
dictionary_for_col_no["time_spend_company"]=4
dictionary_for_col_no["average_montly_hours"]=3
dictionary_for_col_no["number_project"]=2
dictionary_for_col_no["last_evaluation"]=1
dictionary_for_col_no["satisfaction_level"]=0



def entropy_cal_for_catag(data,column_no):
    unique_vals,unique_val_count=np.unique(data[:,column_no],return_counts=True)
    total_len=len(data)
    sys_entropy=overall_entropy(data)
    entropy_sum=0
    for j in range(len(unique_vals)):
        true=0
        false=0
        for i in data:
            if (i[6]==1)and(unique_vals[j]==i[column_no]):
                true+=1
            elif (i[6]==0)and(unique_vals[j]==i[column_no]):
                false+=1

        p_true=true/(true+false)
        p_false=false/(true+false)
        total=unique_val_count[j]
        if p_true and p_false:
            entropy=(-1)*(p_true*(math.log2(p_true))+p_false*(math.log2(p_false)))
            entropy_weighted=(entropy*total)/total_len
            entropy_sum+=entropy_weighted

    information_gain=sys_entropy-entropy_sum
    
    return (information_gain)


def entropy_cal_for_num(data,column_no):
    unique_vals,unique_val_count=np.unique(data[:,column_no],return_counts=True)
    # total_len=len(training_data)
    # max_entropy=0
    # entropy_sum=0
    # dic=dict()
    key=None
    sys_entropy=overall_entropy(data)
    information_gain=0
    unique_vals.sort()
    # print(unique_vals)
    for i in range(len(unique_vals)-1):
        first=unique_vals[i]
        second=unique_vals[i+1]
        t=(first+second)/2

        true1=0
        true2=0
        false1=0
        false2=0
        for j in data:
            if j[column_no]<t:
                if j[6]==1:
                    true1+=1
                else:
                    false1+=1        
            else:
                if j[6]==1:
                    true2+=1
                else:
                    false2+=1    
        p_true1=true1/(true1+false1)
        p_false1=false1/(true1+false1)
        if (p_true1==0)or(p_true1==1):
            local_entropy1=0
        else:
            local_entropy1=(-1)*(p_true1*(math.log2(p_true1))+p_false1*(math.log2(p_false1)))
            
        p_true2=true2/(true2+false2)
        p_false2=false2/(true2+false2)
        if (p_true2==0)or(p_true2==1):
            local_entropy2=0
        else:
            local_entropy2=(-1)*(p_true2*(math.log2(p_true2))+p_false2*(math.log2(p_false2)))

        total=true1+true2+false1+false2
        local_entropy=((true1+false1)*local_entropy1)/total+((true2+false2)*local_entropy2)/total
        if information_gain<(sys_entropy-local_entropy):
            information_gain=(sys_entropy-local_entropy)
            key=t
        # print(key)
    return(information_gain,key)
        
def overall_entropy(data):
    true=0
    false=0
    for i in data[:,6]:
        if i==1:
            true=true+1
        elif i==0:
            false=false+1
    p_true=true/(true+false)
    p_false=false/(true+false)
    entropy_of_training_data=(-1)*(p_true*(math.log2(p_true))+p_false*(math.log2(p_false)))
    
    return entropy_of_training_data


def information_gain(data,col_list):
    max_info_gain=0
    selected=None
    key=None
    for x in col_list:
        if dictionary_for_type[x]==1:
            # numerical
            info_gain,keyy=entropy_cal_for_num(data,x)
            if info_gain>max_info_gain:
                max_info_gain=info_gain
                selected=x
                key=keyy
        else:
            # categorical
            info_gain=entropy_cal_for_catag(data,x)
            if(info_gain<0):
                info_gain=0 
            # print(info_gain)
            if info_gain>=max_info_gain:
                max_info_gain=info_gain
                selected=x
                key=-1
                # print(x)
    return(selected,key)

# print(information_gain(train_data,col_list))  

def get_subtable(training_data,node,value):    
    new_data=[]
    for i in training_data:
        if i[node]==value:
            new_data.append(list(i))
    
    data=np.array(new_data,dtype=object)
    return data

def is_pure(data):
    unique_vals=np.unique(data[:,6])
    if len(unique_vals)==1:
        return unique_vals[0]
    return -1


def decision_tree(column_list,training_data,tree=None): 
    column,key = information_gain(training_data,column_list)
    unique,count=np.unique(training_data[:,6],return_counts=True)
    # print(unique,count)
    if count[0]>count[1]:
        prob=0
    else:
        prob=1
    # print(prob)
    if tree is None:                    
        tree={}
        tree[(name_list[column],key,prob)] = {}

    
    if key==-1:
        unique_vals = np.unique(training_data[:,column])
        column_list.remove(column)   
        for value in unique_vals:
            new_data=get_subtable(training_data,column,value)
            n = is_pure(new_data)
            if n!=-1:            
                tree[(name_list[column],key,prob)][value]=n
            else:
                tree[(name_list[column],key,prob)][value]=decision_tree(column_list,new_data)
        column_list.append(column)
        
    else:
        data_greater=training_data[(training_data[:,column])>key]
        data_smaller=training_data[(training_data[:,column])<key]


        n = is_pure(data_greater)
        if n!=-1:            
            tree[(name_list[column],key,prob)]["greater"]=n
        else:
            tree[(name_list[column],key,prob)]["greater"]=decision_tree(column_list,data_greater)

        n = is_pure(data_smaller)
        if n!=-1:
            # print(n)            
            tree[(name_list[column],key,prob)]["smaller"]=n
        else:
            tree[(name_list[column],key,prob)]["smaller"]=decision_tree(column_list,data_smaller)
    
    return tree

tree=decision_tree(column_list,train_data)
# pprint.pprint(tree)
# print(tree)


def predict(tree,depth,inst):
    
    try:
        for x in tree.keys():
            depth=depth-1
            if (depth==(1)):
                return x[2]
            col=dictionary_for_col_no[x[0]]
            if x[1]==-1:
                value = inst[col]
                tree = tree[x][value]
                prediction = 0
                if type(tree) is dict:
                    prediction = predict(tree,depth,inst)
                else:
                    prediction = tree
                    break;                            
                # print(prediction)
                # return prediction
            else:
                value =inst[col]
                if value<x[1]:
                    tree = tree[x]["smaller"]
                else:
                    tree = tree[x]["greater"]
                prediction = 0
                if type(tree) is dict:
                    prediction = predict(tree,depth,inst)
                else:
                    prediction = tree
                    break;             

        return prediction
    except:
        return 0

def calculate(validate_data,depth):
    true_positive=0
    true_negative=0
    false_positive=0
    false_negative=0
    for i in validate_data:
        p = predict(tree,depth,i)
        if(p == i[6]):
            if(i[6]==1):
                true_positive+=1
            else:
                true_negative+=1
        else:
            if(i[6]==1): 
                false_positive+=1
            else:
                false_negative+=1

    # print(true_negative)
    # print(true_positive)
    # print(false_negative)
    # print(false_positive)
    accuracy=(true_positive+true_negative)/(true_positive+true_negative+false_positive+false_positive)


    # print(accuracy)

    return 1-accuracy

error=[]
error2=[]
depth = list(range(3,20))
for i in range(3,20):
    error.append(calculate(validate_data,i))
    error2.append(calculate(train_data,i))
y = error
y2=error2
x = depth
plt.plot(x, y,c="red")
plt.plot(x, y2,c="blue")
plt.xlabel("depth")
plt.ylabel("error")
plt.title('Error vs Depth ')
plt.show()