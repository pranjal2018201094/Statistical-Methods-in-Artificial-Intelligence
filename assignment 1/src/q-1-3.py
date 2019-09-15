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



def entropy_cal_for_catag(data,column_no,z):
    unique_vals,unique_val_count=np.unique(data[:,column_no],return_counts=True)
    total_len=len(data)
    sys_entropy=overall_entropy(data,z)
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
        
        if z==0:
            if p_true and p_false:
                entropy=(-1)*(p_true*(math.log2(p_true))+p_false*(math.log2(p_false)))
                entropy_weighted=(entropy*total)/total_len
                entropy_sum+=entropy_weighted

        elif z==1:
            gini=2*p_true*(p_false)
            entropy_weighted=(gini*total)/total_len
            entropy_sum+=entropy_weighted

        else:
            mis=min(p_true,p_false)
            entropy_weighted=(mis*total)/total_len
            entropy_sum+=entropy_weighted

    information_gain=sys_entropy-entropy_sum
    
    return (information_gain)


def entropy_cal_for_num(data,column_no,z):
    unique_vals,unique_val_count=np.unique(data[:,column_no],return_counts=True)
    # total_len=len(training_data)
    # max_entropy=0
    # entropy_sum=0
    # dic=dict()
    key=-1
    sys_entropy=overall_entropy(data,z)
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
        p_true2=true2/(true2+false2)
        p_false2=false2/(true2+false2)
        total=true1+true2+false1+false2

        if z==0:
            if (p_true1==0)or(p_true1==1):
                local_entropy1=0
            else:
                local_entropy1=(-1)*(p_true1*(math.log2(p_true1))+p_false1*(math.log2(p_false1)))

            if (p_true2==0)or(p_true2==1):
                local_entropy2=0
            else:
                local_entropy2=(-1)*(p_true2*(math.log2(p_true2))+p_false2*(math.log2(p_false2)))

            local_entropy=((true1+false1)*local_entropy1)/total+((true2+false2)*local_entropy2)/total

        elif z==1:
            local_gini1=2*p_true1*p_false1
            local_gini2=2*p_true2*p_false2
            local_entropy=((true1+false1)*local_gini1)/total+((true2+false2)*local_gini2)/total

        else:
            local_mis1=min(p_true1,p_false1)
            local_mis2=min(p_true2,p_false2)
            local_entropy=((true1+false1)*local_mis1)/total+((true2+false2)*local_mis2)/total


        if information_gain<(sys_entropy-local_entropy):
            information_gain=(sys_entropy-local_entropy)
            key=t
        # print(key)
    return(information_gain,key)
        
def overall_entropy(data,z):
    true=0
    false=0
    for i in data[:,6]:
        if i==1:
            true=true+1
        elif i==0:
            false=false+1
    p_true=true/(true+false)
    p_false=false/(true+false)

    if z==0:
        entropy_of_training_data=(-1)*(p_true*(math.log2(p_true))+p_false*(math.log2(p_false)))

    elif z==1:
        entropy_of_training_data=2*p_true*p_false
    else:
        entropy_of_training_data=min(p_true,p_false)

    return entropy_of_training_data


def information_gain(data,col_list,z):
    max_info_gain=0
    selected=None
    key=-1
    for x in col_list:
        if dictionary_for_type[x]==1:
            # numerical
            info_gain,keyy=entropy_cal_for_num(data,x,z)
            # print(info_gain)
            if info_gain>=max_info_gain:
                max_info_gain=info_gain
                selected=x
                key=keyy
        else:
            # categorical
            info_gain=entropy_cal_for_catag(data,x,z)
            if(info_gain<0):
                info_gain=0 
            # print(info_gain)
            if info_gain>=max_info_gain:
                max_info_gain=info_gain
                selected=x
                key=-1
                # print(x)
    return(selected,key)


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


def decision_tree(column_list,training_data,z,tree=None): 
    column,key = information_gain(training_data,column_list,z)

    if tree is None:                    
        tree={}
        tree[(name_list[column],key)] = {}

    if key==-1:
        unique_vals = np.unique(training_data[:,column])
        column_list.remove(column)   
        for value in unique_vals:
            new_data=get_subtable(training_data,column,value)
            n = is_pure(new_data)
            if n!=-1:            
                tree[(name_list[column],key)][value]=n
            else:
                tree[(name_list[column],key)][value]=decision_tree(column_list,new_data,z)
        column_list.append(column)
        
    else:
        data_greater=training_data[(training_data[:,column])>key]
        data_smaller=training_data[(training_data[:,column])<key]

        # print("pppppppppp")
        # print(len(data_greater),len(data_smaller))
        # print("qqqqqqqqqq")
        # x=[data_greater,data_smaller]
        # for new_data in x:
        n = is_pure(data_greater)
        if n!=-1:            
            tree[(name_list[column],key)]["greater"]=n
        else:
            tree[(name_list[column],key)]["greater"]=decision_tree(column_list,data_greater,z)

        n = is_pure(data_smaller)
        if n!=-1:
            # print(n)            
            tree[(name_list[column],key)]["smaller"]=n
        else:
            tree[(name_list[column],key)]["smaller"]=decision_tree(column_list,data_smaller,z)
    
    return tree

tree=[]
for z in range(3):
    tree.append(decision_tree(column_list,train_data,z))

# pprint.pprint(tree)
# print(tree)


def predict(tree,inst):
    try:
        for x in tree.keys():
            col=dictionary_for_col_no[x[0]]
            if x[1]==-1:
                value = inst[col]
                tree = tree[x][value]
                prediction = 0
                if type(tree) is dict:
                    prediction = predict(tree,inst)
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
                    prediction = predict(tree,inst)
                else:
                    prediction = tree
                    break;                            
                # print(prediction)
        return prediction
    except:
        return 0

##################################################################################################################

for z in range(3):
    true_positive=0
    true_negative=0
    false_positive=0
    false_negative=0
    for i in validate_data:
        p = predict(tree[z],i)
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
    if z==0:
        print("Entropy")
    elif z==1:
        print("Gini")
    else:
        print("Misclassification rate")
    accuracy=(true_positive+true_negative)/(true_positive+true_negative+false_positive+false_positive)

    precision=true_positive/(true_positive+false_positive)

    recall=true_positive/(true_positive+false_negative)

    print("accuracy :",accuracy)
    print("precision :",precision)
    print("recall :",recall)


