#!/usr/bin/env python
# coding: utf-8

# # import Statements

# In[1]:


import numpy as np
import math
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')

import random
from pprint import pprint


# In[2]:


df = pd.read_csv("intrusion_detection/data.csv")
# df = df.rename(columns={"left": "label"})


# In[5]:


# df.info()



# random.seed(0)
# train_df,validate_df=train_validate_split(df,test_size=0.2)


# In[8]:
def distances(test,centroids):
#     distance=0
    distances=[]
    for i in range(5):    
        centroid=centroids[i+1]
        distance=0
        for j in range(14):
            distance=distance+pow((test[j]-centroid[j]),2)
        distances.append(math.sqrt(distance))
#     print(distances)
    return distances

def Agglomerative():
    print("Hierarchical")
    print("...............")
    data = df.values
    data=data[:,:-1]
    label=df.values
    label=label[:,-1]
    # validate_df.head()
    name_list=list(df)

    # print(data.shape,label.shape)

    mean_list=[]
    new_values=data.transpose()
    # print(new_values.shape,data.shape)
    # for x in new_values:
    #     mn=np.mean(x,axis=0)
    #     mean_list.append(mn)

    # print(mean_list)
    X = df.loc[:,'duration':'dst_host_srv_rerror_rate']

    # for i in range(len(data)):
    X.loc[:,'duration':'dst_host_srv_rerror_rate']=(X.loc[:,'duration':'dst_host_srv_rerror_rate'] - X.loc[:,'duration':'dst_host_srv_rerror_rate'].mean())/X.loc[:,'duration':'dst_host_srv_rerror_rate'].std()
        
    # print(X)
    # print(X.shape)
    # Q=X.transpose()
    # print(Q.shape)
    # Q=np.matmul(X,Q)
    # print(Q.shape)
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


    # In[9]:


    cluster = AgglomerativeClustering(n_clusters=5,linkage='single')  
    array=cluster.fit_predict(new_data)

    data2=df.values
    list1=[]
    list2=[]
    list3=[]
    list4=[]
    list5=[]

    catagories=[]
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
            

    maximum=0

    y=np.array(list1)
    uniq,count=np.unique(y[:,-1],return_counts=True)
    catagory1=uniq[np.argmax(count)]
    accuracy1=max(count)/np.sum(count)
    maximum=maximum+max(count)
    print(catagory1,accuracy1)

    y=np.array(list2)
    uniq,count=np.unique(y[:,-1],return_counts=True)
    catagory2=uniq[np.argmax(count)]
    accuracy2=max(count)/np.sum(count)
    maximum=maximum+max(count)
    print(catagory2,accuracy2)

    y=np.array(list3)
    uniq,count=np.unique(y[:,-1],return_counts=True)
    catagory3=uniq[np.argmax(count)]
    accuracy3=max(count)/np.sum(count)
    maximum=maximum+max(count)
    print(catagory3,accuracy3)

    y=np.array(list4)
    uniq,count=np.unique(y[:,-1],return_counts=True)
    catagory4=uniq[np.argmax(count)]
    accuracy4=max(count)/np.sum(count)
    maximum=maximum+max(count)
    print(catagory4,accuracy4)

    y=np.array(list5)
    uniq,count=np.unique(y[:,-1],return_counts=True)
    catagory5=uniq[np.argmax(count)]
    accuracy5=max(count)/np.sum(count)
    maximum=maximum+max(count)
    print(catagory5,accuracy5)

    catagories=[catagory1,catagory2,catagory3,catagory4,catagory5]
    final_acc=[accuracy1,accuracy2,accuracy3,accuracy4,accuracy5]
    final_accuracy=maximum/len(data2)
    print("final_acc  : ",final_accuracy )
    return final_acc,catagories

def Kmeans():
    print("K-means")
    print("...............")
    data = df.values
    data2= df.values
    data=data[:,:-1]
    label=df.values
    label=label[:,-1]
    # validate_df.head()
    name_list=list(df)

    # print(data.shape,label.shape)

    mean_list=[]
    new_values=data.transpose()
    # print(new_values.shape,data.shape)
    # for x in new_values:
    #     mn=np.mean(x,axis=0)
    #     mean_list.append(mn)

    # print(mean_list)
    X = df.loc[:,'duration':'dst_host_srv_rerror_rate']

    # for i in range(len(data)):
    X.loc[:,'duration':'dst_host_srv_rerror_rate']=(X.loc[:,'duration':'dst_host_srv_rerror_rate'] - X.loc[:,'duration':'dst_host_srv_rerror_rate'].mean())/X.loc[:,'duration':'dst_host_srv_rerror_rate'].std()
        
    # print(X.shape)
    # Q=X.transpose()
    # print(Q.shape)
    # Q=np.matmul(X,Q)
    # print(Q.shape)
    covarience=np.cov(np.array(X.T))
    # print(covarience.shape)

    eigen_values,eigen_vectors=np.linalg.eig(covarience)
    # print(eigen_values)

    max_val=[]
    for i in range(len(eigen_values)):
        temp=eigen_values[i]/sum(eigen_values)
        max_val.append((temp,i))
        
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

    # print(type(data),type(new_data))
    new_data=eigen_vectors[:,newdata_list]    
    # print()
    new_data=np.dot(X,new_data)


    # In[12]:


    # In[ ]:


    np.random.seed(200)
    k = 5

    centroids = {}
    for i in range(k):
        centroids[i+1]=new_data[random.randint(0,12000)]
    # print(centroids)

    data_dic={}
    for i in range(len(new_data)):
        distance=distances(new_data[i],centroids)
        data_dic[i]=distance.index(min(distance))+1
        
    # print(data_dic)
    # count=[0,0,0,0,0]
    list1=[]
    list2=[]
    list3=[]
    list4=[]
    list5=[]
    for i in data_dic:
        x=data_dic[i]
        if x==1:
            list1.append(new_data[i])
        elif x==2:
            list2.append(new_data[i])
        elif x==3:
            list3.append(new_data[i])
        elif x==4:
            list4.append(new_data[i])
        elif x==5:
            list5.append(new_data[i])
            
    #     count[data_dic[i]-1]+=1
        
    # print(count)
    # print(len(list1))
    # print(len(list2))
    # print(len(list3))
    # print(len(list4))
    # print(len(list5))

    mn=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    for i in range(20):
        new_centroids={}
        
    #     print(len(mn))
        mn=[float(sum(l))/len(l) for l in zip(*list1)]
        new_centroids[1]=mn

        mn=[float(sum(l))/len(l) for l in zip(*list2)]
        new_centroids[2]=mn

        mn=[float(sum(l))/len(l) for l in zip(*list3)]
        new_centroids[3]=mn

        mn=[float(sum(l))/len(l) for l in zip(*list4)]
        new_centroids[4]=mn

        mn=[float(sum(l))/len(l) for l in zip(*list5)]
        new_centroids[5]=mn

        data_dic={}
        for i in range(len(new_data)):
            distance=distances(new_data[i],new_centroids)
            data_dic[i]=distance.index(min(distance))+1

    #     print(data_dic)
        new_centroids={}
    #     count=[0,0,0,0,0]
        list1=[]
        list2=[]
        list3=[]
        list4=[]
        list5=[]
        
        final_list1=[]
        final_list2=[]
        final_list3=[]
        final_list4=[]
        final_list5=[]

        for i in data_dic:
            x=data_dic[i]
            if x==1:
                list1.append(new_data[i])
                final_list1.append(data2[i])
            elif x==2:
                list2.append(new_data[i])
                final_list2.append(data2[i])
            elif x==3:
                list3.append(new_data[i])
                final_list3.append(data2[i])
            elif x==4:
                list4.append(new_data[i])
                final_list4.append(data2[i])
            elif x==5:
                list5.append(new_data[i])
                final_list5.append(data2[i])
        
        # print()
        # print(len(list1),len(final_list1))
        # print(len(list2),len(final_list2))
        # print(len(list3),len(final_list3))
        # print(len(list4),len(final_list4))
        # print(len(list5),len(final_list5))
        
    # a = np.array(a)
    y=np.array(final_list1)
    uniq,count=np.unique(y[:,-1],return_counts=True)
    catagory1=uniq[np.argmax(count)]
    accuracy1=max(count)/np.sum(count)
    print(catagory1,accuracy1)

    y=np.array(final_list2)
    uniq,count=np.unique(y[:,-1],return_counts=True)
    catagory2=uniq[np.argmax(count)]
    accuracy2=max(count)/np.sum(count)
    print(catagory2,accuracy2)

    y=np.array(final_list3)
    uniq,count=np.unique(y[:,-1],return_counts=True)
    catagory3=uniq[np.argmax(count)]
    accuracy3=max(count)/np.sum(count)
    print(catagory3,accuracy3)

    y=np.array(final_list4)
    uniq,count=np.unique(y[:,-1],return_counts=True)
    catagory4=uniq[np.argmax(count)]
    accuracy4=max(count)/np.sum(count)
    print(catagory4,accuracy4)

    y=np.array(final_list5)
    uniq,count=np.unique(y[:,-1],return_counts=True)
    catagory5=uniq[np.argmax(count)]
    accuracy5=max(count)/np.sum(count)
    print(catagory5,accuracy5)

    catagories=[catagory1,catagory2,catagory3,catagory4,catagory5]
    final_acc=[accuracy1,accuracy2,accuracy3,accuracy4,accuracy5]
    # final_accuracy=maximum/len(data2)
    # print("final_acc  : ",final_acc )

    final_accuracy=(accuracy1+accuracy2+accuracy3+accuracy4+accuracy5)/5

    print("final_acc  : ",final_accuracy )
    return final_acc,catagories


def Gmm():
    print("GMM")
    print("...............")
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

    catagories=[catagory1,catagory2,catagory3,catagory4,catagory5]
    final_acc=[accuracy1,accuracy2,accuracy3,accuracy4,accuracy5]
    final_accuracy=(accuracy1+accuracy2+accuracy3+accuracy4+accuracy5)/5
    print("final_acc  : ",final_accuracy )
    return final_acc,catagories



# fig1, ax1 = plt.subplots()
# ax1.pie(final_acc, autopct='%1.1f%%',shadow=True, startangle=90)
# ax1.axis('equal')
# plt.show()
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print()
acc_kmeans,cata_kmeans=Kmeans()
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print()
acc_gmm,cata_gmm=Gmm()
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print()
acc_agl,cata_agl=Agglomerative()
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

new_x=acc_agl+acc_gmm+acc_kmeans
new_y=cata_agl+cata_gmm+cata_kmeans
# print((new_x),(new_y))
fig1, ax1 = plt.subplots()
ax1.pie(acc_agl, labels=cata_agl, autopct='%1.1f%%',shadow=True, startangle=90)
plt.title("Hierarchical", bbox={'facecolor':'0.8', 'pad':5})
plt.show()

fig2, ax2 = plt.subplots()
ax2.pie(acc_gmm, labels=cata_gmm, autopct='%1.1f%%',shadow=True, startangle=90)
plt.title("GMM", bbox={'facecolor':'0.8', 'pad':5})
plt.show()

fig3, ax3 = plt.subplots()
ax3.pie(acc_kmeans, labels=cata_kmeans, autopct='%1.1f%%',shadow=True, startangle=90)
plt.title("K-means", bbox={'facecolor':'0.8', 'pad':5})

plt.show()
