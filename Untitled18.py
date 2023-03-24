#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
#counts_df=pd.read_csv("Downloads/counts.csv")
counts_df=pd.read_csv("Downloads/covid-selected-data.csv")
labels_df=pd.read_csv("Downloads/covid-selected-data-labels.csv")


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[3]:


display(labels_df)
#display(df_)
counts_df = counts_df.iloc[: , 1:]
display(counts_df)
#display(df_1)


# In[ ]:





# In[4]:


#Using LR


# In[5]:


X_train,X_test,Y_train,Y_test=train_test_split(counts_df.values,labels_df['type'].values,test_size=0.2)


# In[6]:


clf=LogisticRegression(class_weight='balanced')
clf.fit(X_train,Y_train)


# In[7]:


y_pred=clf.predict(X_test)


# In[8]:


from sklearn.metrics import classification_report, roc_auc_score
acc=accuracy_score(Y_test,y_pred)
print(f"Accuracy of Logistic Regression classifier model is:{acc*100}%")
print(classification_report(Y_test, y_pred))


# In[9]:


#KNN
from sklearn.neighbors import KNeighborsClassifier
clf2=KNeighborsClassifier(n_neighbors=3)
clf2.fit(X_train,Y_train)


# In[10]:


y_pred_KNN=clf2.predict(X_test)
acc1=accuracy_score(Y_test,y_pred_KNN)
print(f"Accuracy of KNN  classifier model is:{acc1*100}%")
print(classification_report(Y_test, y_pred_KNN))


# In[11]:


#SVM
from sklearn.svm import SVC
clf = SVC(kernel='rbf', class_weight='balanced')
# train the SVM classifier
clf.fit(X_train, Y_train)

# calculate the accuracy of the SVM classifier
y_pred_SVM = clf.predict(X_test)
acc2=accuracy_score(Y_test,y_pred_SVM)
print(f"Accuracy of SVM  classifier model is:{acc2*100}%")
print(classification_report(Y_test, y_pred_SVM))


# In[17]:


# naive bayes
from sklearn.naive_bayes import GaussianNB
clf3=GaussianNB()
clf3.fit(X_train, Y_train)
# calculate the accuracy of the SVM classifier
y_pred_NB = clf3.predict(X_test)
acc3=accuracy_score(Y_test,y_pred_NB)
print(f"Accuracy of SVM  classifier model is:{acc3*100}%")
print(classification_report(Y_test, y_pred_NB))


# In[21]:


#rf
from sklearn.ensemble import RandomForestClassifier
clf4=RandomForestClassifier(class_weight='balanced')
clf4.fit(X_train, Y_train)
# calculate the accuracy of the RF classifier
y_pred_rf = clf4.predict(X_test)
acc4=accuracy_score(Y_test,y_pred_rf)
print(f"Accuracy of RF  classifier model is:{acc4*100}%")
print(classification_report(Y_test, y_pred_rf))


# In[25]:


#decision tree
from sklearn import tree
clf5=tree.DecisionTreeClassifier(class_weight='balanced')
clf5.fit(X_test,Y_test)
y_pred_DT=clf5.predict(X_test)
acc5=accuracy_score(Y_test,y_pred_DT)
print(f"Accuracy of DT  classifier model is:{acc5*100}%")
print(classification_report(Y_test, y_pred_DT))


# In[35]:


#DEEP LEARNING
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install torch')
import tensorflow as tf


# In[47]:


#nn
import torch


# In[56]:


#create nn

class COVID_Net(nn.Module):
    def __init__(self, input_size, h_layer1, h_layer2, num_classes):
        super(COVID_Net, self).__init__()
        self.fc1 = nn.Linear(input_size, h_layer1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(h_layer1, h_layer2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(h_layer2, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

        


# In[57]:


model=COVID_Net(1999,1000,500,3)
print(model)


# In[60]:


#create dataloader
batch_size=50
file=("Downloads/covid-selected-data.csv")
# Create dataset and dataloaders
dataset = CountMatrixDataset('Downloads/covid-selected-data.csv', 'Downloads/covid-selected-data-labels.csv')
trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

