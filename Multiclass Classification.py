#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[23]:


df=pd.read_csv('Advanced_IoT_Dataset.csv')
df=df.dropna()
unique_classes=df['Class'].unique()
print(f"The unique classes are {unique_classes}")


# In[24]:


mapping={'SA':0,'SB':1,'SC':2,'TA':3,'TB':4,'TC':5}
df['Class']=df['Class'].map(mapping)
df


# In[25]:


X=df.drop(['Random','Class'],axis=1)
y=df[['Class']]
X


# In[26]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scale=StandardScaler()
X_train_scaled=scale.fit_transform(X_train)
X_test_scaled=scale.transform(X_test)


# In[27]:


model=LogisticRegression(multi_class='multinomial',solver='lbfgs',max_iter=200)
model.fit(X_train_scaled,y_train)


# In[28]:


y_train_predict=model.predict(X_train_scaled)
y_test_predict=model.predict(X_test_scaled)
y_test_predict


# In[29]:


accuracy = accuracy_score(y_test, y_test_predict)
print(f"Accuracy: {accuracy}")

print("Classification Report:")
print(classification_report(y_test, y_test_predict))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_predict))


# In[ ]:




