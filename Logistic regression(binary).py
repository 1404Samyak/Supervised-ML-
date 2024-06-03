#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


# In[52]:


df=pd.read_csv('Airline_customer_satisfaction.csv')
#first we have to do some modifications on the existing dataset
df=df.dropna()
df


# In[53]:


mapping={'dissatisfied':0,'satisfied':1}
df['satisfaction']=df['satisfaction'].map(mapping)
df


# In[54]:


X=df.drop(['Customer Type','Type of Travel','Class','satisfaction'],axis=1)
y=df[['satisfaction']]
y


# In[55]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train


# In[56]:


scale=StandardScaler()
X_train_scaled=scale.fit_transform(X_train)
X_test_scaled=scale.transform(X_test)


# In[57]:


model=LogisticRegression()
model.fit(X_train_scaled,y_train)


# In[58]:


y_train_predict=model.predict(X_train_scaled)
y_test_predict=model.predict(X_test_scaled)
y_test_predict


# In[59]:


train_accuracy = accuracy_score(y_train, y_train_predict)
test_accuracy = accuracy_score(y_test, y_test_predict)
train_precision = precision_score(y_train, y_train_predict)
test_precision = precision_score(y_test, y_test_predict)
train_recall = recall_score(y_train, y_train_predict)
test_recall = recall_score(y_test, y_test_predict)
train_f1 = f1_score(y_train, y_train_predict)
test_f1 = f1_score(y_test, y_test_predict)
test_roc_auc = roc_auc_score(y_test, y_test_predict)

print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")
print(f"Training Precision: {train_precision}")
print(f"Testing Precision: {test_precision}")
print(f"Training Recall: {train_recall}")
print(f"Testing Recall: {test_recall}")
print(f"Training F1 Score: {train_f1}")
print(f"Testing F1 Score: {test_f1}")
print(f"Testing ROC-AUC Score: {test_roc_auc}")

# Optional: Confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_predict)
print(f"Confusion Matrix:\n{conf_matrix}")


# In[ ]:




