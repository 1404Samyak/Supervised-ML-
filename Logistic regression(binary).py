#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# In[43]:


df=pd.read_csv('Airline_customer_satisfaction.csv')
#first we have to do some modifications on the existing dataset
df=df.dropna()
df


# In[44]:


mapping={'dissatisfied':0,'satisfied':1}
df['satisfaction']=df['satisfaction'].map(mapping)
df


# In[45]:


X=df.drop(['Customer Type','Type of Travel','Class','satisfaction'],axis=1)
y=df[['satisfaction']]
y


# In[46]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train


# In[47]:


scale=StandardScaler()
X_train_scaled=scale.fit_transform(X_train)
X_test_scaled=scale.transform(X_test)


# In[48]:


model=LogisticRegression()
model.fit(X_train_scaled,y_train)


# In[49]:


y_train_predict=model.predict(X_train_scaled)
y_test_predict=model.predict(X_test_scaled)
y_test_predict


# In[50]:


train_mse=mean_squared_error(y_train_predict,y_train)
test_mse=mean_squared_error(y_test_predict,y_test)
print(f"The mean squared error for the training dataset is {train_mse}")
print(f"The mean squared error for the testing dataset is {test_mse}")


# In[ ]:




