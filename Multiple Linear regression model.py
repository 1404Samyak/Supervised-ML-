#!/usr/bin/env python
# coding: utf-8

# In[16]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[17]:


df=pd.read_csv('fuel-consumption-ratings.csv')
df.head()


# In[25]:


X=df[['Engine size (L)','Cylinders','City (L/100 km)','Highway (L/100 km)','Combined (L/100 km)']]
#Here X is the arry of input parameters
y=df['CO2 emissions (g/km)']
X


# In[19]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scale=StandardScaler()
#First split the data into training and test dataset then apply the scaler on the train and test data separately
X_train_scaled=scale.fit_transform(X_train)
X_test_scaled=scale.transform(X_test)


# In[20]:


model=LinearRegression()
model.fit(X_train_scaled,y_train)


# In[21]:


y_train_predict=model.predict(X_train_scaled)
y_test_predict=model.predict(X_test_scaled)
y_test_predict


# In[23]:


train_mse=mean_squared_error(y_train_predict,y_train)
test_mse=mean_squared_error(y_test_predict,y_test)


# In[24]:


print(f"The mean squared error for the training dataset is {train_mse}")
print(f"The mean squared error for the testing dataset is {test_mse}")


# In[ ]:




