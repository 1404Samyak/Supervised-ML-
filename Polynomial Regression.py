#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# In[2]:


data=pd.read_csv('cars.csv')
data


# In[3]:


X=data[['Kilometers_Driven','Mileage','Engine','Power','Seats']]
y=data[['Price']]


# In[4]:


#First split the data into training and test dataset then apply the scaler on the train and test data separately
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scale=StandardScaler()
X_train_scaled=scale.fit_transform(X_train)
X_test_scaled=scale.transform(X_test)


# In[5]:


#creating polynomial
d=2;
poly=PolynomialFeatures(degree=d)
X_train_poly=poly.fit_transform(X_train_scaled)
X_test_poly=poly.transform(X_test_scaled)
model=LinearRegression()
model.fit(X_train_poly,y_train)


# In[6]:


y_train_predict=model.predict(X_train_poly)
y_test_predict=model.predict(X_test_poly)
y_test_predict


# In[7]:


train_mse=mean_squared_error(y_train_predict,y_train)
test_mse=mean_squared_error(y_test_predict,y_test)
print(f"The mean squared error for the training dataset is {train_mse}")
print(f"The mean squared error for the testing dataset is {test_mse}")


# In[ ]:





# In[ ]:




