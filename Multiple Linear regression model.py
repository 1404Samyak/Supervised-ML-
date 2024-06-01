#!/usr/bin/env python
# coding: utf-8

# In[11]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler


# In[12]:


df=pd.read_csv('fuel-consumption-ratings.csv')
df.head()


# In[13]:


X=df[['Engine size (L)','Cylinders','City (L/100 km)','Highway (L/100 km)','Combined (L/100 km)']]
scale=StandardScaler()
X_scaled=scale.fit_transform(X)
#Here X is the arry of input parameters
y=df['CO2 emissions (g/km)']


# In[14]:


model=linear_model.LinearRegression()
#model.fit(X,y)
model.fit(X_scaled,y)


# In[15]:


X_test=np.array([4.0,5,9.9,7.3,8.7]).reshape(1,5)
X_test_scaled=scale.fit_transform(X_test)
predicted_value=model.predict(X_test_scaled)


# In[16]:


print(predicted_value)


# In[ ]:




