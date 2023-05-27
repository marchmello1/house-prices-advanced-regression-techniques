#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt


# In[22]:


train_data = pd.read_csv(r'C:\Users\mohammad\Downloads\train (1).csv')
test_data = pd.read_csv(r'C:\Users\mohammad\Downloads\train (1).csv')


# In[23]:


num_rows_train = len(train_data.index)
print("num rows in training data: " + str(num_rows_train))
print("num rows in test data: " + str(len(test_data.index)))


y_train = train_data.iloc[:,-1]
train_data.drop(columns=train_data.columns[-1], 
        axis=1, 
        inplace=True)


# In[26]:


all_data = train_data.append(test_data, ignore_index=True)
print("num rows in all data: " + str(len(all_data.index)))
ids = all_data.drop('Id', axis=1)


# In[27]:


all_data = pd.get_dummies(data=all_data)


all_data = all_data.fillna(-1)

all_data.head()


# In[28]:


X_train = all_data.iloc[0:num_rows_train,:]
print("num rows in X train: " + str(len(X_train.index)))
print("num rows in y train: " + str(len(y_train.index)))
X_test = all_data.iloc[num_rows_train:,:]
print("num rows in X test: " + str(len(X_test.index)))
print("Training data features head: ")
print(X_train.head())
print()
print("Training data predictions head: ")
print(y_train.head())
print()
print("Test data features head:")
print(X_test.head())


# In[30]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# In[31]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_val_transf = scaler.transform(X_val)
X_test_transf = scaler.transform(X_test)


# In[32]:


regr = linear_model.LinearRegression()


# In[33]:


regr.fit(X_train, y_train)


# In[34]:


r2_train = regr.score(X_train,y_train)
print("R^2 on training data: " + str(r2_train))


# In[35]:


r2_validation = regr.score(X_val,y_val)
print("R^2 on validation data: " + str(r2_validation))


# In[36]:


pred_val = regr.predict(X_val)
print("Five first predictions: ")
print(pred_val[:5])
print()


# In[39]:


import math
MSE = np.square(np.subtract(y_val,pred_val)).mean()   
   
rsme = math.sqrt(MSE)  
print("Root Mean Square Error:\n")  
print(rsme)


# In[40]:


print(X_train.shape)


# In[41]:


num_components = list(range(1,289))   
list_of_pca_models = []
X_train_images = []
X_val_images = []
regression_models = []
training_rmse = []
validation_rmse = []
idx = 0
for i in num_components:
   
    list_of_pca_models.append(PCA(i))
    list_of_pca_models[idx].fit(X_train) 
    
    X_train_images.append(list_of_pca_models[idx].transform(X_train))
    X_val_images.append(list_of_pca_models[idx].transform(X_val))
    
    
    regression_model = linear_model.LinearRegression()
    regression_model.fit(X_train_images[idx], y_train)
    regression_models.append(regression_model)
    
   
    predictions_train = regression_model.predict(X_train_images[idx])
    predictions_val = regression_model.predict(X_val_images[idx])
    
 
    rmse_training = math.sqrt(np.square(np.subtract(predictions_train,y_train)).mean())
    training_rmse.append(rmse_training)
    rmse_validation =  math.sqrt(np.square(np.subtract(predictions_val,y_val)).mean()) 
    validation_rmse.append(rmse_validation)
    
    idx += 1


# In[42]:


X_test_img = list_of_pca_models[19].transform(X_test)


# In[43]:


pred_test = regression_models[19].predict(X_test_img)
print("First five predictions for test dataset: ")
print(pred_test[:5])
print()


# In[ ]:




