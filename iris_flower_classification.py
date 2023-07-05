
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error
import pandas as pd


# In[12]:


iris=datasets.load_iris()


# In[44]:


iris_x=iris.data


# In[30]:


iris_x_train=iris_x[:-50]
iris_x_test=iris_x[-50:]


# In[31]:


iris_y_train=iris.target[:-50]
iris_y_test=iris.target[-50:]


# In[32]:


model=linear_model.LinearRegression()


# In[33]:


model.fit(iris_x_train,iris_y_train)
iris_y_predict=model.predict(iris_x_test)


# In[38]:


print("mean squared error is:   ",mean_squared_error(iris_y_test,iris_y_predict))
print("weigths:  ",model.coef_)
print("intercet: ",model.intercept_)


# In[16]:


# plt.scatter(iris_y_test,iris_x_test)
plt.plot(iris_x_test,iris_y_predict)
