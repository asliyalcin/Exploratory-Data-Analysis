#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
ad=pd.read_csv("Advertising.csv",usecols=[1,2,3,4] )
df=ad.copy()
df.head()


# In[4]:


X=df.drop("sales",axis=1) #bagımsız değişkenler


# In[5]:


X[0:10]


# In[7]:


y=df["sales"]


# In[11]:


from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict


# In[14]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)


# In[15]:


X_train.shape


# In[16]:


X_test.shape


# In[18]:


training=df.copy()


# In[19]:


training.shape


# In[24]:


##statsmodels##


# In[30]:



import statsmodels.api as sm

lm=sm.OLS(y_train,X_train)


# In[31]:


model= lm.fit()
model.summary()


# In[32]:


##scikit-learn model


# In[34]:


from sklearn.linear_model import LinearRegression
lm=LinearRegression()


# In[35]:


model=lm.fit(X_train,y_train)


# In[36]:


model.intercept_ #sabit katsayı


# In[37]:


model.coef_


# In[38]:


yeni_veri=[[30],[10],[40]]  #30 birim tv harcaması,10 birim radio,50birim gazete harcaması vs tahmini satış değeri


# In[39]:


yeni_veri=pd.DataFrame(yeni_veri).T


# In[40]:


model.predict(yeni_veri)


# In[48]:


import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
rmse=np.sqrt(mean_squared_error(y_train,model.predict(X_train)))


# In[49]:


rmse #eğitim hatası


# In[51]:


rmse=np.sqrt(mean_squared_error(y_test,model.predict(X_test)))


# In[52]:


rmse# test hatası


# In[53]:


cross_val_score(model,X,y,cv=10,scoring="r2").mean()


# In[54]:


cross_val_score(model,X_train,y_train,cv=10,scoring="r2").mean()


# In[57]:


-cross_val_score(model
                ,X_train,y_train,
                cv=10,scoring="neg_mean_squared_error").mean()


# In[58]:


-cross_val_score(model
                ,X_train,y_train,
                cv=10,scoring="neg_mean_squared_error")


# In[60]:


np.sqrt(-cross_val_score(model
                ,X_train,y_train,
                cv=10,scoring="neg_mean_squared_error")).mean() #rms değeri


# In[ ]:




