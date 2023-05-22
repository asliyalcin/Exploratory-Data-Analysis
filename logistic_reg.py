#!/usr/bin/env python
# coding: utf-8

# In[18]:


get_ipython().system(' pip3 install statsmodels')


# In[19]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score #classification_rep
from sklearn.metrics import roc_auc_score,roc_curve
import statsmodel.formula.api as sms
import matplotlib.pyplot as plt
from sklearn.neigbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGMClassifier
from catboost import CatBoostClassifier
from warnings import filterwarnings
filterwarnings ('ignore')


# In[22]:


diabetes = pd.read_csv("diabetes.csv")


# 
# # model

# In[47]:


df=diabetes.copy()
df.dropna(inplace=True)
df.head()


# In[48]:


df["Outcome"].value_counts().plot.barh()


# In[49]:


df.describe().T


# In[50]:


y=df["Outcome"]
X=df.drop(["Outcome"],axis=1)


# In[51]:


log=sm.Logit(y,X)
log_model=log.fit()


# In[52]:


log_model.summary()


# In[53]:


from sklearn.linear_model import LogisticRegression
log=LogisticRegression(solver="liblinear")
log_model=log.fit(X,y)
log_model


# In[54]:


log_model.intercept_


# In[55]:


log_model.coef_


# # model tuning

# In[56]:


y_pred=log_model.predict(X)


# In[57]:


confusion_matrix(y,y_pred)


# In[58]:


accuracy_score(y,y_pred)


# In[59]:


from sklearn.metrics import classification_report
print(classification_report(y,y_pred))


# In[60]:


log_model.predict_proba(X)[0:20] #probabilities for being (0,1)


# In[61]:


log_model.predict(X)[0:5] #prediction values


# In[62]:


y[0:5] #real values 


# In[63]:


y_probs=log_model.predict_proba(X)
y_probs=y_probs[:,1]
y_pred= [1 if i>0.5 else 0 for i in y_probs]


# In[64]:


y_pred[0:5]


# In[65]:


print(classification_report(y,y_pred))


# In[72]:


logit_roc_auc=roc_auc_score(y, log_model.predict(X))

from matplotlib import pyplot as plt

import matplotlib.pyplot as plt
fpr,tpr,tresholds=roc_curve(y,log_model.predict_proba(X)[:,1])
plt.figure()
plt.plot (fpr,tpr,label='AUC(area=%0.2f)'% logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('false positive ratio')
plt.ylabel('true positive ratio')
plt.title('ROC')
plt.show()


# # test-train split

# In[73]:


X_train,X_test,y_train, y_test=train_test_split(X,y,test_size=0.25,random_state=42)


# In[74]:


log=LogisticRegression(solver="liblinear")
log_model=log.fit(X_train,y_train)
log_model


# In[75]:


accuracy_score(y_test,log_model.predict(X_test))


# In[77]:


cross_val_score(log_model,X_test,y_test,cv=10).mean()


# In[ ]:




