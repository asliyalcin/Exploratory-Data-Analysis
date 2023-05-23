#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[6]:


diabetes = pd.read_csv("diabetes.csv")
df=diabetes.copy()
df=df.dropna()
y=df["Outcome"]
X=df.drop(["Outcome"],axis=1)
X_train, X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=42)


# In[7]:


from sklearn.naive_bayes import GaussianNB


# In[8]:


nb=GaussianNB()
nb_model=nb.fit(X_train,y_train)


# In[9]:


nb_model.predict(X_test)[0:10]


# In[10]:


nb_model.predict_proba(X_test)[0:10] #probabilities values


# In[11]:


y_pred=nb_model.predict(X_test)


# In[12]:


accuracy_score(y_test,y_pred)


# In[15]:


cross_val_score(nb_model, X_test,y_test,cv=10).mean()


# In[ ]:




