#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
ad=pd.read_csv("Advertising.csv",usecols=[1,2,3,4] )
df=ad.copy()
df.head()


# In[8]:


df.describe().T


# In[10]:


df.isnull().values.any()


# In[11]:


df.corr()


# In[12]:


import seaborn as sns
sns.pairplot(df,kind="reg")


# In[13]:


sns.jointplot(x="TV",y="sales",data=df,kind="reg")


# In[15]:


import statsmodels.api as sm
X=df[["TV"]]
X[0:5]


# In[16]:


X=sm.add_constant(X)
X[0:5]


# In[17]:


y=df["sales"]
y[0:5]


# In[18]:


lm=sm.OLS(y,X)


# In[19]:


model=lm.fit()


# In[20]:


model.summary()


# In[22]:


model.params


# In[23]:


model.summary().tables[1]


# In[24]:


model.conf_int()


# In[25]:


model.f_pvalue


# In[27]:


print("f_value","%.4f"%model.f_pvalue)


# In[29]:


print("fvalue","%.4f"%model.fvalue)


# In[31]:


model.mse_model #hata kareler ort,  


# In[32]:


model.fittedvalues[0:5]


# In[34]:


#model denklemi yazma
print("sales="+str("%.2f" %model.params[0]) +"+ TV"+"*"+str("%.2f"%model.params[1])) 


# In[47]:


#modelin görselleşmesi
import seaborn as sns
g= sns.regplot(df["TV"],df["sales"],ci=None,scatter_kws={'color':'r','s':9})
g.set_title("model denklemi: sales=7.03+ TV*0.05")
g.set_ylabel("satış sayısı")
g.set_xlabel("tv harcamaları")
plt.xlim(-10,310)
plt.ylim(bottom=0);


# In[48]:


from sklearn.linear_model import LinearRegression


# In[50]:


X=df[["TV"]]
y=df["sales"]
reg=LinearRegression()
model=reg.fit(X,y)
model.intercept_
model.coef_


# In[51]:


model.score(X,y)


# In[52]:


model.predict(X)[0:10]


# In[54]:


model.predict([[30]]) #30 biri tv harcaması olduğunda satışların tahmini değeri


# In[56]:


yeni_veri=[[5],[90],[200]]
model.predict(yeni_veri)


# In[63]:


from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.formula.api as smf


# In[64]:


lm=smf.ols("sales~TV",df)
model=lm.fit()


# In[65]:


mse=mean_squared_error(y,model.fittedvalues)


# In[66]:


mse #hata kareler ortalaması değeri


# In[68]:


import numpy as np
rmse=np.sqrt(mse)


# In[69]:


rmse


# In[70]:


reg.predict(X)[0:10]


# In[73]:


k_t=pd.DataFrame({"gercek_y":y[0:10],
                  "tahmin_y":reg.predict(X)[0:10]
                 })


# In[74]:


k_t


# In[81]:


k_t["hata"]=k_t["gercek_y"]-k_t["tahmin_y"]


# In[76]:


k_t


# In[84]:


k_t["hata_kare"]=k_t["hata"]**2


# In[85]:


k_t


# In[86]:


np.sum(k_t["hata_kare"])


# In[87]:


np.mean(k_t["hata_kare"])


# In[88]:


np.sqrt(np.mean(k_t["hata_kare"])) #hata kare ortlamasının karekök değeri


# In[89]:


model.resid[0:10]


# In[91]:


import matplotlib.pyplot as plt
plt.plot(model.resid)   #hataların yayılımı


# In[ ]:




