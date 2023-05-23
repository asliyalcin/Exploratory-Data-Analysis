#!/usr/bin/env python
# coding: utf-8

# ### aykırı değerleri yakalamak

# In[28]:


import seaborn as sns
df=sns.load_dataset('diamonds')
df=df.select_dtypes(include=['float64','int64'])
df=df.dropna()
df.head()


# In[3]:


df_table=df["table"]


# In[4]:


df_table.head()


# In[5]:


sns.boxplot(x=df_table)


# In[7]:


#eşik değer belirleme
Q1=df_table.quantile(0.25)
Q3=df_table.quantile(0.75)
IQR=Q3-Q1


# In[8]:


Q1


# In[9]:


alt_sinir=Q1-1.5*IQR
ust_sinir=Q3+1.5*IQR


# In[13]:


#aykırı gözlem sorgulaması
(df_table<alt_sinir) | (df_table>ust_sinir)


# In[15]:


aykiri_tf=(df_table<alt_sinir)
aykiri_tf.head()


# In[16]:


df_table[aykiri_tf]


# In[17]:


df_table[aykiri_tf].index


# ### silme

# In[19]:


import pandas as pd
type(df_table)


# In[22]:


df_table=pd.DataFrame(df_table)


# In[24]:


t_df=df_table[~((df_table<(alt_sinir)) | (df_table >(ust_sinir))).any(axis=1)]


# In[25]:


t_df # ~ : sağ tarafındaki koşulun dışındakileri getir


# ### ortalama ile doldurma

# In[7]:


import seaborn as sns
df=sns.load_dataset('diamonds')
df=df.select_dtypes(include=['float64','int64'])
df=df.dropna()
df.head()


# In[31]:


df_table=df["table"]


# In[32]:


aykiri_tf.head()


# In[34]:


df_table[aykiri_tf]


# In[35]:


df_table[aykiri_tf]


# In[38]:


df_table[aykiri_tf]=df_table.mean()


# In[39]:


df_table[aykiri_tf]


# ### baskılama yöntemi

# In[8]:


import seaborn as sns
df=sns.load_dataset('diamonds')
df=df.select_dtypes(include=['float64','int64'])
df=df.dropna()
df.head()


# In[40]:


df_table[aykiri_tf]


# In[42]:


df_table=df["table"]


# In[43]:


df_table[aykiri_tf]


# In[44]:


df_table[aykiri_tf]= alt_sinir


# In[45]:


df_table[aykiri_tf]


#  ### Çok Değişkenli Aykırı Gözlem Analizi

# In[46]:


import seaborn as sns
df=sns.load_dataset('diamonds')
df=df.select_dtypes(include=['float64','int64'])
df=df.dropna()
df.head()


# In[47]:


import numpy as np
from sklearn.neighbors import LocalOutlierFactor


# In[49]:


clf=LocalOutlierFactor (n_neighbors= 20, contamination=0.1)


# In[51]:


clf.fit_predict(df)


# In[52]:


df_scores= clf.negative_outlier_factor_


# In[53]:


df_scores[0:10]


# In[57]:


np.sort(df_scores)[0:20]


# In[59]:


esik_deger=np.sort(df_scores)[10] 


# In[60]:


aykiri_tf=df_scores>esik_deger


# In[61]:


aykiri_tf


# In[62]:


yeni_df=df[df_scores>esik_deger]


# In[ ]:


#silme yöntemi


# In[63]:


yeni_df #aykırı olmayan değerler


# In[65]:


df[df_scores<esik_deger] #aykırı değerler


# In[74]:


df[df_scores==esik_deger]


# In[75]:


baski_deger=df[df_scores==esik_deger]


# In[81]:


aykirilar= df[~aykiri_tf] 


# In[82]:


aykirilar.to_records(index=False)


# In[83]:


res= aykirilar.to_records(index=False)


# In[84]:


res[:]=baski_deger.to_records(index=False)


# In[85]:


res


# In[86]:


df[~aykiri_tf]=pd.DataFrame(res,index=df[~aykiri_tf].index)


# In[87]:


df[~aykiri_tf] #aykırı gözleme eşik değerdeki verileri atadık


# ### Eksik Veri Analizi

# In[1]:


import numpy as np
import pandas as pd
V1=np.array ([1,3,6,np.NaN,9,15])
V2=np.array ([7,np.NaN,5,8,12,np.NaN,2,3])
V3=np.array ([np.NaN,12,5,6,14,7,np.NaN,2,31])
df = pd.DataFrame(
    {"V1":V1,
     "V2":V2,
     "V3":V3}
)
df


# In[ ]:


df[df.isnull().any(Axis=1)] # en az 1 tane null olan saturları getirir
df[df[df.notnull().all(Axis=1)] #hepsi dolu olanlar


# In[ ]:


df["column_name"].fillna(df["column_name"].mean()]
df.apply(lambda x:x.fillna(x.mean()),axis=0) #tüm eksik değerleri ortaamasıyla doldrurdu


# ### eksik veri görselleştirmesim

# In[4]:


get_ipython().system('pip install missingno')


# In[9]:


import missingno as msno
msno.bar(df)


# In[ ]:




