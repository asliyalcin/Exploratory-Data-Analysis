#!/usr/bin/env python
# coding: utf-8

# In[2]:


import seaborn as sns
planets =sns.load_dataset("planets")


# In[3]:


planets.head()


# In[10]:


df=planets.copy()


# In[11]:


df.head()


# In[12]:


df.info()


# In[13]:


import pandas as pd
df.method= pd.Categorical(df.method) # tipini değiştirdik


# In[14]:


df.dtypes


# In[15]:


import seaborn as sns
planets =sns.load_dataset("planets")
df=planets.copy()
df.isnull().values.any() # eksik değer var mı


# In[19]:


df.isnull().sum()


# In[4]:


df["orbital_period"].fillna(df.orbital_period.mean(), inplace=True)


# In[5]:


df.isnull().sum()


# In[7]:


df.fillna(df.mean(), inplace=True)


# In[8]:


df.isnull().sum()


# In[11]:


kat_df= df.select_dtypes(include=["object"])


# In[12]:


kat_df.head()


# In[13]:


df_num=df.select_dtypes(include=["float64", "int64"])


# In[14]:


df_num.head()


# In[15]:


df_num.describe().T


# In[18]:


df_num["distance"].describe()


# In[20]:


mucevherler=sns.load_dataset("diamonds")
df=mucevherler.copy()
df.head()


# In[21]:


(df["cut"]
 .value_counts()
 .plot.barh()
 .set_tittle("cut değişkeninin sınıf frekansları")); #pandasla barplot


# In[23]:


sns.barplot(x="cut", y=df.cut.index, data=df); #seaborn ile barplot


# In[20]:


import seaborn as sns
from pandas.api.types import CategoricalDtype
mucevherler=sns.load_dataset("diamonds")
df=mucevherler.copy()
cut_kategori=["Fair","Good","Very Good","Premium", "Ideal"]
df.cut=df.cut.astype(CategoricalDtype(categories=cut_kategori, ordered=True))
df.head()


# In[23]:


import seaborn as sns
from pandas.api.types import CategoricalDtype
mucevherler=sns.load_dataset("diamonds")
df=mucevherler.copy()
df.dtype()


# In[7]:


sns.catplot(x="cut", y="price", data=df);


# In[20]:


sns.barplot(x="cut", y="price",hue="color", data=df);


# In[22]:


df.groupby(["cut","color"]) ["price"].mean()


# In[23]:


sns.kdeplot(df.price, shade=True);


# In[29]:



(sns
 .FacetGrid(df,
           hue="cut",
           height=5,
           xlim=(0,10000))
 .map(sns.kdeplot, "price", shade=True)
 .add_legend())


# In[30]:


sns.catplot(x="cut", y="price", hue="color",kind="point", data=df);


# In[31]:


get_ipython().run_line_magic('pinfo', 'sns.catplot')


# In[2]:


import seaborn as sns
mucevherler=sns.load_dataset("diamonds")
df=mucevherler.copy()
df.head()


# In[4]:


sns.displot(df.price,kde=True);


# In[5]:


df["cut"].value_counts()


# In[10]:


df_num=df.(["float64","int64"])
df_num


# In[ ]:





# In[ ]:




