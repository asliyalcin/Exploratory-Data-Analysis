#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
tips=sns.load_dataset("tips")
df=tips.copy()
df.head()


# In[2]:


df["sex"].value_counts()


# In[8]:


df["smoker"].value_counts()


# In[9]:


df["day"].value_counts()


# In[10]:


df["time"].value_counts()


# In[11]:


sns.boxplot(x=df["total_bill"]);


# In[14]:


#hangi günler fazla kazanıyoruz
import seaborn as sns
tips=sns.load_dataset("tips")
df=tips.copy()
sns.boxplot(x="day", y="total_bill", data=df)


# In[ ]:


#pazar günü az kişi gelmiş yükselkikten anladık ama daha fazla bahiş bırakılmış.


# In[15]:


#sabah mı akşam mı fazla para kazanılmış
import seaborn as sns
tips=sns.load_dataset("tips")
df=tips.copy()
sns.boxplot(x="time", y="total_bill", data=df);


# In[16]:


#kişi sayısı vs kazanç
sns.boxplot(x="size",y="total_bill",data=df);


# In[18]:


sns.boxplot(x="day", y="total_bill", hue="sex", data= df);


# In[20]:


#violin
sns.catplot(x="day", y="total_bill",hue="sex", kind="violin",data=df);


# In[ ]:




