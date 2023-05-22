# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 14:27:30 2022

@author: Aslı
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("eksikveriler.csv")
print(veriler)

boy = veriler[['boy']]
print(boy)

#class ve nesne tanımı
class insan:
    boy=100
    def kosmak(self,b):
        return b+10
   
ali=insan()
print(ali.boy)
print(ali.kosmak(90))    

#eksikveriler
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy='mean')
Yas=veriler.iloc[:,1:4].values
print(Yas)
imputer= imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print (Yas)

