# -*- coding: utf-8 -*-
"""
Created on Mon Sep 03 21:01:28 2018

@author: Shivam Soni
"""


import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
import statsmodels.stats.outliers_influence as oi
'''df=pd.read_csv('Data_preprocessing_practice1.csv')
X=df.drop('PURCHASED',1)
Y=df.PURCHASED'''

df=pd.read_csv('Churn.csv')
X=df.drop('Churn',1)
Y=df.Churn
for c_n in X.columns:
    #print c_n
   # if X[c_n]=='object' :
    unique_cat=X[c_n].nunique()
    print ("Feature", c_n,"has", unique_cat,"unique categories")
    

#B=df['tenure'].value_counts()
'''from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()
X.iloc[:,0]=labelencoder_X.fit_transform(X.iloc[:,0])
onehotencoder=OneHotEncoder(categorical_features="all")
#next LINE ERROR:X needs to contain only non-negative integers(LETS LEAVE FOR NOW)
X= onehotencoder.fit_transform(X).toarray()'''


#todummy_list=['COUNTRY','GENDER']


todummy_list=['gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod']

for i in todummy_list:
    dummies= pd.get_dummies(X[i],prefix=i)
    #print dummies
    dummies=dummies.iloc[:,1:]
    X=X.drop(i,1)
    X=pd.concat([dummies,X],axis=1)
    #print X

'''for i in range(0,X.shape[0]):
    #print X.iloc[i,3]
    y1=X.iloc[i,3]
   # print y1
    y2=""
    for j in y1:
        if j=='-':
            break
        else:
            y2+=j
    X.iloc[i,3]=int(y2)
    #print X.iloc[i,3]'''

'''X=X.dropna()

c=X.values
c=c.T
cov=np.cov(c)
cov2 = cov.tolist()

mean=[X[i].mean() for i in X.columns]

e,f,g,h,i= np.random.multivariate_normal(mean, cov2, 1000).T
corr=np.corrcoef([e,f,g,h,i])
exog = np.array([e,f,g,h,i]).transpose()
vif0 = oi.variance_inflation_factor(exog, 0)'''




