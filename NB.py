# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 12:00:00 2018

@author: Shivam Soni
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 17:07:35 2018

@author: Shivam Soni
"""

import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
import statsmodels.stats.outliers_influence as oi

df=pd.read_csv('Churn.csv')

for i in df.columns:
    df[i]=df[i].replace(" ",np.NaN)
    
#print (df.isnull().sum())

df.dropna(inplace=True)
df = df.reset_index()[df.columns]
#print (df.isnull().sum())
def tenure_lab(t) :
    
    if t <= 12 :
        return "Tenure_0-12"
    elif (t > 12) & (t <= 24 ):
        return "Tenure_12-24"
    elif (t > 24) & (t <= 48) :
        return "Tenure_24-48"
    elif (t > 48) & (t <= 60) :
        return "Tenure_48-60"
    elif t > 60 :
        return "Tenure_gt_60"

df["tenure"]=df["tenure"].map(tenure_lab)


#since we have 72 categories in tenure we will reduce the number of categories in it
#therefoe we made above function and to check how many categories each column has now,we are using the following loop

for c_n in df.columns:
    #print c_n
   # if X[c_n]=='object' :
    unique_cat=df[c_n].nunique()
    #print ("Feature", c_n,"has", unique_cat,"unique categories")


X=df.drop('Churn',1)
Y=df.Churn

X=X.drop('customerID',1)

todummy_list  =X.nunique()[X.nunique() < 6].keys().tolist()

num_cols   = [x for x in X.columns if x not in todummy_list]

X_org=X.copy()

for i in todummy_list:
    dummies= pd.get_dummies(X[i],prefix=i)
    #print dummies
    dummies=dummies.iloc[:,1:]
    X=X.drop(i,1)
    X=pd.concat([dummies,X],axis=1)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
scaled = std.fit_transform(X[num_cols])
scaled = pd.DataFrame(scaled,columns=num_cols)

X=X.drop(num_cols,1)
print (X.isnull().sum())

#HAVE TO READ ABOUT "merge()"
#X=pd.concat([X,scaled ],axis=1)
X = X.merge(scaled,left_index=True,right_index=True,how = "left")

print (X.isnull().sum())

le = LabelEncoder()
Y= le.fit_transform(Y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
m=gnb.fit(x_train, y_train)

# all parameters not specified are set to their defaults
predictions = m.predict(x_test)
from sklearn import metrics
mett=metrics.classification_report(y_test,predictions)
print (mett)
p=[]

cm = metrics.confusion_matrix(y_test, predictions)
print(cm)


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test, predictions)


#plot no skill
from matplotlib import pyplot
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')
# show the plot
pyplot.show()

auc = roc_auc_score(y_test, predictions)
print('AUC: %.3f' % auc)
probs = m.predict_proba(x_test)
probs = probs[:, 1]


fpr1, tpr1, thresholds1 = roc_curve(y_test, probs)

#plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr1, tpr1, marker='.')
# show the plot
pyplot.show()

auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)


import matplotlib.pyplot as plt
import seaborn as sns

score = gnb.score(x_test, y_test)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);






 