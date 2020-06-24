# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 17:05:00 2020

@author: RADHIKA
"""
##################Bank data Assignment###########################
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
bank=pd.read_csv("D:\\ExcelR Data\\Assignments\\Logistic Regression\\bank.csv")
bank.head()
bank.columns
bank.shape
sb.countplot(x='y',data=bank, palette='hls')
plt.show()
sb.countplot(x='marital',data=bank,palette='hls')
bank.isnull().sum()
sb.countplot(x='job',data=bank,palette='hls')
sb.countplot(x='default',data=bank,palette='hls')
sb.countplot(x='education',data=bank,palette='hls')
sb.countplot(x='contact',data=bank,palette='hls')
sb.countplot(x='housing',data=bank,palette='hls')
sb.countplot(x='loan',data=bank,palette='hls')
sb.countplot(x='month',data=bank,palette='hls')
sb.countplot(x='poutcome',data=bank,palette='hls')
bank.drop(bank.columns[[0, 3, 5, 8, 9, 10, 11, 12, 13, 14, 16]], axis=1, inplace=True)
bank.columns
bank_new = pd.get_dummies(bank, columns =['job', 'marital', 'default', 'housing', 'loan', 'poutcome'])
bank_new.columns
bank_new.shape
sb.heatmap(bank_new.corr())
X=bank_new.iloc[:,1:]
Y=bank_new.iloc[:,0]
classifier=LogisticRegression()
classifier.fit(X,Y)
classifier.coef_
classifier.predict_proba(X)
y_pred = classifier.predict(X)
bank_new["y_pred"] = y_pred
y_prob = pd.DataFrame(classifier.predict_proba(X.iloc[:,:]))
new_df = pd.concat([bank_new,y_prob],axis=1)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y,y_pred)
print (confusion_matrix)
#[[40040     0]
# [    0  5171]]
#Accuracy is 100% this is best model
 ########### ROC curve ###########
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fpr, tpr, thresholds = roc_curve(Y, y_pred)

auc = roc_auc_score(Y, y_pred)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, color='orange', label='ROC')
##############################################################

#############Affairs data Assignment##############################

import pandas as pd
affairsdata=pd.read_csv("D:\\ExcelR Data\\Assignments\\Logistic Regression\\affairs.csv")
affairsdata.head()
affairsdata.columns
affairsdata.shape
affairsdata.isnull().sum()
import seaborn as sb
sb.countplot(x='affairs',data=affairsdata,palette='hls')
sb.countplot(x='gender',data=affairsdata,palette='hls')
sb.countplot(x='age',data=affairsdata,palette='hls')
sb.countplot(x='occupation',data=affairsdata,palette='hls')
affairsdata['affairs'] = (affairsdata.affairs>0).astype(int)
affairsdata.columns
affairsdata['affairs']
sb.countplot(x='affairs',data=affairsdata,palette='hls')
sb.countplot(x='yearsmarried',data=affairsdata,palette='hls')
affairsdatanew=pd.get_dummies(affairsdata,columns=['gender','children'])
affairsdatanew
affairsdatanew.shape
pd.crosstab(affairsdatanew.yearsmarried,affairsdatanew.affairs).plot(kind="bar")
sb.boxplot(x="yearsmarried",y="affairs",data=affairsdatanew,palette = "hls")
X1=affairsdatanew.iloc[:,3:]
Y1=affairsdatanew.iloc[:,0]
classifier=LogisticRegression()
classifier.fit(X1,Y1)
classifier.coef_
classifier.predict_proba(X1)
y1_pred = classifier.predict(X1)
affairsdatanew["y1_pred"] = y1_pred
y1_prob = pd.DataFrame(classifier.predict_proba(X1.iloc[:,:]))
new_df1 = pd.concat([affairsdatanew,y1_prob],axis=1)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y1,y1_pred)
print (confusion_matrix)
#[[435  16]
 #[131  19]]
 ##Accuracy is 75% this is best model
 ########### ROC curve ###########
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fpr, tpr, thresholds = roc_curve(Y1, y1_pred)

auc = roc_auc_score(Y1, y1_pred)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, color='orange', label='ROC')

