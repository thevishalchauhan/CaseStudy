# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('data/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
#!pip install lazypredict

# %%
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import plotly.express as px


# %%
df=pd.read_csv('data/train.csv')
df.head()

# %%
df.shape

# %%
df.info()

# %%
df.dtypes

# %%
px.histogram(df,x='HomePlanet',color='Transported',barmode='group')

# %%
px.histogram(df,x='CryoSleep',color='Transported',barmode='group')

# %%
px.histogram(df,x='Destination',color='Transported',barmode='group')

# %%
px.histogram(df,x='VIP',color='Transported',barmode='group')

# %%
#cols=df.select_dtypes('object').columns.tolist()
#for i in cols:
#    ct=df[i].value_counts()
 #   plt.title(i);
 #   ct.plot(kind='bar')
 #   plt.figure(figsize=(8,5));
    
 #   plt.show();

# %%
cols=df.select_dtypes('object').columns
cols.tolist()

# %%
df['HomePlanet'].fillna(df['HomePlanet'].value_counts().index[0],inplace=True)


# %%
def missingvalue(df):
    cols=df.select_dtypes('object').columns
    cols=cols.tolist()
    #for i in cols:
    df['HomePlanet'].fillna(df['HomePlanet'].value_counts().index[0],inplace=True)
    df['CryoSleep'].fillna(df['CryoSleep'].value_counts().index[0],inplace=True)
    df['Destination'].fillna(df['Destination'].value_counts().index[0],inplace=True)
    df['VIP'].fillna(df['VIP'].value_counts().index[0],inplace=True)


    cols1=df.select_dtypes('float64').columns
    cols1=cols1.tolist()
    for i in cols1:
        df[i]=df[i].fillna(df[i].mean())
    return df

# %%
def Onehotencoding(df1):
    df1=df1.join(pd.get_dummies(df['HomePlanet'],prefix='HomePlanet',prefix_sep='_'))
    df1=df1.join(pd.get_dummies(df['CryoSleep'],prefix='CryoSleep',prefix_sep='_'))
    df1=df1.join(pd.get_dummies(df['Destination'],prefix='Destination',prefix_sep='_'))
    df1=df1.join(pd.get_dummies(df['VIP'],prefix='VIP',prefix_sep='_'))
    df1.drop(['HomePlanet','CryoSleep','Destination','VIP'],axis=1,inplace=True)
    return df1

    

# %%
def pre_processing(df):
    df.drop(['PassengerId','Name','Cabin'],axis=1,inplace=True)
    df=missingvalue(df)
    #df1=df.apply(lambda x:x.fillna(x.value_counts().index[0]))
    #print(df1.info())
    cols=df.select_dtypes('object').columns.tolist()
    df=Onehotencoding(df)
    #for i in cols:
    #    df1=df.join(pd.get_dummies(df[i],prefix=i,prefix_sep='_'))
    #df1.drop(cols,axis=1,inplace=True)
    #print(df1.info())
    #scaler=StandardScaler()
    #scaled=scaler.fit_transform(df1)
    #df2=pd.DataFrame(scaled,index=df1.index,columns=df1.columns)
    return df
    

# %%
#def transform1(df):
#    scaler=StandardScaler()
#    scaled=scaler.fit_transform(df)
#    df1=pd.DataFrame(scaled,index=df.index,columns=df.columns)
#    return df1

# %%
df1=pre_processing(df)

# %%
df1

# %%
df1.info()

# %%
y=df1['Transported']
col=df1.columns

col=col.delete(6)
x=df1[col]
#x=transform1(x)

# %%
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=5,stratify = y,test_size = 0.40)


# %%
clf=LazyClassifier()
model,predictions=clf.fit(x_train,x_test,y_train,y_test)

# %%
print(model)

# %% [markdown]
# ### LGBMClassifier has better performance compared to other classifiers

# %%
clf=lgb.LGBMClassifier(random_state=5)
clf.fit(x_train,y_train)
pred=clf.predict(x_test)

# %%
clf.get_params()

# %%
#acc=accuracy_score(y_test,pred)
#acc
print(clf.score(x_train,y_train))
print(clf.score(x_test,y_test))

# %% [markdown]
# ### Check if the model does not overfit

# %%
y_pred_train=clf.predict(x_train)

# %%
print('{0:0.4f}'.format(accuracy_score(y_train,y_pred_train)))

# %% [markdown]
# ### The accuracy of train and test set are comparable**

# %% [markdown]
# ### Hyperparameter tuning

# %%
param_grid={'max_bin':[150,250],'learning_rate':[0.13,0.03],'num_iterations':[150,300],'min_gain_to_split':[0.1,1],'max_depth':[10,20]}
clf=RandomizedSearchCV(estimator=clf,param_distributions=param_grid, scoring='accuracy')
search=clf.fit(x_train,y_train)
search.best_params_

# %%
search.best_score_

# %%
clf=lgb.LGBMClassifier(max_bin=250,learning_rate=0.03,num_iterations=150,min_gain_to_split=1,max_depth=20)
clf.fit(x_train,y_train)
pred=clf.predict(x_test)
#acc=accuracy_score(y_test,pred)
#acc
print("Accuracy on train data",clf.score(x_train,y_train))
print("Accuracy on test data",clf.score(x_test,y_test))

# %%
clf=lgb.LGBMClassifier(max_bin=250,learning_rate=0.13,num_iterations=150,min_gain_to_split=0.3,max_depth=20)
clf.fit(x_train,y_train)
pred=clf.predict(x_test)
#acc=accuracy_score(y_test,pred)
#print("accuracy",acc)
print("Accuracy on train data",clf.score(x_train,y_train))
print("Accuracy on test data",clf.score(x_test,y_test))

# %%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)

# %%
cm

# %%
import seaborn as sns
sns.heatmap(cm,annot=True)

# %%
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))

# %%
df2=pd.read_csv('data/test.csv')
df2.head()

# %%
df2.shape

# %%
df2.columns

# %%
sub=pd.DataFrame(df2['PassengerId'])
sub

# %%
df3=pre_processing(df2)
df3.head()

# %%
df3.isnull().sum()

# %%
pred1=clf.predict(df3)

# %%
pred1

# %%
sub['Transported']=pred1

# %%
sub['Transported'].value_counts()

# %%
sub.to_csv('submission.csv',index=False)


