#!/usr/bin/env python
# coding: utf-8

# In[76]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[77]:


df=pd.read_csv("adult_data.csv",skipinitialspace=True)


# In[78]:


df


# In[79]:


df.shape


# In[80]:


df.head()


# In[81]:


df.tail()


# In[82]:


df.columns


# In[83]:


df.nunique()


# In[84]:


df.describe()


# In[85]:


df.info()


# In[86]:


df['workclass'].unique()


# In[87]:


df=df.replace('?',np.nan)


# In[88]:


df['workclass'].unique()


# In[89]:


df=df.drop(['fnlwgt','education-num'],axis=1)


# In[90]:


df


# In[91]:


df['education'].unique()


# In[92]:


df['education'].value_counts().plot(kind='bar')


# In[93]:


df['marital-status'].unique()


# In[94]:


df['marital-status'].value_counts().plot(kind='bar')


# In[95]:


df['occupation'].unique()


# In[96]:


df['occupation'].value_counts().plot(kind='bar')


# In[97]:


df['relationship'].unique()


# In[98]:


df['relationship'].value_counts().plot(kind='bar')


# In[99]:


df['race'].unique()


# In[100]:


df['race'].value_counts().plot(kind='bar')


# In[101]:


df['native-country'].unique()


# In[102]:


df['native-country'].value_counts().head().plot(kind='bar')


# In[103]:


df.isna().sum()


# In[104]:


df.isnull().sum()


# In[105]:


df['salary'].value_counts().plot(kind='bar')


# In[106]:


sns.displot(df['hours-per-week'])


# In[107]:


sns.boxplot(df['hours-per-week'])


# In[108]:


#removing outliers 
q1=df['hours-per-week'].quantile(0.25)
q3=df['hours-per-week'].quantile(0.75)
iqr=q3-q1


# In[109]:


lower_range=q1-(iqr*1.5)
upper_range=q3+(iqr*1.5)


# In[110]:


df.loc[df['hours-per-week'] <= lower_range, 'hours-per-week'] = lower_range
df.loc[df['hours-per-week'] >= upper_range, 'hours-per-week'] = upper_range


# In[111]:


sns.boxplot(df['hours-per-week'])


# In[112]:


df['capital-gain'].unique()


# In[113]:


df['capital-gain'] = np.where(df['capital-gain'] == 0, np.nan, df['capital-gain'])


# In[114]:


df['capital-gain'] = np.log(df['capital-gain'])


# In[115]:


df['capital-gain'] = df['capital-gain'].replace(np.nan, 0)


# In[116]:


sns.displot(df['capital-gain'])


# In[117]:


sns.boxplot(df['capital-gain'])


# In[118]:


iqr=df['capital-gain'].quantile(0.75)-df['capital-gain'].quantile(0.25)
lower_range=df['capital-gain'].quantile(0.25)-(1.5*iqr)
upper_range=df['capital-gain'].quantile(0.75)+(1.5*iqr)


# In[119]:


df.loc[df['capital-gain'] <= lower_range, 'capital-gain'] = lower_range
df.loc[df['capital-gain'] >= upper_range, 'capital-gain'] = upper_range


# In[120]:


sns.boxplot(df['capital-gain'])


# In[121]:


sns.boxplot(df['capital-loss'])


# In[122]:


iqr=df['capital-loss'].quantile(0.75)-df['capital-loss'].quantile(0.25)
lower_range=df['capital-loss'].quantile(0.25)-(1.5*iqr)
upper_range=df['capital-loss'].quantile(0.75)+(1.5*iqr)
df.loc[df['capital-loss'] <= lower_range, 'capital-loss'] = lower_range
df.loc[df['capital-loss'] >= upper_range, 'capital-loss'] = upper_range


# In[123]:


sns.boxplot(df['capital-loss'])


# In[124]:


df


# In[125]:


df['salary'].value_counts()


# In[126]:


df['salary'] = df['salary'].replace('>50K', '1')
df['salary'] = df['salary'].replace('<=50K', '0')


# In[127]:


df['salary'].value_counts()


# In[128]:


df['sex'] = np.where(df['sex'] == "Male", 1, 0)


# In[129]:


df['sex'].value_counts()


# In[130]:


label_race={value:key for key,value in enumerate(df['race'].unique())}
df['race']=df['race'].map(label_race)


# In[131]:


df['race'].value_counts()


# In[132]:


df


# In[133]:


label_na_country={value:key for key,value in enumerate(df['native-country'].unique())}
df['native-country']=df['native-country'].map(label_na_country)


# In[134]:


label_relationship={value:key for key,value in enumerate(df['relationship'].unique())}
df['relationship']=df['relationship'].map(label_relationship)


# In[135]:


label_occupation={value:key for key,value in enumerate(df['occupation'].unique())}
df['occupation']=df['occupation'].map(label_occupation)


# In[136]:


label_marital_status={value:key for key,value in enumerate(df['marital-status'].unique())}
df['marital-status']=df['marital-status'].map(label_marital_status)


# In[137]:


label_marital_status={value:key for key,value in enumerate(df['marital-status'].unique())}
df['marital-status']=df['marital-status'].map(label_marital_status)
label_workclass={value:key for key,value in enumerate(df['workclass'].unique())}
df['workclass']=df['workclass'].map(label_workclass)
label_education={value:key for key,value in enumerate(df['education'].unique())}
df['education']=df['education'].map(label_education)


# In[138]:


df


# In[139]:


df.corr()


# In[140]:


plt.figure(figsize=(10, 10))
corr = df.corr()
sns.heatmap(corr, annot=True)


# In[141]:


X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


# In[143]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[147]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[151]:


X_train.shape


# In[152]:


X_test.shape


# In[153]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[157]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print("Accuracy Score: {}".format(accuracy_score(y_test, y_pred)))
print("Confusion Matrix:\n {}".format(confusion_matrix(y_test, y_pred)))
print("Classification Report:\n {}".format(classification_report(y_test, y_pred)))


# In[159]:


from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=10)
model= RandomForestClassifier(n_estimators = 300)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy_score(y_test, predictions)


# In[ ]:




