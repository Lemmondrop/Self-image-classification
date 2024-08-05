#!/usr/bin/env python
# coding: utf-8

# # Pine species taxonomy codes using labeling techniques

# ## Linear Regression

# In[ ]:


import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[ ]:


df = pd.read_csv('./labeling csv file path')


# In[ ]:


df.corr()


# In[ ]:


df.columns


# In[ ]:


X = df['Name of a specific column']
y = df['Name of a specific column']


# In[ ]:


print(len(X))
print(len(y))


# In[ ]:


plt.plot(X, y , 'o')
plt.show()


# In[ ]:


line_fitter = LinearRegression()
line_fitter.fit(X.values.reshape(-1, 1), y)


# In[ ]:


line_fitter.predict([[10]])


# In[ ]:


line_fitter.coef_


# In[ ]:


# 기존 X 값으로 y를 예측, 도식화
plt.plot(X, y, 'o')
plt.plot(X, line_fitter.predict(X.values.reshape(-1,1)))
plt.show()


# In[ ]:


line_fitter


# ## Logistic Regression

# In[ ]:


df = pd.read_csv('./labeling csv file path')
print(df.shape)
print(df.head(5))


# In[ ]:


df['name'].unique()


# In[ ]:


df['name'] = df['name'].map({'Categories 1':1, 'Categories 2':2, 'Categories 3':3,
       'Categories 4':4})


# In[ ]:


df.columns


# In[ ]:


features = df[['file coulumns']]
name = df['name']


# In[ ]:


train_features, test_features, train_labels, test_labels = train_test_split(features, name)


# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_fetures = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)


# In[ ]:


from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()
log_model.fit(train_features, train_labels)


# In[ ]:


print(log_model.score(train_features, train_labels))


# In[ ]:


print(log_model.score(test_features, test_labels))


# In[ ]:


log_model.coef_[0]


# In[ ]:


cols = pd.DataFrame(x_train).columns.tolist()
y_pos = np.arange(len(cols))

plt.rcParams['figure.figsize'] = [5, 4]
fig, ax = plt.subplots()
ax.barh(y_pos, log_model.coef_[0], align='center', ecolor='black')
plt.rc('font', family='Malgun Gothic')
ax.set_yticks(y_pos)
ax.set_yticklabels(cols)
ax.invert_yaxis()
ax.set_xlabel('Coef')
ax.set_title("Each Feature's Importance")

plt.show()

