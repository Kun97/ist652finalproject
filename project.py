#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 03:35:57 2020

@author: kun
"""

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

data = pd.read_csv('/Users/kun/OneDrive - Syracuse University/IST 652 Scripting for Data Analysis/Project/data1.csv')

data.drop(columns = 'Unnamed: 0', inplace = True)

"""
Problem: how to win the game?
Real-world Organizational: find out the rules to win the match for E-Sports teams like TSM and C9
"""

"""
problem framing: this problem is a typical classfication problem because this is a yes or no question: win or lose.
"""

"""
Data Understanding
"""
len(data.columns) # there are total 21 columns
data.dtypes # it seems that there are 7 categorical colmuns and 14 numerical columns

# gameId
data['gameId'].describe()
data['gameId'].head()


# blueWins: categorical: just yes or no
data['blueWins'].value_counts()

# blueWardsPlaced: numerical
data['blueWardsPlaced'].describe()

# blueTotalGold: categorical: normal, few, very few, many, very many
data['blueTotalGold'].value_counts()

"""
EDA
"""
"""
# method1: a useful package: pandas_profiling
from pandas_profiling import ProfileReport
prof = ProfileReport(data)
prof.to_file(output_file = 'output.html')
"""

# method2: manual 
# overview of data
data.sample(10)
data.nunique()
data.shape #9879 rows and 21 columns

# visualization of target: blueWins
plt.bar(data['blueWins'].value_counts().index, data['blueWins'].value_counts())

cat_col = ['blueFirstBlood', 'blueTotalGold', 'blueTotalExperience', 'blueTotalMinionsKilled', 'blueTotalJungleMinionsKilled', 'blueCSPerMin', 'blueGoldPerMin']
num_col = ['blueWardsPlaced', 'blueWardsDestroyed', 'blueKills', 'blueDeaths', 'blueAssists', 'blueEliteMonsters', 'blueDragons', 'blueHeralds', 'blueTowersDestroyed', 'blueAvgLevel', 'blueGoldDiff', 'blueExperienceDiff']

# visualization of catgorical columns
fig1, ax1 = plt.subplots(len(cat_col), figsize = (10, 40))

for i, col_val in enumerate(cat_col):

    ax1[i].bar(data[col_val].value_counts().index, data[col_val].value_counts())
    ax1[i].set_title(col_val)

plt.show()
fig1.savefig(fname = 'catgorical', quality = 100, dpi = 500)

# visualization of numerical columns
fig2, ax2 = plt.subplots(len(num_col), figsize = (10,40))

for i, col_val in enumerate(num_col):

    sns.distplot(data[col_val], hist = True, ax = ax2[i])
    ax2[i].set_title('Freq dist ' + col_val, fontsize = 10)
    ax2[i].set_xlabel('')

plt.show()
fig2.savefig(fname = 'numerical', quality = 100, dpi = 500)

# visualization of correlation
corr_col = num_col
corr_col.append('blueWins')

plt.figure(figsize=(20,15))
fig3 = sns.heatmap(data[corr_col].corr(), annot = True, cmap = 'Reds')
fig3 = fig3.get_figure()
fig3.savefig(fname = 'corr', quality = 100, dpi = 500)

"""
Data Preprocessing and Preparation
"""
# just gameId, no useful meaning: delete
data.drop(columns = 'gameId', inplace = True)

# missing values
data.isnull().sum() # no missing value

# duplicated records
data.duplicated().sum() # no duplicated records

# outlier: based on report, there are no outliers for all numerical columns except blueWardsPlaced
# for blueWardsPlaced, based on fact, low records can be exists but too many can't
# delete above 99-th percentile records
p99 = np.percentile(sorted(data['blueWardsPlaced']), 99)
data = data[data['blueWardsPlaced'] < p99]


# transformation: convert categorical columns to number
data['blueWins'] = data['blueWins'].map({'Yes': 1, 'No': 0})

data['blueTotalGold'] = data['blueTotalGold'].map({'Very Few': 0, 'Few': 1, 'Normal': 2, 'Many': 3, 'Very Many': 4})

data['blueTotalExperience'] = data['blueTotalExperience'].map({'Low': 0, 'Normal': 1, 'High': 2})

data['blueTotalMinionsKilled'] = data['blueTotalMinionsKilled'].map({'Low': 0, 'Normal': 1, 'High': 2})

data['blueTotalJungleMinionsKilled'] = data['blueTotalJungleMinionsKilled'].map({'Low': 0, 'Normal': 1, 'High': 2})

data['blueCSPerMin'] = data['blueCSPerMin'].map({'Low': 0, 'Normal': 1, 'High': 2})

data['blueGoldPerMin'] = data['blueGoldPerMin'].map({'Very Few': 0, 'Few': 1, 'Normal': 2, 'Many': 3, 'Very Many': 4})

data['blueFirstBlood'] = data['blueFirstBlood'].map({'Yes': 1, 'No': 0})

data.info()

"""
Machine Learning and Statistical Modeling
"""
# train test split
from sklearn.model_selection import train_test_split
X = data.drop(columns = 'blueWins')
y = data['blueWins']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 25)


"""
supervised learning algorithm 
"""
"""Naive Bayes"""
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# fit the model
nb = GaussianNB()
nb.fit(X_train, y_train)

pred_nb = nb.predict(X_test)

# get the accuracy score
acc_nb = accuracy_score(pred_nb, y_test)
print(acc_nb)

"""Decision Tree"""
# fit the decision tree model
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
pred_tree = tree.predict(X_test)

# get the accuracy score
acc_tree = accuracy_score(pred_tree, y_test)
print(acc_tree)

"""Random Forests"""
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

acc_rf = accuracy_score(pred_rf, y_test)
print(acc_rf)

"""Logistic Regression"""
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

acc_lr = accuracy_score(pred_lr, y_test)
print(acc_lr)

"""K-nearest neighbours"""
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier() 
knn.fit(X_train,y_train) 
pred_knn = knn.predict(X_test) 

acc_knn = accuracy_score(pred_knn, y_test)
print(acc_knn)

"""
unsupervised learning algorithm
"""
"""kmeans"""
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X_train)
pred_kmeans = kmeans.predict(X_test)

pred_kmeans1 = []
for item in pred_kmeans:
    if item == 0:
    else:
        pred_kmeans1.append(0)
        
acc_kmeans = accuracy_score(pred_kmeans, y_test)
acc_kmeans1 = accuracy_score(pred_kmeans1, y_test)

"""
model performance evaluation and algorithm fine-tuning
"""
acc_result = {'Naive Bayes': [acc_nb], 'DT': [acc_tree], 'Random Forest': [acc_rf], 'Logistic Regression': [acc_lm], 'K_nearest Neighbors': [acc_knn]}
df_c = pd.DataFrame.from_dict(acc_result, orient='index', columns=['Accuracy Score'])
print(df_c)

# the most accurate is Logistic Regression so I will just fine-tuning this algorithm
from sklearn.model_selection import GridSearchCV
dual = [True, False]
max_iter = [100, 150, 200, 500, 1000]
param_grid = dict(dual = dual, max_iter = max_iter)

grid = GridSearchCV(estimator = lr, param_grid = param_grid, cv = 3, n_jobs = -1)
grid_result = grid.fit(X_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

"""
model interpretation
"""
# best result: using {'dual': False, 'max_iter': 120}
final_lr = LogisticRegression(dual = False, max_iter = 1000)
model = final_lr.fit(X_train, y_train)
final_result = final_lr.predict(X_test)

coefficients = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(np.transpose(final_lr.coef_))], axis = 1)


























