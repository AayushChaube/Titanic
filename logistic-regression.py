#Step 1 : Collecting the data

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
data_titanic=pd.read_excel("titanic.xlsx")
print(data_titanic.head)
print("Total number of passengers travelling : ", str(len(data_titanic)))
sexwise_list=data_titanic['sex'].tolist()
print("Total male passengers : ", sexwise_list.count('male'))
print("Total female passengers : ", sexwise_list.count('female'))

#Step 2 : Analyzing the data

fig, ax=plt.subplots(3, 2)

    #finding the number of people who survived
sns.countplot(x="survived", data=data_titanic, ax=ax[0][0])

    #finding the number of males and females who survived
sns.countplot(x="survived", hue='sex', data=data_titanic, ax=ax[0][1])

    #to find the survivors classwise
sns.countplot(x="survived", hue='pclass', data=data_titanic, ax=ax[1][0])

    #to find the survivors based on embarked
sns.countplot(x="survived", hue='embarked', data=data_titanic, ax=ax[1][1])

fig.show()

#if you wish to see all columns available
data_titanic.info()

#Step 3 : Data Wrangling

#to find out the existing null values in the data and eliminate them
print(data_titanic.isnull())
print(data_titanic.isnull().sum())
#drawing heat maps
sns.heatmap(data_titanic.isnull(), yticklabels='false', cmap='viridis', ax=ax[2][0])
#dropping columns with lot of null data

data_titanic.drop('body', axis=1, inplace=True)#inplace is boolean, when true it will replace the value and return nothing.false returns the value
data_titanic.drop('cabin', axis=1, inplace=True)
data_titanic.drop('boat', axis=1, inplace=True)
data_titanic.drop('home.dest', axis=1, inplace=True)
data_titanic.drop('age', axis=1, inplace=True)
sns.heatmap(data_titanic.isnull(), yticklabels='false', cmap='viridis', ax=ax[2][1])
print(data_titanic.isnull().sum())#this will display the null data, we will now find negligible number of null data

#to replace all non categorical data to categorical one
sex_categorical=pd.get_dummies(data_titanic['sex'])
print(sex_categorical)
sex_categorical=pd.get_dummies(data_titanic['sex'], drop_first=True)
print(sex_categorical)
#similarly changing other columns
embarked_categorical=pd.get_dummies(data_titanic['embarked'], drop_first=True)
pclass_categorical=pd.get_dummies(data_titanic['pclass'], drop_first=True)
print(embarked_categorical)
print(pclass_categorical)
#concatinating the above new columns to existing data
dat_titanic=pd.concat([data_titanic, sex_categorical, embarked_categorical, pclass_categorical], axis=1)
print(dat_titanic)
#now that new columns have been added, we will drop the existing columns, they are now reduntant
data_titanic.drop(['sex', 'embarked', 'name', 'pclass'], axis=1, inplace=True)
print(data_titanic.head(5))

#Step 4 : Testing and Training

#Now, based on the data, we need to predict if the person survived or not
#So survived becomes my dependent variable and the remaining stay as independent
y=data_titanic['survived']
x=data_titanic.drop(['survived', 'ticket'], axis=1)#ticket is dropped as it might have some data as numeric and some as non numeric
#we don't need this in independent data, so dropping
print(data_titanic.head())
from sklearn.model_selection import train_test_split #we are now splitting the data into training and testing data, so this is imported
X_train, X_test, Y_train, Y_test=train_test_split(x, y, test_size=0.33, random_state=1)
# by 0.3, we have divided the data in 70-30 ratio,
#randomstate = 1 means that the same data is to be used all the time
X_train.fillna(X_train.mean(), inplace=True) #this will replace all NAN values by their mean
Y_train.fillna(Y_train.mean(), inplace=True) #this will replace all NAN values by their mean
from sklearn.linear_model import LogisticRegression #
logmodel=LogisticRegression()
#now fitting the model
logmodel.fit(X_train, Y_train)

#now making predictions
predictions=logmodel.predict(X_test)
#now checking the accuracy of the model
from sklearn.metrics import classification_report
print(classification_report(Y_test, predictions))
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, predictions))