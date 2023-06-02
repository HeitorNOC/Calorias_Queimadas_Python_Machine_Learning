import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor

calorias = pd.read_csv('calories.csv')
exercicio_data = pd.read_csv('exercise.csv')

calorias_data = pd.concat([exercicio_data, calorias['Calories']], axis=1)

calorias_data.isnull().sum()

calorias_data.describe()

sns.set()

sns.countplot(x=calorias_data['Gender'])
sns.displot(calorias_data['Age'], kde=True)
sns.displot(calorias_data['Height'], kde=True)
sns.displot(calorias_data['Weight'], kde=True)

correlacao = calorias_data.corr()

plt.figure(figsize=(10,10))
sns.heatmap(correlacao, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')

calorias_data.replace({"Gender":{'male':0,'female':1}}, inplace=True)

X = calorias_data.drop(columns=['User_ID','Calories'], axis=1)
Y = calorias_data['Calories']

print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

model = XGBRegressor()

model.fit(X_train, Y_train)

test_data_prediction = model.predict(X_test)
print(test_data_prediction)

absolute_error = metrics.mean_absolute_error(Y_test, test_data_prediction)
print("Mean Absolute Error = ", absolute_error)