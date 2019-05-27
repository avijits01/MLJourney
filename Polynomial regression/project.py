import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_squared_error
#importing the data
test_set_Y=test_set_Y["medv"]
training_set=pd.read_csv("train.csv")
training_set.head()
sns.set()
X=training_set.drop('medv',axis=1) #by default axis=0
Y=training_set['medv']


#splitting trainig into training and validation set
X_train,X_valid,Y_train,Y_valid=train_test_split(X,Y,test_size=0.3)

#base tests. Simple Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, Y_train)

print('Training score: {}'.format(lr_model.score(X_train, Y_train)))
print('Valid score: {}'.format(lr_model.score(X_valid, Y_valid)))
y_pred = lr_model.predict(X_valid)
mse = mean_squared_error(Y_valid, y_pred)
rmse = math.sqrt(mse)

print('RMSE: {}'.format(rmse))

#pipeline lets you use different sklearn objects in one go
#using min max scaling. Poly degree of 2
steps=[('scalar',MinMaxScaler()),('poly',PolynomialFeatures(degree=2)),('model',LinearRegression())]
pipeline=Pipeline(steps)
pipeline.fit(X_train,Y_train)
print('Training score: {}'.format(pipeline.score(X_train, Y_train)))
print('Test score: {}'.format(pipeline.score(X_valid, Y_valid)))

#Poly degree of 3 and 4 producing undesirable results
#Clearly a case of overfitting in poly 2. No over fitting as test set score=valid set score
#Regularisation L2
#used both standard scalar and min max scalar
steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Ridge(alpha=10)) #less than one meaning slightly overfitting the variables
      ]

pipeline = Pipeline(steps)

pipeline.fit(X_train, Y_train)

print('Training score: {}'.format(pipeline.score(X_train, Y_train)))
print('Test score: {}'.format(pipeline.score(X_valid, Y_valid)))

#Using MinMaxScaler

steps = [
    ('scalar', MinMaxScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Ridge(alpha=10))
     ]

lasso_pipe = Pipeline(steps)

lasso_pipe.fit(X_train, Y_train)

print('Training score: {}'.format(lasso_pipe.score(X_train, Y_train)))
print('Test score: {}'.format(lasso_pipe.score(X_valid, Y_valid)))

steps = [
    ('scalar', MinMaxScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Ridge(alpha=10))
     ]

lasso_pipe = Pipeline(steps)

lasso_pipe.fit(X_train, Y_train)

print('Training score: {}'.format(lasso_pipe.score(X_train, Y_train)))
print('Test score: {}'.format(lasso_pipe.score(X_valid, Y_valid)))
