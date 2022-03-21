import pandas as pd
import sklearn as sk
import os
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

game_results = pd.read_csv("/data/clean/game_results_clean.csv", encoding="ISO-8859-1")

game_results.head()
## fills the single missing rpi with the mean of the all the rpis
mean_value=game_results['team2_rpi'].mean()
game_results['team2_rpi'].fillna(value=game_results['team2_rpi'].mean(), inplace=True)

game_results['team2_rpi'].isnull().sum()



## linear regression with score_diff being the y variable and the others being x variables
x = game_results[['team1_osrs','team1_dsrs','team1_rpi','team2_osrs','team2_dsrs','team2_rpi']]
y = game_results['score_diff']

## creates the regression and prints the beta coefficients that we will be using
regr = LinearRegression(fit_intercept=False).fit(x,y)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

Intercept: 
 0.0
Coefficients: 
 [  1.02808354   1.11286337 -30.53191958  -0.96870805  -1.07654445
  29.23121343]

game_results.head()

