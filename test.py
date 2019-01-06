from nista_learn.regressions_functions import *
from nista_learn.regressions import LinearRegression
import pandas as pd


df = pd.read_csv('50_startups.csv', header=0)
X = df[['Administration', 'Marketing Spend', 'R&D Spend', 'Marketing Spend']].values
Y = df['Profit'].values
X_train = X[:45]
Y_train = Y[:45]
X_test = X[8:12, :]
Y_test = Y[8:12]

lr = LinearRegression()
lr.fit(X_train, Y_train, iterations=1000, show=True)
y_pred = lr.predict(X_test)
print('---')
print(Y_test)
print('---')
print(y_pred.T)
print('scored: ', score(Y_train, Y_test))
#lr.plot_cost()
