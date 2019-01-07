from nista_learn.regressions_functions import *
from nista_learn.regressions import LinearRegression
import pandas as pd


# df = pd.read_csv('50_startups.csv', header=0)
# X = df[['Administration', 'Marketing Spend', 'R&D Spend', 'Marketing Spend']].values
# Y = df['Profit'].values
# X_train = X[:45]
# Y_train = Y[:45]
# X_test = X[17:18]
# Y_test = Y[17:18]

df_test = pd.read_csv('test.csv', header=0)
df_train = pd.read_csv('train.csv', header=0)

X_train = df_train['x'].values.reshape(-1, 1)[:210]

Y_train = df_train['y'].values[:210]
X_test = df_test[['x']].values.reshape(-1, 1)[50:54]
Y_test = df_test['y'].values[50:54]


# compute_gradient(X_train, Y_train)
lr = LinearRegression()
lr.fit(X_train, Y_train, iterations=3000, learning_rate=0.01, show=True)
y_pred = lr.predict(X_test)
print('---')
print(Y_test)
print('---')
print(y_pred)
lr.plot_cost()
lr.plot_model(X_train, Y_train)
