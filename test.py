from nista_learn.regressions_functions import *
from nista_learn.regressions import LinearRegression, LogisticRegression
import pandas as pd

# Linear Regression train test data
# df = pd.read_csv('50_startups.csv', header=0)
# X = df[['Administration', 'Marketing Spend', 'R&D Spend', 'Marketing Spend']].values
# Y = df['Profit'].values
# X_train = X[:45]
# Y_train = Y[:45]
# X_test = X[17:18]
# Y_test = Y[17:18]

# Linear Regression train test data
# df_test = pd.read_csv('test.csv', header=0)
# df_train = pd.read_csv('train.csv', header=0)
#
# X_train = df_train['x'].values.reshape(-1, 1)[:210]
#
# Y_train = df_train['y'].values[:210]
# X_test = df_test[['x']].values.reshape(-1, 1)[50:54]
# Y_test = df_test['y'].values[50:54]

# Linear regression run and predict
# lr = LinearRegression()
# lr.fit(X_train, Y_train, iterations=3000, learning_rate=0.01, show=True, normalize=True)
# y_pred = lr.predict(X_test)
# print('---')
# print(Y_test)
# print('---')
# print(y_pred)
# lr.plot_cost()
# lr.plot_model(X_train, Y_train)

# Logistic regression training and test data
df = pd.read_csv('Data_for_UCI_named.csv', header=0)
df['stabf'] = df['stabf'].map({'unstable': 0, 'stable': 1})
Y = df['stabf'].values
X = df.drop(['stabf'], axis=1).values

X_train = X[:9000]
Y_train = Y[:9000]
X_test = X[9000:]
Y_test = Y[9000:]

# Logistic regression run and predict
log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train, iterations=200000, learning_rate=0.25, show=True)
y_pred = log_reg.predict(X_test[20:28])
print('---')
print(Y_test[20:28])
print('---')
print(y_pred)
log_reg.plot_cost()
# log_reg.plot_model(X_train, Y_train)

