# nista_learn
This is a Python ML librabry like scikit-learn

You can create your ML model or use some ML algorithms on your project

## Example: Logistic Regression
Read csv file and slip data into training and test data
````
import pandas as pd
df = pd.read_csv('Data_for_UCI_named.csv', header=0)
df['stabf'] = df['stabf'].map({'unstable': 0, 'stable': 1})
Y = df['stabf'].values
# sometimes it's needed to reshape data
X = df.drop(['stabf'], axis=1).values

X_train = X[:9000]
Y_train = Y[:9000]
X_test = X[9000:]
Y_test = Y[9000:]
````
Let's use our library
````
# call the LogisticRegression class
from nista_learn.regressions import LinearRegression, LogisticRegression

log_reg = LogisticRegression()
# fitting data
log_reg.fit(X_train, Y_train, iterations=200000, learning_rate=0.25, show=True)
# predict a small dataset
y_pred = log_reg.predict(X_test[20:28])
print('--- small value ---')
print(Y_test[20:28])
print('--- predicted data ---')
print(y_pred)
# plotting the cost function
log_reg.plot_cost()
```