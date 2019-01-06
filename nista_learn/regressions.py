from nista_learn.regressions_functions import *


class LinearRegression:
    """Linear Regression class"""
    def __init__(self):
        self.mse = None
        self.cost_list = None
        self.theta = None
        self.iterations = None

    def fit(self, x, y, iterations=1000, learning_rate=0.001, show=False):
        """Fitting data"""
        self.theta, self.cost_list = compute_gradient(x, y, iterations=iterations, learning_rate=learning_rate,
                                                      show=show)
        self.mse = self.cost_list[-1]
        self.iterations = iterations

    def predict(self, x_test):
        """Predict data"""
        ones = np.ones([x_test.shape[0], 1])
        x_test = np.concatenate((ones, x_test), axis=1)
        y_pred = np.dot(x_test, self.theta)
        return y_pred

    def plot_cost(self):
        """Show the cost values variation"""
        import matplotlib.pyplot as plt
        x_plot = range(100, self.iterations+1, 100)
        plt.plot(x_plot, self.cost_list)
        plt.plot()
        plt.show()





