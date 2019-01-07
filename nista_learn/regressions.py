from nista_learn.regressions_functions import *
import matplotlib.pyplot as plt


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
        if len(self.cost_list) > 0:
            self.mse = self.cost_list[-1]
        self.iterations = iterations

    def predict(self, x_test):
        """Predict data"""
        x_test = concatenate_with_ones_vector(x_test)
        y_pred = np.dot(x_test, self.theta)
        return y_pred.T

    def plot_cost(self):
        """Show the cost values variation"""
        import matplotlib.pyplot as plt
        x_plot = range(100, self.iterations+1, 100)
        plt.plot(x_plot, self.cost_list)
        plt.show()

    def plot_model(self, x, y):
        x_plot = x[:, 0]
        x_plot = x_plot.reshape((len(x_plot), 1))
        x_plot = concatenate_with_ones_vector(x_plot)
        plt.plot(x[:, 0], y, "x")
        plt.plot(x[:, 0], np.dot(x_plot, self.theta[:2]), "r-")
        plt.show()






