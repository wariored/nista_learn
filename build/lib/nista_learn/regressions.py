from nista_learn.regressions_functions import *
import matplotlib.pyplot as plt


class Regression:
    """Regression parent"""

    def __init__(self):
        self.mse = None
        self.cost_list = None
        self.theta = None
        self.iterations = None

    def plot_cost(self):
        """Show the cost values variation"""
        x_plot = range(100, self.iterations + 1, 100)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost Function')
        plt.plot(x_plot, self.cost_list)
        plt.show()


class LinearRegression(Regression):
    """Linear Regression class"""

    def fit(self, x, y, iterations=1000, learning_rate=0.001, show=False, normalize=False):
        """Fitting data"""
        self.theta, self.cost_list = compute_gradient_linear(x, y, iterations=iterations,
                                                             learning_rate=learning_rate,
                                                             show=show, normalize=normalize)
        if len(self.cost_list) > 0:
            self.mse = self.cost_list[-1]
        self.iterations = iterations

    def predict(self, x_test):
        """Predict data"""
        x_test = concatenate_with_ones_vector(x_test)
        y_pred = np.dot(x_test, self.theta)
        return y_pred.T

    def plot_model(self, x, y):
        x_plot = x[:, 0]
        x_plot = x_plot.reshape((len(x_plot), 1))
        x_plot = concatenate_with_ones_vector(x_plot)
        plt.plot(x[:, 0], y, "x")
        plt.plot(x[:, 0], np.dot(x_plot, self.theta[:2]), "r-")
        plt.show()


class LogisticRegression(Regression):
    def fit(self, x, y, iterations=1000, learning_rate=0.001, show=False):
        """Fitting data"""
        self.theta, self.cost_list = compute_gradient_logistic(x, y, iterations=iterations,
                                                               learning_rate=learning_rate,
                                                               show=show)
        if len(self.cost_list) > 0:
            self.mse = self.cost_list[-1]
        self.iterations = iterations

    def predict(self, x_test, bound=0.5):
        """Predict data"""
        if bound < 0.5:
            raise Exception("bound cannot be less than 50%")
        x_test = concatenate_with_ones_vector(x_test)
        y_pred = np.dot(x_test, self.theta)
        y_pred = sigmoid(y_pred)
        y_pred = np.where(y_pred < bound, 0, 1)
        return y_pred.T

    def plot_model(self, x, y):
        import seaborn as sns
        x_plot = x[:, 0]
        x_plot = x_plot.reshape((len(x_plot), 1))
        x_plot = concatenate_with_ones_vector(x_plot)
        y_plot = sigmoid(np.dot(x_plot, self.theta[:2]))
        plt.plot(x[:, 0], y, "x")
        sns.regplot(x=x[:, 0], y=y_plot, logistic=True)
        plt.show()
