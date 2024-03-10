import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data


def getThetaClosedSolution(X, Y):
    Y = np.mat(Y)
    return np.linalg.pinv(np.dot(X.T, X)) * np.dot(X.T, Y)  # Wynik wzoru 1.13


data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy().reshape(-1, 1)  # Zamienamy na macierz o jednej kolumnie
x_train = train_data['Weight'].to_numpy().reshape(-1, 1)

# TODO: calculate closed-form solution
ones = np.ones(x_train.shape)  # Tworzymy macierz jedynek (jedna kolumna) - wyrazy wolne
x_train_ = np.hstack((ones, x_train))  # Laczymy - otrzymujemy macierz obserwacji

theta = getThetaClosedSolution(x_train_, y_train)
theta_0 = theta[0, 0]  # b
theta_1 = theta[1, 0]  # a

y_test = test_data['MPG'].to_numpy().reshape(-1, 1)
x_test = test_data['Weight'].to_numpy().reshape(-1, 1)

theta_best = [theta_0, theta_1]

# TODO: calculate error
mse = np.mean(((theta_best[1] * x_test + theta_best[0]) - y_test) ** 2)
print("Calculate error MSE:")
print(mse)

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization
mean_x = np.mean(x_train)
deviation_x = np.std(x_train)
x_train = (x_train - mean_x) / deviation_x

mean_y = np.mean(y_train)
deviation_y = np.std(y_train)
y_train = (y_train - mean_y) / deviation_y

x_test_ = (x_test - mean_x) / deviation_x
y_test_ = (y_test - mean_y) / deviation_y

x_train_ = np.hstack((np.ones(x_train.shape), x_train)) # nowa macierz obserwacji (po standaryzacji)

# TODO: calculate theta using Batch Gradient Descent
theta_best = np.random.rand(2, 1)
learning_rate = 0.01
mse = 0
m = x_train.shape[0]
for i in range(0, 10000):
    # print(mse)
    gradient = 2 / m * x_train_.T.dot(x_train_.dot(theta_best) - y_train)
    theta_best = theta_best - learning_rate * gradient

# TODO: calculate error
y_test_predicted = theta_best[0][0] + theta_best[1][0]*x_test_

y_test_predicted = y_test_predicted*deviation_y+mean_y

mse = np.mean((y_test_predicted - y_test) ** 2)
print("Calculate error MSE (Gradient descent):")
print(mse)
# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
x_stand = (x-mean_x)/deviation_x
y_stand = float(theta_best[0][0]) + float(theta_best[1][0]) * x_stand
y = y_stand*deviation_y+mean_y
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()
