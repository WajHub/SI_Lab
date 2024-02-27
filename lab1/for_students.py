import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data


def predict(X, theta):
    return np.dot(X, theta)  # Iloczy wektorowy

def getThetaClosedSolution(X, Y):
    Y = np.mat(Y)
    return np.linalg.pinv(np.dot(X.T, X)) * np.dot(X.T, Y) # Wynik wzoru 1.13


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
theta_0 = 0
theta_1 = 0

ones = np.ones((x_train.shape[0], 1))  # Tworzymy macierz jedynek (jedna kolumna) - wyrazy wolne
x_train_ = np.hstack((ones, x_train))  # Laczymy - otrzymujemy macierz obserwacji

theta = getThetaClosedSolution(x_train_,y_train)
theta_0 = theta [0,0] #b
theta_1 = theta[1,0] #a

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

theta_best = [theta_0, theta_1]

# TODO: calculate error
m = sum(len(wiersz) for wiersz in x_train)
mse = 0
for i in range(len(x_train)):
    for j in range(len(x_train[i])):
        mse=mse + ((theta_1*x_train[i,j]+theta_0)-y_train[i,j])**2
mse = mse/m
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

# TODO: calculate theta using Batch Gradient Descent

# TODO: calculate error

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()
