#import your packages

import pandas as pd
# extracts data from dataset (.csv file)
import csv
# used to read and write to csv files
import numpy as np
# used to convert input into numpy arrays to be fed into the model
import matplotlib.pyplot as plt
# to plot/visualize sales data and sales forecasting
import tensorflow as tf
# acts as the framework upon which this model is built
from tensorflow import keras
# defines layers and functions in the model


def main():
    print("Load the Dataset:")
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    data = pd.read_csv('BostonHousing.csv', header=0)
    data.head()
    print(data)
    print('\n')

    # dependent and independent variables
    data_ = data.loc[:,['lstat', 'medv']]
    data.head(5)
    print('\n')

    # visualize the change in the variables
    data.plot(x='lstat', y='medv', style='o')
    plt.xlabel('lstat')
    plt.ylabel('medv')
    plt.show()

    # divide the data into independent and dependent variables
    X = pd.DataFrame(data['lstat'])
    y = pd.DataFrame(data['medv'])

    # split the data into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # shape of the train and test sets
    print("Shape of the train and test sets:")
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    # train the algorithm
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # retrieve the intercept
    print("Intercept:", regressor.intercept_)

    # retrieve the slope
    print("Slope:", regressor.coef_)

    # predicted value
    print("Predicted Value:\n")
    y_pred = regressor.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=['Predicted'])
    print(y_pred)

    # actual value
    print("Actual Value:\n")
    print(y_test)

    # evaluate the algorithm
    print("Evaluation of the Algorithm:")
    from sklearn import metrics
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



if __name__ == '__main__':
    main()
