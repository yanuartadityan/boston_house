import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation


class Common(object):
    def __init__(self):
        self.lreg = LinearRegression()
        self.x_data = pd.DataFrame()
        self.y_data = pd.DataFrame()
        self.train_ratio = 0.5
        self.train_data_x = pd.DataFrame()
        self.train_data_y = pd.DataFrame()
        self.test_data_x = pd.DataFrame()
        self.test_data_y = pd.DataFrame()
        self.model_err_sq = 0.0
        self.valid_err_sq = 0.0

    def split(self):
        self.train_data_x, self.test_data_x, self.train_data_y, self.test_data_y = \
        cross_validation.train_test_split(self.x_data, self.y_data, train_size=self.train_ratio)

    def fit(self):
        self.lreg.fit(self.train_data_x, self.train_data_y)

    def predict(self):
        self.pred_train_y = self.lreg.predict(self.train_data_x)
        self.pred_test_y = self.lreg.predict(self.test_data_x)

    def calc_error(self):
        self.model_err_sq = np.mean((self.train_data_y - self.pred_train_y) ** 2)
        self.valid_err_sq = np.mean((self.test_data_y - self.pred_test_y) ** 2)

    def best_fit_lin_reg(self):
        pass

class UnivariateLinReg(Common):
    """a class for univariate (single feature) linear regression"""
    
    def __init__(self, data=None, feature_name=None, target_data=pd.Series(name='None'), train_ratio=0.5):
        """constructor"""

        # construct superclass
        Common.__init__(self)

        # build local properties
        self.train_ratio = train_ratio
        self.x_label = feature_name

        # feature X to use
        if isinstance(data, pd.DataFrame) and not self.x_label:
            self.x_data = data
        elif not isinstance(data, pd.DataFrame) and self.x_label:
            self.x_data = pd.DataFrame(data, columns=self.x_label)   
        else:
            raise ValueError('expecting correct data input as DataFrame or NDArray object')

        # target Y
        if isinstance(target_data, pd.Series):
            self.y_data = target_data
        else:
            self.y_data = pd.Series(target_data)

        # regression
        self.perform_regression()

    def perform_regression(self):
        self.X = np.array([[v, 1] for v in self.x_data._values])

        # start the linear regression with numpy
        result = np.linalg.lstsq (self.X, self.y_data)

        # get best fit line's gradient
        self.coefficient = result[0][0]

        # get the intercept value
        self.intercept = result[0][1]

        # error
        self.valid_err_sq = np.sqrt(result[1]/len(self.X))

    def plot(self):
        # plot number of rooms vs house prices
        plt.figure(0)
        plt.plot(self.X, self.y_data._values, 'o')
        # plt.plot(self.X, self.coefficient*self.X + self.intercept, 'r', label='Best fit line')
        plt.title('Best fit line is f(X) = %.2fX + %.2f' % (self.coefficient, self.intercept))
        plt.xlabel('Number of rooms')
        plt.ylabel('House prices in $1000')
        plt.show()


class CrossValidationLinReg(Common):
    """a class for linear regression for multi features"""

    def __init__(self, data=None, feature_list=None, target_data=pd.Series(name='None'), train_ratio=0.5):
        """constructor"""
        
        # construct superclass
        Common.__init__(self)

        # build local properties
        self.x_data = pd.DataFrame(data=data, columns=feature_list)
        
        # get target data -- propietary
        if not target_data.empty:
            self.target_name = target_data.name
            self.target_data = target_data
        else:
            raise ValueError('target_data must not be None type')

        # split train data and test data
        self.perform_regression()

    def perform_regression(self):
        # split
        self.split()

        # fit
        self.fit()

        # predict
        self.predict()

        # calc error
        self.calc_error()
