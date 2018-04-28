import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation


class CrossValidationLinReg:
    """a class for linear regression for multi features"""

    def __init__(self, data=None, feature_list=None, target_data=pd.Series(name='None'), train_ratio=0.5):
        """constructor"""
        
        # build local properties
        self.train_ratio = train_ratio
        self.df_data = pd.DataFrame(data=data, columns=feature_list)
        
        # get target data -- propietary
        if not target_data.empty:
            self.target_name = target_data.name
            self.target_data = target_data
        else:
            raise ValueError('target_data must not be None type')
        
        # create a linear regression object
        self.lreg = LinearRegression()

        # split train data and test data
        self.split()

        # fit
        self.fit()

        # predict
        self.predict()

        # calc error
        self.calc_error()

    def split(self):
        self.train_data_x, self.test_data_x, self.train_data_y, self.test_data_y = \
        cross_validation.train_test_split(self.df_data, self.target_data, train_size=self.train_ratio)

    def fit(self):
        self.lreg.fit(self.train_data_x, self.train_data_y)

    def predict(self):
        self.pred_train_y = self.lreg.predict(self.train_data_x)
        self.pred_test_y = self.lreg.predict(self.test_data_x)

    def calc_error(self):
        self.model_err_sq = np.mean((self.train_data_y - self.pred_train_y) ** 2)
        self.valid_err_sq = np.mean((self.test_data_y - self.pred_test_y) ** 2)