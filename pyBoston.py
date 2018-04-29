import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation

# local
from linRegMulti import CrossValidationLinReg, UnivariateLinReg


def main_univariate_with_class():
    # load
    boston = load_boston()

    # create dataframe
    df_boston = pd.DataFrame(boston['data'], columns=boston['feature_names'])
    sr_target = pd.Series(data=boston['target'], name='target')

    # create object for the regressor
    lreg_uni = UnivariateLinReg(data=df_boston['RM'], feature_name=['RM'], target_data=sr_target)

    # plot
    lreg_uni.plot()


def main_with_class():
    # load
    boston = load_boston()

    # create dataframe
    df_boston = pd.DataFrame(boston['data'], columns=boston['feature_names'])
    sr_target = pd.Series(data=boston['target'], name='target')

    # create object for the regressor
    lreg_local = CrossValidationLinReg(data=df_boston, 
                                       feature_list=df_boston.columns, 
                                       target_data=sr_target)

    # print
    print('average squared error for trained: %.2f' % lreg_local.model_err_sq)
    print('average squared error for test: %.2f' % lreg_local.valid_err_sq)


if __name__ == "__main__":
    main_univariate_with_class()
    # main_with_class()