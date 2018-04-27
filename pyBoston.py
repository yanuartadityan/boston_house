import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation


def univariate_lreg(dFrame, feature_list, target_arr):
    """univariate linear regression using np.linalg.lsqt"""

    print('---univariate linear regression---')

    # iterate
    for feat in feature_list:
        if feat == 'RM':
            X = dFrame[feat]
            print(X.shape)

            """Y=mX+b
            that's the definition of linear equation, in which
            Y = target_arr which is known and X supposed to be
            the features, which is now currently 'RM'.
            m and b are values we would like to estimate and refine.
            m is gradient of the linear line and b is the intercept
            value"""
            X = np.array([[v, 1] for v in X])

            # start the linear regression with numpy
            result = np.linalg.lstsq (X, target_arr)

            # get best fit line's gradient
            m = result[0][0]

            # get the intercept value
            b = result[0][1]

            # plot number of rooms vs house prices
            plt.plot(X, target_arr, 'o')
            plt.plot(X, m*X + b, 'r', label='Best fit line')
            plt.xlabel('Number of rooms')
            plt.ylabel('House prices in $1000')

            # print
            print('the coefficient is %.2f' % m)
            print('the intercept is %.2f' % b)

            # error
            rmse = np.sqrt(result[1]/len(X))
            print('the RMSE is %.3f' % rmse)


def multivariate_lreg(dFrame, feature_list, target_arr):
    """multivariate linear regression using sklearn multi"""

    print('---multivariate linear regression---')

    # create a lreg object
    lreg = LinearRegression()

    # fit linear regression on multiple features
    lreg.fit(dFrame.drop('TARGET', 1), target_arr)

    # print the estimated intercept coefficient
    print('the estimated intercept coefficient is %.2f ' % lreg.intercept_)
    print('the number of coefficients used was %d ' % len(lreg.coef_))

    # report the coefficient estimate
    coeff_df = pd.DataFrame(feature_list)
    coeff_df.columns = ['Features']

    # fill the coeff estimate
    coeff_df['Coefficient Estimate'] = pd.Series(lreg.coef_)

    # set a new column
    print(coeff_df)

def cross_validation_fit_predict(dFrame, feature_list, target_arr):
    """cross validation for fitting and prediction, including
    splitting data into training and validation"""

    print('---cross validation fit and predict---')

    # split the data (Xn-Y) pairs for training and validation
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(dFrame.drop('TARGET',1),
                                                                         target_arr,
                                                                         train_size=0.9)

    # now that we have all the pair of train-test data, we can start the normal fitting and validation
    print('shape of train-test data: ', X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    # create object for Linear Regression
    lreg = LinearRegression()

    # fit
    lreg.fit(X_train, Y_train)


def main():
    # 1. load the database
    boston = load_boston()

    # 2. create dataframe out of boston data
    df_boston = pd.DataFrame(boston.data, columns=boston.feature_names)
    df_boston['TARGET'] = pd.Series(boston.target)

    # 3. perform linear regression (univariate)
    # univariate_lreg(df_boston, df_boston.columns, df_boston['TARGET'])

    # 4. perform multivariate regression (multivariate)
    multivariate_lreg(df_boston, df_boston.columns, df_boston['TARGET'])

    # 5. cross validation
    cross_validation_fit_predict(df_boston, df_boston.columns, df_boston['TARGET'])

    # do this last
    plt.show()

if __name__ == "__main__":
    main()
