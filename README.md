# Finding Best Predictor for Boston House Database

A project for learning few optimal solutions to predict housing prices using Boston Housing Prices (provided by SKLearn).

## Data Preprocessing

The data provided by the SKLearn (Boston House) is rather clean and fit for purpose of learning. In practices, there might be a big chance that the data is less useful and preprocessing is rather needed. This project shall not focus on preprocessing such as, interpolate missing values, etc.

## Feature Selections

There are 13 features are gonna be used according to the database which you can find further details from [here](https://www.kaggle.com/c/boston-housing) and how to load it with SKLearn from [here](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html#sklearn.datasets.load_boston).

(As quoted from the DESCR attributes of the dataset) All attribute Information in order are:
* CRIM     per capita crime rate by town
* ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
* INDUS    proportion of non-retail business acres per town
* CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
* NOX      nitric oxides concentration (parts per 10 million)
* RM       average number of rooms per dwelling
* AGE      proportion of owner-occupied units built prior to 1940
* DIS      weighted distances to five Boston employment centres
* RAD      index of accessibility to radial highways
* TAX      full-value property-tax rate per $10,000
* PTRATIO  pupil-teacher ratio by town
* B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
* LSTAT    % lower status of the population
* MEDV     Median value of owner-occupied homes in $1000's

In the project, each features regression coefficient will be evaluated to see how relevant they are to build the best predictor.

## More...

More updates to come...

