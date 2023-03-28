# Import Library
import numpy as np
from numpy import mean
from numpy import absolute
from numpy import sqrt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection  import train_test_split
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import sklearn.metrics as sm
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

def preprocessData(data):
    print("=========Pre-Process=========")
    x = data.isna().any(axis=1).sum()
    print(x)
    print(data.shape)
    print(data.dtypes)

    # Drop any row with NA values
    data_1 = data.dropna(0, how='any')
    data_2 = data_1

    # Normalize Numeric value
    for (columnName, columnData) in data_1.iteritems():
        if data_1[columnName].dtype.kind in 'iufc':
            data_2[columnName] = (data_1[columnName] - data_1[columnName].min()) / (data_1[columnName].max() - data_1[columnName].min())
    #data_2.NA_Sales = (data_1.NA_Sales - data_1.NA_Sales.min()) / (data_1.NA_Sales.max() - data_1.NA_Sales.min())

    # Display the first 10 valid data
    print(data_2.head(10))
    print("=========Done Pre-Process=========")
    return data_2

def displayHist(data):
    # Display a histogram for NA_Sales
    plt.figure(figsize=(9,5))
    data.NA_Sales.hist(bins=15)
    plt.xlabel('NA Sales')
    plt.ylabel('Frequency')
    plt.title('Video Game Sales NA Sales Histogram')
    plt.show()

def eliminateOutlier(_data):
    print("=========Eliminate Outlier=========")
    # Eliminate Outlier
    p = sns.boxplot(data=_data.NA_Sales)

    q1 = _data["NA_Sales"].quantile(0.25)
    q3 = _data["NA_Sales"].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    q_low  = q1-1.5*iqr
    q_hi = q3+1.5*iqr

    data_filtered = _data[(_data["NA_Sales"] < q_hi) & (_data["NA_Sales"] > q_low)]
    print(data_filtered.shape)
    print("=========Done Eliminate Outlier=========")
    return data_filtered

def correlationMap(_data):
    # # Correlation Heatmap
    str_list = [] # empty list to contain columns with strings (words)
    for colname, colvalue in _data.iteritems():
        if pd.api.types.is_string_dtype(_data[colname].dtype):
            str_list.append(colname)
    print(str_list)
    # Get to the numeric columns by inversion
    num_list = _data.columns.difference(str_list)
    # Create Dataframe containing only numerical features
    data_num = _data[num_list]
    f, ax = plt.subplots(figsize=(14, 11))
    plt.title('Pearson Correlation of Video Game Numerical Features')
    # Draw the heatmap using seaborn
    sns.heatmap(data_num.astype(float).corr(),linewidths=0.25,vmax=1.0,
                square=True, cmap="cubehelix_r", linecolor='k', annot=True)

def dummyVariable(data_filtered):
    print("=========Dummy Variable=========")
    # Add dummy variables and drop uninterested features
    data_filtered_2 = pd.get_dummies(data=data_filtered, drop_first=True, columns=['Platform', 'Genre', 'Rating'])
    data_filtered_2 = data_filtered_2.drop(['Name', 'Publisher', 'Global_Sales'], axis=1)
    print(data_filtered_2.head(10))
    print("=========Done Adding Dummy Variable=========")
    return data_filtered_2

def getGoodNumFeatures(X, y):
    print("=========Get Number of features=========")
    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    nof_list=np.arange(1,len(X_train.columns) + 1)
    high_score=0
    #Variable to store the optimum features
    nof=0
    score_list =[]
    for n in range(len(nof_list)):
        model = LinearRegression()
        rfe = RFE(model, n_features_to_select=nof_list[n])
        X_train_rfe = rfe.fit_transform(X_train,y_train)
        X_test_rfe = rfe.transform(X_test)
        model.fit(X_train_rfe,y_train)
        score = model.score(X_test_rfe,y_test)
        score_list.append(score)
        if(score>high_score):
            high_score = score
            nof = nof_list[n]
    print("Optimum number of features: %d" %nof)
    print("Score with %d features: %f" % (nof, high_score))
    print("=========Done Number of features=========")
    return nof

def getSelectedFeatures(X, y, nof):
    print("=========Get Selected Features=========")
    RFE_regressor = LinearRegression()
    #Initializing RFE model
    rfe = RFE(RFE_regressor, n_features_to_select=nof)
    #Transforming data using RFE
    X_rfe = rfe.fit_transform(X,y)
    #Fitting the data to model
    RFE_regressor.fit(X,y)
    list_of_selected_feature = []
    for i in range(len(rfe.support_)):
        if rfe.support_[i]:
            list_of_selected_feature.append(i)
    print("List of Selected Features ID: ")
    print(list_of_selected_feature)

    data_selected=X.iloc[:,list_of_selected_feature]
    print(data_selected.head(10))
    print("=========Done Selected Features=========")
    return data_selected

def linearRegression(X, y):
    print("=========Linear Regression=========")
    # Linear Regression

    nof = getGoodNumFeatures(X, y)

    data_selected = getSelectedFeatures(X, y, nof)

    # Display which features are selected
    print(data_selected)
    print("")

    # Perform Linear Regression
    X_train, X_test, y_train, y_test = train_test_split(data_selected,y, test_size = 0.3, random_state = 0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    print("Training score: ",regressor.score(X_train,y_train))
    print("Model slopes:   ", regressor.coef_)
    print("Model intercept:", regressor.intercept_)
    print("Model slop (rounded): ", np.around(regressor.coef_, 3))
    print("")

    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    print("Testing score: ",regressor.score(X_test,y_test))
    print("Model slopes:   ", regressor.coef_)
    print("Model intercept:", regressor.intercept_)
    print("")

    print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_pred), 2))
    print("Mean squared error =", round(sm.mean_squared_error(y_test,y_pred), 2))
    print("Median absolute error =", round(sm.median_absolute_error(y_test, y_pred), 2))
    print("Explain variance score =", round(sm.explained_variance_score(y_test, y_pred), 2))
    print("")
    print("=========Done Linear Regression=========")
    return data_selected

if __name__== "__main__":
    # Read data
    data = pd.read_csv('Video_Game_Sales_as_of_Jan_2017.csv')

    # Display the first 10 original data
    data.head(10)

    data_2 = preprocessData(data)

    displayHist(data_2)

    data_filtered = eliminateOutlier(data_2)

    # Check Distribution of NA_Sales after removing outlier
    displayHist(data_filtered)
    data_filtered.head(10)

    # # Standardize Data (NA_Sales) to make it normal distribution
    data_filtered = data_filtered[data_filtered.NA_Sales > 0.0000000001]
    data_filtered.NA_Sales = np.log(data_filtered.NA_Sales)
    displayHist(data_filtered)

    # Correlation Heatmap
    correlationMap(data_filtered)

    # Add dummy variables and drop uninterested features
    data_filtered_2 = dummyVariable(data_filtered)

    y = data_filtered_2.NA_Sales
    X = data_filtered_2[data_filtered_2.columns.difference(['NA_Sales'])]
    data_selected = linearRegression(X, y)