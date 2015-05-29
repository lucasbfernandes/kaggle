#Lucas Borges Fernandes
#2015

import pandas
import numpy
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn import cross_validation
from datetime import datetime

#Get month of formatted string
def get_month(date_string):
    return datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S").date().month

def cross_val(model, features, count):
    #KFold(number_of_elements, 
    kf = cross_validation.KFold(n = len(features), n_folds = 2, indices = True, shuffle = True, random_state = 4)
    #train_set = train_set.as_matrix()
    scoreArray = cross_validation.cross_val_score(model, features, count, cv = kf, n_jobs = 1)
    #score
    score = numpy.mean(scoreArray) * 100
    return score

def main():
    #Directory in which the .csv files are located
    path = "/home/borgesl/Downloads/"
    #Reading training data
    train_set = pandas.read_csv(path + "train.csv")
    #Reading testing data
    test_set = pandas.read_csv(path + "test.csv")

    #Adding a new feature by applying the function get_month to each
    #datetime entry of the .csv vile
    train_set['month'] = train_set['datetime'].map(get_month)
    test_set['month'] = test_set['datetime'].map(get_month)

    #Preparing training data
    #These are the features that will be used to train the classifier
    #After adding new features, we will have to manually add their name to this
    #list
    features = ['season','holiday','workingday','weather','temp','atemp','humidity','windspeed', 'month']
    #Getting the desired features from the training set
    features_train = train_set[features]
    #Getting the labels of the data contained in the training set
    count_train = train_set['count']

    #Feeding classifier with training data
    #Can be changed to other classifiers, such as LogisticRegression,
    #KNearestNeighbors etc.
    #model = KNeighborsClassifier(n_neighbors=1, weights = 'uniform')
    model = RandomForestRegressor(n_estimators = 10)
    #model = DecisionTreeClassifier()
    model.fit(features_train, count_train)

    score = cross_val(model, features_train, count_train)
    print(score)

    #Now we will feed the classifier with the test data, so that we can
    #classify it
    #Getting desired features from the testing set
    features_test = test_set[features]
    #Predicting it
    predicted = model.predict(features_test)
    
    #Creating new DataFrame that will be used to write the results on a .csv
    #file
    #One column named 'count' that has the predicted list as its rows
    data = {'count' : predicted}
    #DataFrame will have the datetimes as the index of the predicted values
    data_frame = pandas.DataFrame(data = data, index = test_set['datetime'])

    #Writing to .csv file
    data_frame.to_csv("results/bike_sharing_simplest.csv")

main()
