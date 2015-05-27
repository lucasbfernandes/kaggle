#Lucas Borges Fernandes
#2015

import pandas
import numpy
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

#Get year of formatted string
def get_year(date_string):
    return datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S").date().year 

#Get month of formatted string
def get_month(date_string):
    return datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S").date().month

#Get hour of formatted string
def get_hour(date_string):
    return datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S").time().hour

#Get day of the week -- Sunday = 0, Monday = 1, ... Saturday = 6
def get_day(date_string):
    return datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S").strftime("%w")

def main():
    #Directory in which the .csv files are located
    path = "/home/borgesl/Downloads/"
    #Reading training data
    train_set = pandas.read_csv(path + "train.csv")
    #Reading testing data
    test_set = pandas.read_csv(path + "test.csv")
    
    #Feature Engineering
    #Adding new features
    functions = [get_year, get_month, get_hour, get_day]
    columns = ['year', 'month', 'hour', 'day']
    for col, func in zip(columns, functions):
        train_set[col] = train_set['datetime'].map(func)
        test_set[col] = test_set['datetime'].map(func)

    #Preparing training data
    #These are the features that will be used to train the classifier
    #After adding new features, we will have to manually add their name to this
    #list
    features = ['season','holiday','workingday','weather','temp','atemp','humidity','windspeed', 'year', 'month', 'hour', 'day']
    #Getting the desired features from the training set
    features_train = train_set[features]
    #Getting the labels of the data contained in the training set
    labels_train = train_set['count']
    
    #Feeding classifier with training data
    #Can be changed to other classifiers, such as LogisticRegression,
    #KNearestNeighbors etc.
    #model = DecisionTreeClassifier()
    model = RandomForestClassifier(n_estimators = 10)
    model.fit(features_train, labels_train)
    
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
    data_frame.to_csv("results/bike_sharing_improved.csv")
 
main()
