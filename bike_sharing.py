import pandas
import numpy
from sklearn.tree import DecisionTreeClassifier

def main():
    #Directory in which the .csv files are located
    path = "/home/borgesl/Downloads/"
    #Reading training data
    train_set = pandas.read_csv(path + "train.csv")
    #Reading testing data
    test_set = pandas.read_csv(path + "test.csv")
    
    #Preparing training data
    features = ['season','holiday','workingday','weather','temp','atemp','humidity','windspeed']
    #Matrix of features
    x_train = train_set[features]
    #Array with labels
    y_train = train_set['count']
    
    #Feeding classifier with training data
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    
    #
    #test_features = numpy.array(test_set[features])
    test_features = test_set[features]
    #
    predicted = model.predict(test_features)
    
    #Creating new DataFrame
    data = {'count' : predicted}
    data_frame = pandas.DataFrame(data = data, index = test_set['datetime'])

    #Writing to .csv file
    data_frame.to_csv("bike_sharing.csv")

main()
