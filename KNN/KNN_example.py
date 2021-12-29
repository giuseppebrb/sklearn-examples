# We want to know if a car can be classified as unacceptable, acceptable, good or vary good, using a classification model
# based on 3 features.
# Car dataset source: https://archive.ics.uci.edu/ml/datasets/car+evaluation

import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('car.data')

X = data[['buying', 'maint', 'safety']].values  # main components from our dataset
y = data[['class']]

# Conversions of string features
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])  #convert string values into numbers

# Conversion of target variable, from string to numbers
label_mapping = {
    'unacc': 0,
    'acc': 1,
    'good': 2,
    'vgood': 3
}
y['class'] = y['class'].map(label_mapping)
y = np.array(y)


# Creating the model
knn = neighbors.KNeighborsClassifier(25, weights='uniform')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
accuracy = metrics.accuracy_score(y_test, predictions)
print('predictions: ', predictions)
print('actual: ', y_test)
print('accuracy: ', accuracy)
