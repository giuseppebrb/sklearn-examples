# We want to classify iris flowers using sepal length, sepal width, petal length and petal width as features.
# We will create a classification model using SVM algorithm.

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

iris_dataset = datasets.load_iris()
X = iris_dataset.data
y = iris_dataset.target
classes = ['setosa', 'versicolor', 'virginica']
print(iris_dataset)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = svm.SVC()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy_score = metrics.accuracy_score(y_test, predictions)

print('predictions: ', predictions)
print('actual: ', y_test)
print('accuracy score: ', accuracy_score)

print('predictions showing names:')

for i in range(len(predictions)):
    print(classes[predictions[i]])
