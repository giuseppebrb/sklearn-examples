# We want to predict the price of housing in Boston

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

boston_dataset = datasets.load_boston()
X = boston_dataset.data
y = boston_dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

linear_reg = linear_model.LinearRegression()
model = linear_reg.fit(X_train, y_train)

predictions = model.predict(X_test)
print('Predictions: ', predictions)
print('R^2: ', linear_reg.score(X, y))
