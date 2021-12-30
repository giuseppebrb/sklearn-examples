from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt

diabetes = datasets.load_diabetes()

X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print('Coefficients: ', model.coef_)
print('Intercept: ', model.intercept_)
print('Mean Squared Error (MSE): %2f' % mean_squared_error(y_test, predictions))
print('Coefficient of determination (R^2): %2f' % r2_score(y_test, predictions))

model_plot = plt.scatter(y_test, predictions, alpha=0.5)
plt.show()