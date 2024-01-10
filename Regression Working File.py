import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model

data = pd.read_csv("student-mat.csv", sep=";")

# print(data.head())  # Print the first 5 elements

data = data[["G1", "G2", "G3", "studytime", "absences", "failures"]]

predict = "G3"

# Two arrays that define attributes and labels

X = np.array(data.drop([predict], axis=1))  # Returns a new df excluding "G3"
Y = np.array(data[predict])

# Split into 4 diff arrays
# Splits 10% of data into test samples
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

linear = linear_model.LinearRegression()

linear.fit(X_train, Y_train)  # Fits the daya to find a line of best fit

linear.score(X_test, Y_test)  # Return a value that represents the accuracy of our model
acc = linear.score(X_test, Y_test)
print(acc)

# Get y-intercept
print("Co: \n ", linear.coef_)
print("Intercept: \n ",  linear.intercept_)

# Predict student grades
predictions = linear.predict(X_test)

for x in range(len(predictions)):
    print(predictions[x], X_test[x], Y_test[x])  # Prints prediction, input data, actual value of final grade
