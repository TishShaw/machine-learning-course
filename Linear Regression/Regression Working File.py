import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

# print(data.head())  # Print the first 5 elements

data = data[["G1", "G2", "G3", "studytime", "absences", "failures"]]

predict = "G3"

# Two arrays that define attributes and labels

X = np.array(data.drop([predict], axis=1))  # Returns a new df excluding "G3"
Y = np.array(data[predict])
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

"""
best = 0
for _ in range(30):
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(X_train, Y_train)  # Fits the daya to find a line of best fit
    acc = linear.score(X_test, Y_test)  # Return a value that represents the accuracy of our model
    print(acc)

    if acc > best:
        best = acc
        with open("student_performance_model.pickle", "wb") as f:
            pickle.dump(linear, f)
"""

pickle_in = open("student_performance_model.pickle", "rb")
linear = pickle.load(pickle_in)

# Get y-intercept
print("Co: \n ", linear.coef_)
print("Intercept: \n ",  linear.intercept_)

# Predict student grades
predictions = linear.predict(X_test)

for x in range(len(predictions)):
    print(predictions[x], X_test[x], Y_test[x])  # Prints prediction, input data, actual value of final grade

p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()