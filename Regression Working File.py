import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")

print(data.head())  # Print the first 5 elements

data = data[["G1", "G2", "G3", "studytime", "absences", "failures"]]

predict = "G3"

# Two arrays that define attributes and labels

X = np.array(data.drop([predict], 1))  # Returns a new df excluding "G3"
Y = np.array(data[predict])

# Split into 4 diff arrays
# Splits 10% of data into test samples
X_train, Y_train, X_test, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
