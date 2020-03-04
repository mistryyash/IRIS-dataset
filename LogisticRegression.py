# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
df = pd.read_csv("iris.csv")
df = df[["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species"]]
X = np.array(df.drop(["Species"],1))
y = np.array(df["Species"])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)

# Fitting the Model to the dataset
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)
conf = clf.score(X_test, y_test)
print(conf)
