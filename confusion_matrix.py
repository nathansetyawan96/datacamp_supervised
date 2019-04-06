from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

df = pd.read_csv("diabetes.csv")

df.head(5)

X = df.iloc[:, :7].copy()
y = df.iloc[:, 8].copy()

X.head(5)
y.head(5)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.4)

knn = KNeighborsClassifier(n_neighbors=6)

knn.fit(X_train, y_train)

prediction = knn.predict(X_test)

print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))

df.describe()