from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv("diabetes.csv")

X = df.iloc[:, :7].copy()
y = df.iloc[:, 8].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

prediction = logreg.predict(X_test)

print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))
print("The accuracy is", accuracy_score(y_test, prediction))