import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv("diabetes.csv")
del df["SkinThickness"]

X = df.drop(columns=["Outcome"])
y = df.drop(columns=["Pregnancies","Glucose","BloodPressure","Insulin","BMI","DiabetesPedigreeFunction","Age"])
split_test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=42)


print("{0:0.2f}% in training set".format((len(X_train)/len(df.index)) * 100))
print("{0:0.2f}% in test set".format((len(X_test)/len(df.index)) * 100))


nb_model = DecisionTreeClassifier()

nb_model.fit(X_train,y_train)
nb_predict_train = nb_model.predict(X_train)

# Accuracy
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train, nb_predict_train)))


# predict values using the testing data
nb_predict_test = nb_model.predict(X_test)

# training metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, nb_predict_test)))
