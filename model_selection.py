import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Create a list of different models
models = [LogisticRegression(), SVC(), DecisionTreeClassifier(), RandomForestClassifier()]

# Load the dataset
X = np.load('data.csv')
y = np.load('labels.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize variables to store the best model and its performance
best_model = None
best_accuracy = 0

# Iterate over the models
for model in models:
    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Evaluate the model on the test data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print the performance of the model
    print('Model: {}'.format(model))
    print('Accuracy: {}'.format(accuracy))
    print('F1 score: {}'.format(f1))

    # Update the best model and its performance if necessary
    if accuracy > best_accuracy:
        best_model = model
        best_accuracy = accuracy

# Print the best model and its performance
print('Best model: {}'.format(best_model))
print('Best accuracy: {}'.format(best_accuracy))
