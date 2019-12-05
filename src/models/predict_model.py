# Predicting the Test set results
def predict(X_test, classifier):
    y_pred = classifier.predict(X_test)
    return y_pred

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
def confusion_mat(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print(cm)