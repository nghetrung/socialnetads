# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
def train(X_train, y_train):
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    return classifier

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
def cross_val(X_train, y_train, classifier):
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
    print("Accuracies mean is:", accuracies.mean())
    print("Accuracies standard deviation is:", accuracies.std())