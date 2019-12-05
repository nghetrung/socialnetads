import pandas as pd

# Importing the dataset
def importData(file_path):
    dataset = pd.read_csv(file_path)
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values
    return X, y

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
def split_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    return X_train, X_test, y_train, y_test

# Feature Scaling
from sklearn.preprocessing import StandardScaler
def scale(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test