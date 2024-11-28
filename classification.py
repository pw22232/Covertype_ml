import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_covtype
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

random_seed = 42
np.random.seed(random_seed)

def load_data():
    data = fetch_covtype()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=random_seed)

    # Scale data for algorithms
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def logistic_regression():
    logistic_model = LogisticRegression(max_iter=1000, random_state=random_seed)
    logistic_model.fit(X_train, y_train)
    y_pred = logistic_model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def decision_tree():
    decision_model = DecisionTreeClassifier(random_state=random_seed)
    decision_model.fit(X_train, y_train)
    y_pred = decision_model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def random_forest():
    random_model = RandomForestClassifier(random_state=random_seed)
    random_model.fit(X_train, y_train)
    y_pred = random_model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def svm_method():
    svm_model = SVC(random_state=random_seed)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    return accuracy_score(y_test, y_pred)

if __name__ == "__main__":
    # Code Task 6
    X_train, X_test, y_train, y_test = load_data()
    # Code Task 7
    log_reg_accuracy = logistic_regression()
    # Code Task 8
    decision_tree_accuracy = decision_tree()
    # Code Task 9
    random_forest_accuracy = random_forest()
    # SVM classifier
    #svm_accuracy = svm_method()

    # Print results
    print("Logistic Regression Accuracy:", f"{log_reg_accuracy * 100:.2f}%")
    print("Decision Tree Accuracy:", f"{decision_tree_accuracy * 100:.2f}%")
    print("Random Forest Accuracy:", f"{random_forest_accuracy * 100:.2f}%")
   # print("SVM Accuracy:", f"{svm_accuracy * 100:.2f}%")
