import numpy as np
import pandas as pd
import time

from sklearn import cross_validation
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def preprocess_features(features):
    processed_features = pd.DataFrame(index=features.index)

    for col, col_data in features.iteritems():
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # If still non-numeric, convert to one or more dummy variables(e.g. 'school' => 'school_GP', 'school_MS')
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)
        processed_features = processed_features.join(col_data)

    return processed_features

def train_classifier(clf, X_train, y_train):
    print "Training {}...".format(clf.__class__.__name__)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print "Done!\nTraining time (secs): {:.3f}".format(end - start)

def predict_labels(clf, features, target):
    print "Predicting labels using {}...".format(clf.__class__.__name__)
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    print "Done!\nPrediction time (secs): {:.3f}".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')

def train_predict(clf, X_train, y_train, X_test, y_test):
    print "------------------------------------------"
    print "Training set size: {}".format(len(X_train))
    train_classifier(clf, X_train, y_train)
    print "F1 score for training set: {}".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {}".format(predict_labels(clf, X_test, y_test))

student_data = pd.read_csv("student-data.csv")
feature_cols = list(student_data.columns[:-1])
target_col = student_data.columns[-1]

X_all = student_data[feature_cols]
y_all = student_data[target_col]
X_all = preprocess_features(X_all)

num_all = student_data.shape[0]
num_train = 300
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_all, y_all, train_size=(num_train/float(num_all)))

clf_tree = DecisionTreeClassifier()
train_predict(clf_tree, X_train[:100], y_train[:100], X_test, y_test)
train_predict(clf_tree, X_train[:200], y_train[:200], X_test, y_test)
train_predict(clf_tree, X_train[:300], y_train[:300], X_test, y_test)

clf_knn = KNeighborsClassifier()
train_predict(clf_knn, X_train[:100], y_train[:100], X_test, y_test)
train_predict(clf_knn, X_train[:200], y_train[:200], X_test, y_test)
train_predict(clf_knn, X_train[:300], y_train[:300], X_test, y_test)

clf_svc = SVC()
train_predict(clf_svc, X_train[:100], y_train[:100], X_test, y_test)
train_predict(clf_svc, X_train[:200], y_train[:200], X_test, y_test)
train_predict(clf_svc, X_train[:300], y_train[:300], X_test, y_test)

clf_tune_params = {
    'C': np.arange(0.1, 1, 0.1),
    'kernel': ['poly'],
    'degree': np.arange(1, 3, 1),
    'gamma': np.arange(1, 3, 1)}
f1_scorer = make_scorer(f1_score, pos_label="yes")
clf_tune = GridSearchCV(SVC(), clf_tune_params, scoring=f1_scorer)
train_predict(clf_tune, X_train, y_train, X_test, y_test)
print clf_tune.best_params_

clf_final = SVC(0.1, 'poly', 1, 1)
train_predict(clf_final, X_train, y_train, X_test, y_test)
