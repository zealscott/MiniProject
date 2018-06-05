# classifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import random
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import Doc2Vec
from Doc2Vec import GetData
from Doc2Vec import SaveResult


class ClassifierTrain(object):
    """
    using machine learning tech to train data  
    """

    def __init__(self, trainData, trainLabel, testData=None, testLabel=None):
        self.trainData = trainData
        self.trainLabel = trainLabel
        self.testData = testData
        self.testLabel = testLabel

    def SVMTest(self):
        """  
        Pipeline+GridSearchCV
        """
        kfold = StratifiedKFold(n_splits=5, shuffle=True)
        parameters = [
            {
                'pca__n_components': [20, 40, 60, 80, 100],
                'svm__kernel':['rbf'],
                'svm__gamma':[1e-3, 1e-2, 1e-1],
                'svm__C':[1e-2, 1e-1, 1, 5, 10]
            },
            {
                'pca__n_components': [20, 40, 60, 80, 100],
                'svm__kernel':['linear'],
                'svm__C':[1e-2, 1e-1, 1, 5, 10]
            }
        ]
        pipeline = Pipeline(
            steps=[
                ('pca', PCA()),  # 'pca'对应'pca__'
                ('svm', SVC())  # 'svm'对应'svm__'
            ]
        )
        clf = GridSearchCV(
            estimator=pipeline,
            param_grid=parameters,
            cv=kfold,
            scoring="accuracy",
            n_jobs=4
        )
        clf.fit(self.trainData, self.trainLabel)
        print("using SVM, Best: %f using %s" %
              (clf.best_score_, clf.best_params_))
        best_clf = clf.best_estimator_
        return best_clf

    def SVM(self):
        clf = SVC()
        clf.fit(self.trainData, self.trainLabel)
        score = clf.score(self.testData, self.testLabel)
        print("using SVM, score = %f" % score)
        best_clf = clf
        return best_clf

    def mnb(self):
        clf = MNB()
        clf.fit(self.trainData, self.trainLabel)
        print("20 Fold CV Score for Multinomial Naive Bayes: %f" % (np.mean(cross_val_score
                                                                            (clf, self.trainData, self.trainLabel, cv=20, scoring='roc_auc'))))

    def sgd(self):
        # Regularization parameter
        sgd_params = {'alpha': [0.00006, 0.00007, 0.00008, 0.0001, 0.0005]}

        clf = GridSearchCV(SGD(max_iter=5,random_state=0, shuffle=True, loss='modified_huber'),
                           sgd_params, scoring='roc_auc', cv=20)  # Find out which regularization parameter works the best.

        clf.fit(self.trainData, self.trainLabel)  # Fit the model.
        print("using SGD, Best: %f using %s" %
              (clf.best_score_, clf.best_params_))
        best_clf = clf.best_estimator_
        return best_clf

    def LibSVM(self):
        clf = LinearSVC()
        clf.fit(self.trainData, self.trainLabel)
        score = clf.score(self.testData, self.testLabel)
        print("using LibSVM, score = %f" % score)
        best_clf = clf
        return best_clf

    def NB(self):
        clf = MultinomialNB()
        clf.fit(self.trainData, self.trainLabel)
        score = clf.score(self.testData, self.testLabel)
        print("using MultinomialNB, score = %f" % score)
        best_clf = clf
        return best_clf

    def LR(self):
        # Decide which settings you want for the grid search.
        grid_values = {'C': [30]}

        clf = GridSearchCV(LogisticRegression(penalty='l2', dual=True, random_state=0),
                           grid_values, scoring='roc_auc', cv=20)

        clf.fit(self.trainData, self.trainLabel) # Fit the model.
        print("using LogisticRegression, Best: %f using %s" %
              (clf.best_score_, clf.best_params_))
        best_clf = clf.best_estimator_
        return best_clf


    def KNN(self):
        clf = KNeighborsClassifier(n_neighbors=100)
        clf.fit(self.trainData, self.trainLabel)
        score = clf.score(self.testData, self.testLabel)
        print("using KNeighbors, score = %f" % score)
        best_clf = clf
        return best_clf

    def DT(self):
        clf = DecisionTreeClassifier()
        clf.fit(self.trainData, self.trainLabel)
        score = clf.score(self.testData, self.testLabel)
        print("using DecisionTree, score = %f" % score)
        best_clf = clf
        return best_clf

    def RF(self):
        clf = RandomForestClassifier()
        clf.fit(self.trainData, self.trainLabel)
        score = clf.score(self.testData, self.testLabel)
        print("using RandomForest,score = %f" % score)
        best_clf = clf
        return best_clf


def run():
    data = GetData("./Model/imdb.d2v")

    # trainData, trainLabel, testData, testLabel, X_test = data.GetArray()
    trainData, trainLabel, X_test = data.GetArray()

    print("trainData size:", trainData.shape)
    print("trainLabel size:", trainLabel.shape)
    print("test size:", X_test.shape)

    # clf = ClassifierTrain(trainData, trainLabel, testData, testLabel)
    classifer = ClassifierTrain(trainData, trainLabel)
    # clf = classifer.sgd()
    # clf = classifer.mnb()
    clf = classifer.LR()

    # clf = classify(data, label)
    result = clf.predict_proba(X_test)
    print(result.shape)
    print(result)
    # result = [round(value)for value in result]
    SaveResult(result)
    # clf = clf.SVM()
    # clf.LibSVM()
    # clf.RF()
    # clf.DT()
    # clf.KNN()

if __name__ == '__main__':
    run()
