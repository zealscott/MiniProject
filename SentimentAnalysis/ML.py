import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.feature_extraction.text import TfidfVectorizer as TFIV
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
import xgboost as xgb
import pickle


class classify():
    def __init__(self, X_train, Y_Train, X_Test):
        self.X_train = X_train
        self.Y_Train = Y_Train
        self.X_Test = X_Test

    def LR(self):
        """ 
        use LogisticRegression and GridSearchCV to find best parameters
        """
        # Decide which settings you want for the grid search.
        grid_values = {'C': [1e-3,1e-2,1e-1,1,2]}
        # grid_values = {'C': [1e-5,1e-4,1e-3,1e-2,1e-1]}

        clf = GridSearchCV(LR(penalty='l2', dual=True, random_state=0),
                           grid_values, scoring='roc_auc', cv=20,n_jobs=4)
        # Try to set the scoring on what the contest is asking for.
        # The contest says scoring is for area under the ROC curve, so use this.
        clf.fit(self.X_train, self.Y_Train)  # Fit the model.
        print("using LogisticRegression, Best: %f using %s" %
              (clf.best_score_, clf.best_params_))
        self.best_clf = clf.best_estimator_
        return self.best_clf

    def mnb(self):
        clf = MNB()
        clf.fit(self.X_train, self.Y_Train)
        print("20 Fold CV Score for Multinomial Naive Bayes: %f" % (np.mean(cross_val_score
                                                                            (clf, self.X_train, self.Y_Train, cv=20, scoring='roc_auc'))))
        self.best_clf = clf
        return clf

    def SVMTest(self):
        """  
        Pipeline+GridSearchCV
        """
        parameters = [{'kernel': ['rbf'], 'gamma': [1e-2,0.005,1e-3],
                       'C': [0.5,1,1.5,2,4]},
                      {'kernel': ['linear'], 'C': [1e-3, 1e-2,0.1,1]}]
        clf = GridSearchCV(
            SVC(probability=True),
            parameters,
            cv=5,
            scoring="roc_auc",
            n_jobs=4
        )
        clf.fit(self.X_train, self.Y_Train)
        print("using SVM, Best: %f using %s" %
              (clf.best_score_, clf.best_params_))
        self.best_clf = clf.best_estimator_
        return self.best_clf

    def sgd(self):
        # Regularization parameter
        # sgd_params = {'alpha': [ 0.18,0.17,0.19,0.185]}
        sgd_params = {'alpha': [1e-1,0.5,1,1.5]}

        clf = GridSearchCV(SGD(max_iter=5, random_state=0,loss='modified_huber',n_jobs=4),sgd_params, scoring='roc_auc', cv=20)  # Find out which regularization parameter works the best.

        clf.fit(self.X_train, self.Y_Train)
        print("using SGD, Best: %f using %s" %
              (clf.best_score_, clf.best_params_))
        self.best_clf = clf.best_estimator_
        return self.best_clf

    def sgdboot(self):
        cv_params = {'max_depth': [7,9,10], 'min_child_weight': [1, 3, 5]}
        ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8,
                      'objective': 'binary:logistic'}
        clf = GridSearchCV(xgb.XGBClassifier(**ind_params),
                           cv_params,
                           scoring='roc_auc', cv=5, n_jobs=4,verbose=True)
        clf.fit(self.X_train, self.Y_Train)
        print("using sgdboot, Best: %f using %s" %
              (clf.best_score_, clf.best_params_))
        # print(clf.grid_scores_)
        self.best_clf = clf.best_estimator_
        return self.best_clf

    def save(self, filename):

        Y_test = self.best_clf.predict_proba(self.X_Test)[:, 1]
        DataFrame = pd.read_csv("RowData\\TestData.tsv", sep='\t', quoting=3)
        name = DataFrame['id']
        df = pd.DataFrame({'id': name, 'sentiment': Y_test})
        df.to_csv(filename, index=False, quoting=3)
        print("save to "+filename)


if __name__ == '__main__':
    with open("./Persistence/Doc2VecArray.pkl","rb+") as f:
        X_train, y_train, X_test = pickle.load(f)

        
    clf = classify(X_train, y_train, X_test)
    # clf.mnb()
    # clf.save("./mnb.csv")
    clf.LR()
    clf.save("./LR.csv")
    clf.sgd()
    clf.save("./sgd.csv")
    # clf.SVMTest()
    # clf.save("./svm.csv")
    # clf.sgdboot()
    # clf.save("./sgdboot.csv")
