import re
import numpy as np
from bs4 import BeautifulSoup
import pickle
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Bidirectional, Embedding
from keras.layers.convolutional import MaxPooling1D, Conv1D
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.models import load_model
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
import xgboost as xgb


class Preprocessing():
    def __init__(self, TrainDataPath, TestDataPath, Unlabeled=None):
        self.Unlabeled = Unlabeled
        self.train = pd.read_csv(
            TrainDataPath, header=0, delimiter="\t", quoting=3)
        self.test = pd.read_csv(TestDataPath, header=0,
                                delimiter="\t", quoting=3)
        self.y_train = self.train['sentiment']

    def ReviewToWordlist(self, review):
        '''
        Meant for converting each of the IMDB reviews into a list of words.
        '''
        # remove the HTML.
        review_text = BeautifulSoup(review, "lxml").get_text()

        # left only words
        review_text = re.sub("[^a-zA-Z]", " ", review_text)

        # Convert words to lower case and split them into separate words.
        words = review_text.lower().split()

        # Return ["word","word",...,"last word"]
        return(words)

    def DataClean(self):
        """  
        return clean by list of list
        """
        self.traindata = []
        for i in range(0, len(self.train['review'])):
            self.traindata.append(
                " ".join(self.ReviewToWordlist(self.train['review'][i])))
        self.testdata = []
        for i in range(0, len(self.test['review'])):
            self.testdata.append(
                " ".join(self.ReviewToWordlist(self.test['review'][i])))

    def tfidf(self, ngram=2):
        tfv = TFIV(min_df=3,  max_features=None,
                   strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                   ngram_range=(1, ngram), use_idf=1, smooth_idf=1, sublinear_tf=1,
                   stop_words='english')

        # Combine both to fit the TFIDF vectorization.
        X_all = self.traindata + self.testdata
        lentrain = len(self.traindata)

        tfv.fit(X_all)  # This is the slow part!
        X_all = tfv.transform(X_all)

        self.X = X_all[:lentrain]  # Separate back into training and test sets.
        self.X_test = X_all[lentrain:]

        print("vectorization data size: ", self.X.shape)
        return self.X, self.y_train, self.X_test


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
        grid_values = {'C': [30]}

        clf = GridSearchCV(LR(penalty='l2', dual=True, random_state=0),
                           grid_values, scoring='roc_auc', cv=20)
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
        kfold = StratifiedKFold(n_splits=5, shuffle=True)

        parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                       'C': [1e-3, 1e-2,0.1,1]},
                      {'kernel': ['linear'], 'C': [1e-3, 1e-2,0.1,1]}]
        clf = GridSearchCV(
            SVC(probability=True),
            parameters,
            cv=kfold,
            scoring="roc_auc",
            n_jobs=4
        )
        clf.fit(self.X_train, self.Y_Train)
        print("using SVM, Best: %f using %s" %
              (clf.best_score_, clf.best_params_))
        self.best_clf = clf.best_estimator_
        return self.best_clf

    def SVM(self):
        clf = SVC()
        clf.fit(self.X_train, self.Y_Train)
        score = clf.score(self.X_train, self.Y_Train)
        print("using SVM, score = %f" % score)
        self.best_clf = clf
        return self.best_clf

    def sgd(self):
        # Regularization parameter
        sgd_params = {'alpha': [0.00006, 0.00007, 0.00008, 0.0001, 0.0005]}

        clf = GridSearchCV(SGD(max_iter=5, random_state=0, shuffle=True, loss='modified_huber'),
                           sgd_params, scoring='roc_auc', cv=20)  # Find out which regularization parameter works the best.

        clf.fit(self.X_train, self.Y_Train)
        print("using SGD, Best: %f using %s" %
              (clf.best_score_, clf.best_params_))
        self.best_clf = clf.best_estimator_
        return self.best_clf

    def sgdboot(self):
        cv_params = {'max_depth': [3, 5, 7], 'min_child_weight': [1, 3, 5]}
        ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8,
                      'objective': 'binary:logistic'}
        clf = GridSearchCV(xgb.XGBClassifier(**ind_params),
                           cv_params,
                           scoring='accuracy', cv=5, n_jobs=4)
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


def MyLSTM(max_features=25000, embedding_size=128, maxlen=2000):
    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    # add conv1D layer
    model.add(Conv1D(filters=32, kernel_size=3, padding='same',
                     activation='relu', input_shape=(1000, 100)))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.2))
    # add conv1D layer
    model.add(Conv1D(filters=32, kernel_size=3,
                     padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.2))
    # add conv1D layer
    model.add(Conv1D(filters=32, kernel_size=3,
                     padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.2))
    # add conv1D layer
    model.add(Conv1D(filters=32, kernel_size=3,
                     padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.2))
    # add LSTM layer
    model.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2))
    # model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # model complete
    print(model.summary())
    return model


if __name__ == '__main__':
    # process = Preprocessing(
    #     "./RowData/LabeledTrainData.tsv", "./RowData/TestData.tsv")
    # process.DataClean()
    # AllData= process.tfidf(ngram=4)

    # X, y_train, X_test = AllData[0],AllData[1],AllData[2]
    with open("./Persistence/ngram.pickle", "rb") as f:
        X, y_train, X_test = pickle.load(f)
        f.close()

    # model = MyLSTM(maxlen=X.shape[1])
    # model.fit(X, y_train, validation_split=0.2, epochs=10, batch_size=64)
    # Y_test = model.predict(X_test)
    # print(Y_test)
    # print(Y_test.ndim)

    clf = classify(X, y_train, X_test)
    # clf.mnb()
    # clf.save("./mnb.csv")
    # clf.LR()
    # clf.save("./LR.csv")
    # clf.sgd()
    # clf.save("./sgd.csv")
    clf.SVMTest()
    clf.save("./svm.csv")
    # clf.sgdboot()
    # clf.save("./sgdboot.csv")
