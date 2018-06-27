# ignore gensim warnings
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import numpy as np
from random import shuffle
import random
import pandas as pd
from datetime import datetime
import pickle


class LabeledLineSentence(object):
    """
    tag each line sentence from multi-files
    the document should be one line with space
    """

    def __init__(self, sources):
        """
        the source should be ``{filename:tag_prefix}``
        """
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        """
        convert sentence to taggedDocument for train
        """
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(
                        utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def shuffle(self):
        """
        random shuffle sentences
        """
        shuffle(self.sentences)
        return self.sentences


class Train(object):
    def __init__(self, sources):
        """
        the source should be dict: ``{filename:tag_prefix}``
        """
        self.sources = sources
        self.sentences = LabeledLineSentence(self.sources)

    def process(self, vector_size=100, min_count=1, window=10, epoch=10):
        """
        Train Doc2Vec model  
        """
        # set para according to kaggle
        self.model = Doc2Vec(min_count=min_count, window=window, size=vector_size,
                             sample=1e-3, workers=4)

        self.model.build_vocab(self.sentences.to_array())

        for i in range(epoch):
            self.model.train(self.sentences.shuffle(),
                             total_examples=self.model.corpus_count, epochs=self.model.iter)

        print("Finish Trainning model")

    def save(self, filename):
        """
        save model to disk  
        """
        self.model.save(filename)

        self.model = Doc2Vec.load(filename)

        print("save model to" + filename)

    def similar(self, word):
        """
        return the similar words  
        """
        return self.model.wv.most_similar(word)


class GetData(object):
    def __init__(self, ModelName, pos_prefix="TRAIN_POS_", neg_prefix="TRAIN_NEG_", test_prefix="TEST_"):
        """
        get the file path and prefix\n
        """
        self.ModelName = ModelName
        self.pos_prefix = pos_prefix
        self.neg_prefix = neg_prefix
        self.test_prefix = test_prefix
        self.model = Doc2Vec.load(self.ModelName)

    def GetModel(self):
        """
        return model  
        """
        return self.model

    def GetArray(self):
        """
        ``X_train``.shape = (25000,model.vector_size)\n
        ``Y_train``.shape = (25000)\n
        return [X_train, Y_train, X_test]  
        """
        X_train = np.zeros((25000, self.model.vector_size))
        Y_train = np.zeros(25000)
        for i in range(12500):
            prefix_train_pos = self.pos_prefix + str(i)
            prefix_train_neg = self.neg_prefix + str(i)
            X_train[i] = self.model[prefix_train_pos]
            X_train[12500 + i] = self.model[prefix_train_neg]
            Y_train[i] = 1
            Y_train[12500 + i] = 0

        X_test = np.zeros((25000, self.model.vector_size))
        for i in range(25000):
            prefix_test = self.test_prefix + str(i)
            X_test[i] = self.model[prefix_test]

        # combine data for dump
        alldata = []
        alldata.append(X_train)
        alldata.append(Y_train)
        alldata.append(X_test)

        with open("./Persistence/Doc2VecArray.pkl",'wb+') as f:
            pickle.dump(alldata,f,protocol=4)

        print("save to ./Persistence/Doc2VecArray")
        return X_train, Y_train, X_test


    def SaveDataForRNN(self, feature=1000):
        """
        save X_train,Y_train,X_test to ``./Persistence/``\n
        X_train.shape = (25000,feature_size,model.vector.size)\n
        Y_train.shape = (25000,2)  [1,0] for ``positive`` [0,1] for ``negative``\n
        too big to return, read data from ``./Persistence``
        """
        sources = {'CleanData\\test.txt': 'TEST', 'CleanData\\train-neg.txt': 'TRAIN_NEG',
                   'CleanData\\train-pos.txt': 'TRAIN_POS', 'CleanData\\train-unsup.txt': 'TRAIN_UNS'}

        # dict sentences_dict["TRAIN_NEG_0"] = ["a","b",..."last word"]
        sentences = LabeledLineSentence(sources)
        sentences_array = sentences.to_array()
        sentences_dict = {}
        for i in range(0, len(sentences_array)):
            sentences_dict[sentences_array[i][1][0]] = sentences_array[i][0]

        # Index2word is a list that contains the names of the words in
        # the model's vocabulary. Convert it to a set, for speed
        index2word_set = set(self.model.wv.index2word)

        # when short of word, using zero instead
        empty_word = np.zeros(self.model.vector_size,dtype='float16')


        X_train = np.zeros((25000, feature, self.model.vector_size),dtype='float16')
        Y_train = np.zeros(25000)

        # get first 1000 feature vectors from 1000 words
        for i in range(12500):
            prefix_train_pos = 'TRAIN_POS_' + str(i)
            prefix_train_neg = 'TRAIN_NEG_' + str(i)
            # length of document
            len1 = len(sentences_dict[prefix_train_pos])
            len2 = len(sentences_dict[prefix_train_neg])
            cout = j = 0
            # for pos document
            while cout < feature:
                if j < len1:
                    word = sentences_dict[prefix_train_pos][j]
                    if word in index2word_set:
                        X_train[i, cout, :] = self.model[word]
                        cout += 1
                else:
                    X_train[i, cout, :] = empty_word
                    cout += 1
                j += 1

            # reset j/cout, for neg document
            cout = j = 0
            while cout < feature:
                if j < len2:
                    word = sentences_dict[prefix_train_neg][j]
                    if word in index2word_set:
                        X_train[12500+i, cout, :] = self.model[word]
                        cout += 1
                else:
                    X_train[12500+i, cout, :] = empty_word
                    cout += 1
                j += 1

            Y_train[i] = 1
            Y_train[12500 + i] = 0

        np.save("./Persistence/LSTM/X_train",X_train)

        np.save("./Persistence/LSTM/Y_train",Y_train)

        X_test =np.zeros((25000, feature, self.model.vector_size),dtype='float16')

        for i in range(25000):
            prefix_test = 'TEST_' + str(i)
            len1 = len(sentences_dict[prefix_test])
            cout = j = 0
            while cout < feature:
                if j < len1:
                    word = sentences_dict[prefix_test][j]
                    if word in index2word_set:
                        X_test[i, cout, :] = self.model[word]
                        cout += 1
                else:
                    X_test[i, cout, :] = empty_word
                    cout += 1
                j += 1

        np.save("./Persistence/LSTM/X_test",X_test)

        print("save narray success to ./Persistence")

        # return X_train,Y_train,X_test


def LoadDataTrain():
    """
    load Train data fot ``LSTM``  
    """
    X_train = np.load("./Persistence/LSTM/X_train.npy")

    Y_train = np.load("./Persistence/LSTM/Y_train.npy")

    print("load train data success!")

    # shuffle data, maybe not necessary
    # random.seed(datetime.now())
    # index = list(range(0, X_train.shape[0]))
    # random.shuffle(index)
    # X_train = X_train[index]
    # Y_train = Y_train[index]

    return X_train, Y_train


def LoadDataTest():
    """
    load Test data fot ``LSTM``  
    """
    X_test = np.load("./Persistence/LSTM/X_test.npy")

    print("load test data success!")
    return X_test


def SaveResult(Y_test):
    """
    Y_test.shape = (25000) or (25000,1)\n
    save result to ``./result.csv``  
    """
    Y_test = Y_test.reshape(-1,)
    DataFrame = pd.read_csv("./RowData/TestData.tsv", sep='\t', quoting=3)
    name = DataFrame['id']
    df = pd.DataFrame({'id': name, 'sentiment': Y_test})
    df.to_csv("result.csv", index=False, quoting=3)
    print("save to result.csv")


def run():
    # sources = {'CleanData\\test.txt': 'TEST', 'CleanData\\train-neg.txt': 'TRAIN_NEG',
    #            'CleanData\\train-pos.txt': 'TRAIN_POS', 'CleanData\\train-unsup.txt': 'TRAIN_UNS'}

    # save model
    # test = Train(sources)
    # test.process(vector_size=300,min_count=30,window=20)
    # test.save('./Persistence/Model/imdb.d2v')

    # save for RNN
    # data = GetData("./Persistence/Model/imdb.d2v")
    # print("model vector size = %d" % (data.model.vector_size))
    # data.SaveDataForRNN(feature=700)

    # save for doc2vec
    # data = GetData("./Persistence/Model/imdb.d2v")
    # data.GetArray()

    return


if __name__ == '__main__':
    run()
