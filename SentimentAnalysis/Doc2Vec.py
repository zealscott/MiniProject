# ignore gensim warnings
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import pickle
import numpy as np
from random import shuffle
import random
import pandas as pd
from datetime import datetime


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

    def process(self, vector_size=100, min_count=1, window=10, epoch=50):
        """
        Train Doc2Vec model  
        """
        self.model = Doc2Vec(min_count=min_count, window=window, size=vector_size,
                             sample=1e-4, negative=5, workers=4)

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
    def __init__(self, ModelName, pos_prefix="TRAIN_POS_", neg_prefix="TRAIN_NEG_", test_prefix="TEST_", ):
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

    def GetArray(self,shuffle = False):
        """
        one vector (100,1) for one doc\n
        ``train_arrays``.shape = (25000,100)\n
        ``train_label``.shape = (25000,1)\n
        return [train_arrays,train_labels,test_arrays]  
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

        if not shuffle:
            return X_train,Y_train,X_test
        
        # 5000个测试数据， 其余验证数据+训练数据
        indexList = list(range(0, X_train.shape[0]))
        random.seed(1)
        random.shuffle(indexList)

        # 取前20000个为训练集，后5000个为测试集
        testIndex = indexList[-5000:]
        trainIndex = indexList[:-5000]

        trainData = X_train[trainIndex]
        trainLabel = Y_train[trainIndex]

        testData = X_train[testIndex]
        testLabel = Y_train[testIndex]

        return trainData, trainLabel, testData, testLabel, X_test

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

        # data for RNN get first 1000 feature vectors from 1000 words
        X_train = np.zeros(
            shape=(25000, feature, self.model.vector_size)).astype('float16')
        # Y_train = np.zeros(shape=(25000, 2)).astype('float16')
        Y_train = np.zeros(25000)

        # one-hot for Y_train [1][0] for positive [0][1] for negative
        # pos = np.zeros(shape=(1, 2)).astype('float16')
        # pos[0][0] = 1
        # neg = np.zeros(shape=(1, 2)).astype('float16')
        # neg[0][1] = 1

        # when short of word, using zero instead
        empty_word = np.zeros(self.model.vector_size).astype('float16')

        # get first 1000 feature vectors from 1000 words
        for i in range(12500):
            prefix_train_pos = 'TRAIN_POS_' + str(i)
            prefix_train_neg = 'TRAIN_NEG_' + str(i)
            len1 = len(sentences_dict[prefix_train_pos])
            len2 = len(sentences_dict[prefix_train_neg])
            for j in range(feature):
                if j < len1:
                    # have enough word
                    X_train[i, j, :] = self.model[sentences_dict[prefix_train_pos][j]]
                else:
                    X_train[i, j, :] = empty_word

                if j < len2:
                    X_train[12500+i, j,
                            :] = self.model[sentences_dict[prefix_train_neg][j]]
                else:
                    X_train[12500+i, j, :] = empty_word

            Y_train[i] = 1
            Y_train[12500 + i] = 0
            # Y_train[i, :] = pos
            # Y_train[12500 + i, :] = neg

        with open('./Persistence/X_train.pickle', 'wb') as f:
            pickle.dump(X_train, f, protocol=4)
            f.close()

        with open('./Persistence/Y_train.pickle', 'wb') as f:
            pickle.dump(Y_train, f, protocol=4)
            f.close()

        X_test = np.zeros(
            shape=(25000, feature, self.model.vector_size)).astype('float16')

        for i in range(25000):
            prefix_test = 'TEST_' + str(i)
            len1 = len(sentences_dict[prefix_test])
            for j in range(feature):
                if j < len1:
                    X_test[i, j, :] = self.model[sentences_dict[prefix_test][j]]
                else:
                    X_test[i, j, :] = empty_word

        with open('./Persistence/X_test.pickle', 'wb') as f:
            pickle.dump(X_test, f, protocol=4)
            f.close()

        print("save narray success to ./Persistence")

        # return X_train,Y_train,X_test

def LoadDataTrain():
    """
    load Train data fot ``LSTM``  
    """
    with open('./Persistence/X_train.pickle', 'rb') as f:
        X_train = pickle.load(f)
        f.close()
    with open('./Persistence/Y_train.pickle', 'rb') as f:
        Y_train = pickle.load(f)
        f.close()
    print("load train data success!")

    # random
    random.seed(datetime.now())
    index = list(range(0, X_train.shape[0])) 
    random.shuffle(index)   
    X_train = X_train[index]  
    Y_train = Y_train[index]  

    return X_train,Y_train

def LoadDataTest():
    """
    load Test data fot ``LSTM``  
    """
    with open('./Persistence/X_test.pickle', 'rb') as f:
        X_test = pickle.load(f)
        f.close()
    print("load test data success!")
    return X_test

def SaveResult(Y_test):
    """
    Y_test.shape = (25000) or (25000,2)\n
    save result to ``./result.csv``  
    """
    if Y_test.ndim == 2:
        result = np.zeros(shape=(25000)).astype('int')
        for i in range(25000):
            a = Y_test[i][0]
            b = Y_test[i][1]
            result[i] = (1 if a > b else 0)
        Y_test = result
    testDataPath = "RowData\\TestData.tsv"
    DataFrame = pd.read_csv(testDataPath, sep='\t', quoting=3)
    name = DataFrame['id']
    df = pd.DataFrame({'id': name, 'sentiment': Y_test})
    df.to_csv('result.csv', index=False)
    print("save to ./result.csv")


def run():
    sources = {'CleanData\\test.txt': 'TEST', 'CleanData\\train-neg.txt': 'TRAIN_NEG',
               'CleanData\\train-pos.txt': 'TRAIN_POS', 'CleanData\\train-unsup.txt': 'TRAIN_UNS'}

    test = Train(sources)
    test.process(vector_size=100,window=10)
    test.save('.\\Model\\imdb.d2v')

    # data = GetData("./Model/imdb.d2v")
    # print("model vector size = %d"%(data.model.vector_size))
    # data.SaveDataForRNN()
    return


if __name__ == '__main__':
    run()
    