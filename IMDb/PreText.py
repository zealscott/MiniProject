import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup  
import re
# import from my file
from replacers import RegexReplacer


class TextPreProcess(object):
    """ Token/Lemmatizer/Clean text\n
        OutPath1 is positive data path\n
        OutPath2 is negative data path\n
    """

    def __init__(self, FilePath, OutPath1,OutPath2=None):
        self.FilePath = FilePath
        self.OutPath1 = OutPath1
        self.OutPath2 = OutPath2
        self.DataFrame = pd.read_csv(FilePath, sep='\t', quoting=3)

        # split paragraph to sentence class
        self.PunktTokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
        # remove HTML tag pattern
        self.rc = re.compile(r"\<.*?\>")
        # Replacer class
        self.replacer = RegexReplacer()
        # split sentence into word
        pattern = r'[\d.,]+|[A-Z][.A-Z]+\b\.*|\w+|\S'
        self.tokenizer = RegexpTokenizer(pattern)
        # Lemmatizer
        self.wordnet_lemmatizer = WordNetLemmatizer()

    def SplitPhase(self, row):
        """ split paragraph to sentence """
        return self.PunktTokenizer.tokenize(row['review'])

    def RemoveHTML(self, row):
        """ remove HTML tags """
        return [BeautifulSoup(sentence,"lxml").get_text() for sentence in row['review']]

    def ReplaceAbbre(self, row):
        """ Replace abbreviation """
        return [self.replacer.replace(sentence) for sentence in row['review']]

    def SplitSent(self, row):
        """ split sentence to words """
        return [self.tokenizer.tokenize(sentence) for sentence in row['review']]

    def lemma(self, tags):
        """ lemmatizer for tagged words """
        WORD = []
        for word, tag in tags:
            wntag = tag[0].lower()
            wntag = wntag if wntag in ['a', 'r', 'n', 'v', 'n', 's'] else None
            if not wntag:
                lemma = word
            else:
                lemma = self.wordnet_lemmatizer.lemmatize(word, wntag)
            WORD.append(lemma)
        return WORD

    def Lemmatizer(self, row):
        """ Lemmatizer words use WordNet """
        return [self.lemma(nltk.pos_tag(sentence)) for sentence in row['review']]

    def CleanWords(self, sentence):
        """ remove len < 3 and non alpha and lowercase """
        if self.word2vector:
            return [word.lower() for word in sentence if (word.isalpha() and len(word) >= 2) or word.isdigit()]
        else:
            stops = set(stopwords.words("english"))
            return [word.lower() for word in sentence if len(word) >= 3 and word.isalpha() and not word in stops]

    def CleanSentences(self, row):
        """ clean sentences """
        return [self.CleanWords(sentence) for sentence in row['review']]
    
    def ToStr(self,row):
        str=""
        for sentence in row['review']:
            for word in sentence:
                str += (word + " ")
        return str[:-1]

    def process(self, word2vector=True):
        """ Remove HTML tags, Replace Abbre, Split into words
            if use word2vector, should not use ``stopwords ``
        """
        self.word2vector = word2vector
        # split phase
        self.DataFrame['review'] = self.DataFrame.apply(
            self.SplitPhase, axis=1)
        # remove HTML tags
        self.DataFrame['review'] = self.DataFrame.apply(
            self.RemoveHTML, axis=1)
        # replace abbre
        self.DataFrame['review'] = self.DataFrame.apply(
            self.ReplaceAbbre, axis=1)
        # split sentences
        self.DataFrame['review'] = self.DataFrame.apply(self.SplitSent, axis=1)
        # lemmatizer
        self.DataFrame['review'] = self.DataFrame.apply(
            self.Lemmatizer, axis=1)
        # clean sentences
        self.DataFrame['review'] = self.DataFrame.apply(
            self.CleanSentences, axis=1)
        # convert list to str
        self.DataFrame['review'] = self.DataFrame.apply(
            self.ToStr, axis=1)


    def save(self,Label=False):
        if Label:
            a =self.DataFrame['review'][self.DataFrame.sentiment==1]
            a.to_csv(self.OutPath1,index=False)
            b =self.DataFrame['review'][self.DataFrame.sentiment==0]
            b.to_csv(self.OutPath2,index=False)
            print("save data success to "+ self.OutPath1 + " and " + self.OutPath2)
        else:
            # drop column and save
            self.DataFrame.drop(columns=['id']).to_csv(self.OutPath1, index=False,header=False)
            print("save to" + self.OutPath1)


if __name__ == '__main__':
    LabelTrainDataPath = "RowData\\LabeledTrainData.tsv"
    unLabelTrainDataPath = "RowData\\UnlabeledTrainData.tsv"
    testDataPath = "RowData\\TestData.tsv"

    labeltrain = TextPreProcess(
        LabelTrainDataPath, "CleanData\\train-pos.txt","CleanData\\train-neg.txt")
    labeltrain.process()
    labeltrain.save(Label=True)

    unlabeltrain = TextPreProcess(
        unLabelTrainDataPath, "CleanData\\train-unsup.txt")
    unlabeltrain.process()
    unlabeltrain.save()

    testData = TextPreProcess(testDataPath, "CleanData\\test.txt")
    testData.process()
    testData.save()