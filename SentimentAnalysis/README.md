使用普通的机器学习，本项目的准确率能达到95%。

使用LSTM神经网络，准确率能达到97%，在kaggle上排名前1%。

# 数据介绍

- `PreText.py`
  - 对数据进行预处理，并将其分类保存为txt文档。
  - 其中使用了`replacers.py`对缩略词进行处理。
- `TFIDF.py`
  - 使用`TF-IDF`对数据进行向量化，并使用基本的机器学习方法进行训练。
- `Doc2Vec.py`
  - 使用`Doc2Vec`方法对文档进行向量化，并将`numpy`数组保存到磁盘。
- `ML.py`
  - 使用机器学习的方法对`Doc2Vec`的向量化数组进行训练。
- `LSTM.py`
  - 使用`LSTM`深度学习网络进行训练。
- `Visualize.py`
  - 使用`TensorBoard`对训练的`Doc2Vec`模型可视化。
- `/Persistence`
  - 保存持久化数据
- `/Reference`
  - 参考文献
- `jupyternotebook`
  - 以上步骤的可视化代码
- `result.csv`
  - 最终的预测结果

# Reference

## 文本预处理

1. [正则提取出HTML正文](https://blog.csdn.net/pingzi1990/article/details/41698331)
2. [replacer](https://github.com/PacktPublishing/Natural-Language-Processing-Python-and-NLTK/blob/master/Module%203/__pycache__/replacers.py)
3. [RegexReplacer](https://groups.google.com/forum/#!topic/nltk-users/BVelLz2UNww)
4. [词干提取与词性还原](https://blog.csdn.net/march_on/article/details/8935462)
5. [pos tag type](https://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa)
6. [Stemming and Lemmatization](https://www.jianshu.com/p/22be6550c18b)
7. [IMDB电影评论集](http://ai.stanford.edu/~amaas/data/sentiment/)

## Doc2Vec

- 利用Doc2Vec的改进

  - [Sentiment Analysis Using Doc2Vec](http://linanqiu.github.io/2015/10/07/word2vec-sentiment/)
- Kaggle针对Word2vector

  - [kaggle tutorial](https://www.kaggle.com/c/word2vec-nlp-tutorial#part-2-word-vectors)
- Gensim发明者写的

  - [Doc2vec tutorial](https://rare-technologies.com/doc2vec-tutorial)
- Github上的一篇，但没太看懂

  - [Gensim Doc2vec Tutorial on the IMDB Sentiment Dataset](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb)
- Kaggle上的讨论

  - [Using Doc2Vec from gensim.](https://www.kaggle.com/c/word2vec-nlp-tutorial/discussion/12287)
- gensim官方参数文档

  - [Deep learning with paragraph2vec](https://radimrehurek.com/gensim/models/doc2vec.html)
  - [IMDB](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb)
- 关于参数`negative sampling`

  - [Negative Sampling](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)
- [关于window的调参](https://stackoverflow.com/questions/22272370/word2vec-effect-of-window-size-used)


## LSTM

- 使用`TFlearn`

  - [从代码学AI ——情感分类(LSTM on TFlearn)](https://blog.csdn.net/hitxueliang/article/details/77550819?locationNum=5&fps=1)
  - [tflearn中lstm文本分类相关实现](https://blog.csdn.net/luoyexuge/article/details/78243107)

- [使用Keras训练LSTM](https://github.com/danielmachinelearning/Doc2Vec_CNN_RNN)




## 可视化

- tensorboard使用方法：[点击这里](https://zhuanlan.zhihu.com/p/33786815)

## 优秀代码

- [只用机器学习](http://nbviewer.jupyter.org/github/jmsteinw/Notebooks/blob/master/NLP_Movies.ipynb)