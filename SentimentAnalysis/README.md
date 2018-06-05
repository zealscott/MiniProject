

# 预处理

- 注意`read_csv`的时候要加`quoting = 3`
  - [pandas read_csv](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html)
- 不使用`stopwords`
- 不移除数字



# Usage

1. `run text Pre Process`

   ```python
   import PreText
   Data = TextPreProcess(DataPath, OutPath)
   Data.process()
   ```

2. ​

   ​

# Doc2Vec

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


# LSTM

- 使用`TFlearn`

  - [从代码学AI ——情感分类(LSTM on TFlearn)](https://blog.csdn.net/hitxueliang/article/details/77550819?locationNum=5&fps=1)
  - [tflearn中lstm文本分类相关实现](https://blog.csdn.net/luoyexuge/article/details/78243107)

- [使用Keras训练LSTM](https://github.com/danielmachinelearning/Doc2Vec_CNN_RNN)

- ​

  ​



# 文本预处理

1. [正则提取出HTML正文](https://blog.csdn.net/pingzi1990/article/details/41698331)
2. [replacer](https://github.com/PacktPublishing/Natural-Language-Processing-Python-and-NLTK/blob/master/Module%203/__pycache__/replacers.py)
3. [RegexReplacer](https://groups.google.com/forum/#!topic/nltk-users/BVelLz2UNww)
4. [词干提取与词性还原](https://blog.csdn.net/march_on/article/details/8935462)
5. [pos tag type](https://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa)
6. [Stemming and Lemmatization](https://www.jianshu.com/p/22be6550c18b)
7. [IMDB电影评论集](http://ai.stanford.edu/~amaas/data/sentiment/)
8. ​





# 可视化

1. ![52776363609](http://wx3.sinaimg.cn/mw690/0060lm7Tly1frupr6lqnej31gm0ptjwg.jpg)
2. tensorboard使用方法：[点击这里](https://zhuanlan.zhihu.com/p/33786815)
3. ![52776398234](http://wx1.sinaimg.cn/mw690/0060lm7Tly1frupx4my6hj30ox0m5mzj.jpg)

   ​

# 优化

1. 使用情感字典：数据集[点击这里](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon)

# 优秀代码

1. [只用机器学习](http://nbviewer.jupyter.org/github/jmsteinw/Notebooks/blob/master/NLP_Movies.ipynb)
2. ​