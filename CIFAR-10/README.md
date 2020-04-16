# 数据介绍

`CIFAR-10` 由60000张32*32的 RGB 彩色图片构成，共10个分类：飞机；汽车；鸟；猫；鹿；狗；青蛙；马；船；卡车。

50000张训练，10000张测试（交叉验证）。这个数据集最大的特点在于将识别迁移到了普适物体，而且应用于多分类。

本项目所有的代码都使用Google提供的免费GPU：[Colab](https://colab.research.google.com/)运行。

本文最终达到的测试集准确率为89.17%。

# 组织结构

- `functions.py`

  主要是一些预处理以及可视化函数，用于对原始数据的提取和可视化。

- `Data Preprocess.ipynb`

  预处理的主要步骤（结果使用HTML展示在`result`中的同名文件中）。

- `naive_CNN.ipynb`

  使用简单的卷积神经网络训练数据，可视化结果。

- `ALL_CNN.ipynb`

  使用`All convolutional net`以及`Data Augment`进行改进。

- `Good_init.ipynb`

  参考论文《[All you need is a good init](https://arxiv.org/abs/1511.06422)》对初始化权重，网络结构进行改进。

- `lsuv_init.py`

  参考论文的`keras`实现，源文件在[Github上](https://github.com/ducha-aiki/LSUV-keras)。

- `/reference`

  对于所参考的论文，给出了PDF格式。

- `/model`

  保存神经网络训练中的最好模型。

- `/result`

  `jupyter notebook`的HTML格式，方便查看训练过程。

  `result.csv`存放最后10000张test照片的识别信息。

# 数据预处理

## 训练数据

主要API接口为`get_train_data()`。

将5个训练数据使用`pickle`进行提取和reshape，使得其变为`images_train, size, size, channels`这样的四维数组。并将其存入同一个`numpy`数组中进行存储。

最后，将对应的`labels`使用`one hot`方法向量化，作为神经网络的输入。

一共有50000个测试数据。

## 测试数据

主要API接口为`get_test_data()`。

与训练数据类似，将`test_batch`按照`pickle`规范进行提取，将对应的`labels`使用`one hot`方法向量化，作为神经网络的输入。

一共有10000个测试数据。

## 标签名称

主要API接口为`get_class_names()`。

将`batches.meta`中的标签名称使用`list`数组返回，方便之后可视化。

## 画图相关

- `predict_classes(model, images_test, labels_test)`
  - 根据模型得出预测信息，以及与真实标签对比的Bool数组。

- plot_images(images, true_labels, class_names, labels_predict=None)`
  - 绘制最多9副图像，若不包含神经网络的预测数据，则只绘制出真实的标签。

- `plot_model(model_info)`
  - 绘制由`Kears`训练处的神经网络的模型准确率与损失随着迭代次数的变化而变化的图像，方便分析模型的过拟合和鲁棒性。

- `error_plot(test_images, test_labels, class_names, predict_labels, correct)`
  - 绘制出神经网络训练处的错误图像信息。


# Naive CNN

对于多分类图像识别，第一个想到的就是CNN。

在`naive_CNN.ipynb`中，实现了基本的卷积神经网络。

其结构主要包括4层卷积神经网络，2层全连接层，使用最大池化和Dropout技术：

```python
def build_model():
    
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMAGE_SIZE,IMAGE_SIZE,CHANNELS)))    
    model.add(Conv2D(32, (3, 3), activation='relu'))    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.summary()
    return model
```

优化器和损失函数选择：

```python
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics = ['accuracy']) 
```

其中，`Adam`一般是较好的优化器选择（默认），而由于我们是多分类问题，采用`categorical_crossentropy`是更好的选择（[参考文献](https://jamesmccaffrey.wordpress.com/2013/11/05/why-you-should-use-cross-entropy-error-instead-of-classification-error-or-mean-squared-error-for-neural-network-classifier-training/)中解释了为什么交叉熵是更好的损失函数）。

在经过调参和训练迭代100次之后，最好的准确率达到了80%左右。

可以发现，其神经网络包含了一百多万的参数，但我们的训练数据只有5万个，很容易出现过拟合。将模型在训练集和测试集中的准确率和`Loss`绘制出来可以看见：

![](http://wx3.sinaimg.cn/mw690/0060lm7Tly1fsgmux6x3nj30v10bn75f.jpg)

模型在训练集上表现很好，而在测试集上随着迭代次数的增加，`Loss`并没有明显减少，发生了过拟合问题。

我们可以查看网络训练出错的图片：

![](http://wx3.sinaimg.cn/mw690/0060lm7Tly1fsgmv7dtv9j30g00gqn2i.jpg)

# 改进的CNN

## 图片生成器

我们在之前发现有明显的过拟合问题，这是由于我们训练集的样本太少，而网络过于庞大造成的。但一般来说，更大的网络具有更好的特性，因此我考虑如何增加样本的数量。

对于图像来说，一般的处理方法

- 对图像进行随机的左右翻转；
- 随机变换图像的亮度；
- 随机变换图像的对比度；
- 图片会进行近似的白化处理。

本文采用`kears`自带的`ImageDataGenerator`进行图片生成，它用以生成一个`batch`的图像数据，支持实时数据提升。训练时该函数会无限生成数据，直到达到规定的`epoch`次数为止。

具体代码为：

```python
datagen = ImageDataGenerator(
    rotation_range=45,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(images_train)
```

## All convolutional net

对于一般的CNN，我们使用卷积层和最大池化，最后再加上一个小的全连接网络进行输出。而这篇参考文献《[STRIVING FOR SIMPLICITY:THE ALL CONVOLUTIONAL NET](https://arxiv.org/pdf/1412.6806.pdf)》采用了一种新的`deconvolution approach`，只使用卷积层，最后使用`global averaging pooling layer`代替`fully-connected layer`。

论文在`CIFAR-10`数据集上进行试验，取得了较好的效果。本文根据论文思路，重构了论文的网络框架。具体实现为：

```python
def cnn_model():
    
    model = Sequential()
    
    model.add(Conv2D(96, (3, 3), activation='relu', padding = 'same', input_shape=(IMAGE_SIZE,IMAGE_SIZE,CHANNELS)))    
    model.add(Dropout(0.2))
    
    model.add(Conv2D(96, (3, 3), activation='relu', padding = 'same'))  
    model.add(Conv2D(96, (3, 3), activation='relu', padding = 'same', strides = 2))    
    model.add(Dropout(0.5))
    
    model.add(Conv2D(192, (3, 3), activation='relu', padding = 'same'))    
    model.add(Conv2D(192, (3, 3), activation='relu', padding = 'same'))
    model.add(Conv2D(192, (3, 3), activation='relu', padding = 'same', strides = 2))    
    model.add(Dropout(0.5))    
    
    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (1, 1),padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(10, (1, 1), padding='valid'))
    model.add(GlobalAveragePooling2D())   
    model.add(Activation('softmax'))

    model.summary()
    return model
```

其中参数和网络的设置都是完全参考论文，自己也还没琢磨透。。。

## 效果展示

使用`Data augment`和`ALL CONVOLUTIONAL NET`，我们得到了86.85%的准确率，较之前已经有的比较大的提升。

![](http://wx3.sinaimg.cn/mw690/0060lm7Tly1fsoeoc5p1sj30ut0bm75p.jpg)

# 改进初始化权重

参考了ICLR2016《[All you need is a good init](https://arxiv.org/abs/1511.06422)》这篇文章。文章阐述了`CNN`学习中参数初始化对最总学习效果的影响，并提出了结合了`Batch Normalization`效果并适合`ReLU`的初始化方式，在多个数据集上展现了良好的效果。

这篇文章提到了使用新的初始化权重方法，称为`Layer-sequential unit-variance (LSUV) initialization`，这种方法在GitHub上已经有实现：我参考其中了[Kears实现的代码](https://github.com/ducha-aiki/LSUV-keras)。

同时，使用更激进的`ImageDataGenerator`方法，重新设计了网络结构，达到了16层，主要网络设计如下：

```python
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(48, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(48, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(80, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(80, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(80, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(80, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(80, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(GlobalMaxPooling2D())
model.add(Dropout(0.25))

model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
```

由于网络太深，训练了将近两天，最终虽然没有达到论文中的效果，但准确率也有89.17%，目测再进行调参可以上到90%，但由于使用的Google的GPU，并不稳定，因此没有再尝试成功。

# Todo

- 对于图像预处理，增加白化等技术，进一步扩大训练集。
- 使用更深的`Deep Residual Learning`进行训练。

# Reference

## python使用技巧

- [快速转换为ont -hot 数组](https://keras.io/utils/)

- [只用一个循环快速画图](https://stackoverflow.com/questions/46862861/what-does-axes-flat-in-matplotlib-do)

- [使用checkpoint在每次`epoch`保存当前最好模型](https://keras.io/callbacks/)

## 神经网络选择与技巧

- [为什么使用交叉熵代价函数](https://jamesmccaffrey.wordpress.com/2013/11/05/why-you-should-use-cross-entropy-error-instead-of-classification-error-or-mean-squared-error-for-neural-network-classifier-training/)
- [图片生成器ImageDataGenerator](http://keras-cn.readthedocs.io/en/latest/preprocessing/image/)

## 神经网络实现参考

- [STRIVING FOR SIMPLICITY:THE ALL CONVOLUTIONAL NET](https://arxiv.org/pdf/1412.6806.pdf)
- [All you need is a good init](https://arxiv.org/abs/1511.06422)

