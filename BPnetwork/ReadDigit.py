# import tensorflow.examples.tutorials.mnist.input_data as input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import gzip
import numpy as np
import matplotlib.pyplot as plt

class ReadDigit(object):
    def __init__(self):
        self.filepath = r"MNIST_data\train-images-idx3-ubyte.gz"

    def _read32(self,bytestream):
        dt = np.dtype(np.int32).newbyteorder('>')
        return np.frombuffer(bytestream.read(4), dtype=dt)[0]
    def imagine_arr(self,filepath, index):
        with open(filepath, 'rb') as f:
            with gzip.GzipFile(fileobj=f) as bytestream:
                magic = self._read32(bytestream)
                if magic != 2051:
                    raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, f.name))
                num = self._read32(bytestream)
                rows = self._read32(bytestream)
                cols = self._read32(bytestream)
                if index >= num:
                    index = 0
                    print("wrong numbers!\n")
                bytestream.read(rows * cols * index)
                buf = bytestream.read(rows * cols)
                data = np.frombuffer(buf, dtype=np.ubyte)
                date_new = data.reshape(rows, cols)
                return date_new


    def showPic(self,number):
            im = self.imagine_arr(self.filepath, number)
            fig = plt.figure()
            plotwindow = fig.add_subplot(111)
            plt.axis('off')
            plt.imshow(im, cmap='gray')
            plt.show()
            plt.close()

    def save(self):
        result = np.zeros(shape=(60000, 784,1))
        for i in range(60000):
            data = self.imagine_arr(r"MNIST_data\\train-images-idx3-ubyte.gz".filepath,i)
            result[i] = np.reshape(data, (784, 1))


# init class
# a = ReadDigit()
# show the picture
# a.showPic(5)

# a.save()
