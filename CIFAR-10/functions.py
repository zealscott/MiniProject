import pickle
import numpy as np
from keras.utils import np_utils
from matplotlib import pyplot as plt

path = "data/"  # Path to data

# Height or width of the images (32 x 32)
size = 32

# 3 channels: RGB
channels = 3

# Number of classes
num_classes = 10

# Each file contains 10000 images
image_batch = 10000

# 5 training files
num_files_train = 5

# Total number of training images
images_train = image_batch * num_files_train


def unpickle(file):

    with open(path + file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')

    # Dictionary with images and labels
    return dict


def convert_images(raw_images):
    """
    Convert images to numpy arrays\n
    normalize it\n
    return [image_number, height, width, channel]
    """

    # Convert raw images to numpy array and normalize it
    raw = np.array(raw_images, dtype=float) / 255.0

    # Reshape to 4-dimensions - [image_number, channel, height, width]
    images = raw.reshape([-1, channels, size, size])

    # [image_number, channel, height, width] = [0,1,2,3]
    # convert to [image_number, height, width,channel]
    images = images.transpose([0, 2, 3, 1])

    return images


def load_data(file):
    """  
    Load file, unpickle\n
    return return images with their labels
    """
    data = unpickle(file)

    images_array = data[b'data']

    # Convert image and labels
    images = convert_images(images_array)
    labels = np.array(data[b'labels'])

    return images, labels


def get_test_data():
    images, labels = load_data(file="test_batch")

    # return images, their labels and one-hot vectors
    return images, labels, np_utils.to_categorical(labels, num_classes)


def get_train_data():
    """ 
    Load all training data in 5 files\n
    return images, labels and one-hot label
    """

    images = np.zeros(shape=[images_train, size, size, channels], dtype=float)
    labels = np.zeros(shape=[images_train], dtype=int)

    # Starting index of training dataset
    batch_start = 0

    for i in range(num_files_train):
        images_batch, labels_batch = load_data(file="data_batch_" + str(i+1))

        batch_end = batch_start + image_batch

        images[batch_start:batch_end, :] = images_batch
        labels[batch_start:batch_end] = labels_batch

        batch_start = batch_end

    return images, labels, np_utils.to_categorical(labels, num_classes)


def get_class_names():
    """ return a list of class names """
    raw = unpickle("batches.meta")[b'label_names']

    names = [x.decode('utf-8') for x in raw]

    return names


def save_result(predict_labels, class_names):
    """ save result to ``./result.csv ``"""
    import pandas as pd
    d = {'id': range(1, len(predict_labels)+1),
         'label': [class_names[i] for i in predict_labels]}
    df = pd.DataFrame(data=d)
    df.to_csv("result.csv",index=False)
    print("save result success.")


# following functions are used for plot images

def plot_images(images, true_labels, class_names, labels_predict=None):
    """
    plot 9 images from decode data  with or without predict labels
    """
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))

    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], interpolation='spline16')

        true_labels_name = class_names[true_labels[i]]

        # Show true and predicted classes
        if labels_predict is None:
            xlabel = "True: "+true_labels_name
        else:
            # Name of the predicted class
            labels_predict_name = class_names[labels_predict[i]]

            xlabel = "True: " + true_labels_name + "\n" + "Predicted: " + labels_predict_name

        ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def plot_model(model_info):
    """ plot model accuracy using keras model informations """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # plot model accuracy
    axs[0].plot(range(1, len(model_info.history['acc'])+1),
                model_info.history['acc'])
    axs[0].plot(range(1, len(model_info.history['val_acc'])+1),
                model_info.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(
        1, len(model_info.history['acc'])+1), len(model_info.history['acc'])/10)
    axs[0].legend(['train', 'validation'], loc='best')
    # plot model loss
    axs[1].plot(range(1, len(model_info.history['loss'])+1),
                model_info.history['loss'])
    axs[1].plot(range(1, len(model_info.history['val_loss'])+1),
                model_info.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(
        1, len(model_info.history['loss'])+1), len(model_info.history['loss'])/10)
    axs[1].legend(['train', 'validation'], loc='best')

    plt.show()


def error_plot(test_images, test_labels, class_names, predict_labels, correct):
    """ plot error classification images """
    incorrect = (correct == False)

    images_error = test_images[incorrect]

    labels_error = predict_labels[incorrect]

    labels_true = test_labels[incorrect]

    # plot first 9 images
    plot_images(images=images_error[0:9],
                true_labels=labels_true[0:9],
                class_names=class_names,
                labels_predict=labels_error[0:9])


def predict_classes(model, images_test, labels_test):
    """ using model to get test data predictcions """
    class_pred = model.predict(images_test, batch_size=32)

    labels_predict = np.argmax(class_pred, axis=1)

    # Boolean array that tell if predicted label is the true label
    correct = (labels_predict == labels_test)

    return correct, labels_predict
