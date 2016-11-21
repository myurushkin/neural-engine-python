import os
import codecs
import array
import struct
import numpy as np
import nn
import urllib
import gzip

from keras.models import Sequential                                    
from keras.layers import Dense, Activation, Dropout                   
from keras.optimizers import SGD                              
from keras.utils.np_utils import to_categorical

def download_and_unpack_file(url):
    name_packed = os.path.basename(url)
    name = os.path.splitext(name_packed)[0]
    if not os.path.exists(name):
        print name
        urllib.urlretrieve(url, name_packed)
        fi = gzip.open(name_packed, 'rb')
        fo = codecs.open(name, 'wb')
        fo.write(fi.read())
        fi.close()
        fo.close()


def load(samples_filename, labels_filename):
    labels_file = codecs.open(labels_filename, 'rb')
    magic, labels_size = struct.unpack('>II', labels_file.read(8))
    assert magic == 2049
    labels_array = array.array('B', labels_file.read())

    samples_file = codecs.open(samples_filename, 'rb')
    magic, samples_size, rows, cols = struct.unpack('>IIII', samples_file.read(16))
    assert magic == 2051
    assert labels_size == samples_size
    samples_array = array.array('B', samples_file.read())

    samples = np.zeros(shape = (samples_size, rows, cols), dtype=np.uint8)
    labels = np.zeros(shape = (samples_size, 1), dtype = np.uint8)

    for i in xrange(samples_size):
        labels[i] = labels_array[i]
        samples[i] = np.array(samples_array[i*rows*cols: (i+1)*rows*cols]).reshape(rows, cols)
    return samples, labels


def preprocess(X, y):
    X = X.astype('float32').reshape(X.shape[0], X.shape[1] * X.shape[2])
    X /= 255
    y = to_categorical(y, 10)
    idxs = np.random.permutation(np.arange(X.shape[0]))
    return X[idxs,:],y[idxs,:]


def read_data():
    download_and_unpack_file('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
    download_and_unpack_file('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
    download_and_unpack_file('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
    download_and_unpack_file('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')

    test_images_filename = 't10k-images-idx3-ubyte'
    test_labels_filename = 't10k-labels-idx1-ubyte'
    train_images_filename = 'train-images-idx3-ubyte'
    train_labels_filename = 'train-labels-idx1-ubyte'

    test_images, test_labels = load(test_images_filename, test_labels_filename)
    train_images, train_labels = load(train_images_filename, train_labels_filename)

    n_train = len(train_labels)
    n_test = len(test_labels)

    X_train, y_train = preprocess(train_images, train_labels)
    X_test, y_test = preprocess(test_images, test_labels)
    return X_train, y_train, X_test, y_test


def test_keras(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(Dense(512, input_dim=len(X_train[0]), init='uniform'))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.3))
    model.add(Dense(512, init='uniform'))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.3))
    model.add(Dense(10, init='uniform'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, momentum=0.0, decay=0.0)
    model.compile(loss='categorical_crossentropy',
            optimizer=sgd,
            metrics=['accuracy'])

    history = model.fit(X_train, y_train,
            nb_epoch=40,
            batch_size=128,
            verbose=1,
            validation_data=(X_test, y_test))



def test_nn(X_train, y_train, X_test, y_test):
    model = nn.NeuralNet()
    model.add(nn.DenseLayer(512))
    model.add(nn.SigmoidLayer())
    model.add(nn.DropoutLayer(0.3))

    model.add(nn.DenseLayer(512))
    model.add(nn.SigmoidLayer())
    model.add(nn.DropoutLayer(0.3))

    model.add(nn.DenseLayer(10))
    model.add(nn.SoftmaxLayer())

    my_history = model.fit(X_train, y_train, num_epochs=20,\
                        learning_rate=0.01, batch_size=128,\
                        X_test=X_test, y_test=y_test)
    predictions = model.predict(X_test)
    predictions = np.argmax(predictions, axis=0)
    labels = np.argmax(y_test, axis=1)
    print "accuracy of my model: {}".format(sum(predictions == labels)*1.0/len(predictions))


X_train, y_train, X_test, y_test = read_data()
test_keras(X_train, y_train, X_test, y_test)
test_nn(X_train, y_train, X_test, y_test)