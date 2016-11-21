import numpy as np
import random

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

class DenseLayer:
    def __init__(self, output_dim):
        self.output_dim = output_dim
        self.weights = None
        self.dweights = None
        self.inputs = None

    def update(self, learning_rate):
        self.weights = self.weights - learning_rate*self.dweights

    def fprop(self, inputs, pass_type='train'):
        if self.weights is None:
            self.weights = np.random.uniform(low=-0.01, high=0.01,
                                             size=(self.output_dim, inputs.shape[0] + 1))

        self.inputs = np.vstack((inputs, [-1]*inputs.shape[1]))
        result = np.dot(self.weights, self.inputs)
        return result

    def bprop(self, error):
        self.dweights = np.dot(error, np.transpose(self.inputs))
        error = np.dot(np.transpose(self.weights[:,:-1]), error)
        return error

class SigmoidLayer:
    def __init__(self):
        self.result = None

    def update(self, learning_rate):
        pass

    def fprop(self, inputs, pass_type='train'):
        if pass_type == 'train':
            self.result = sigmoid(inputs)
            return self.result
        return sigmoid(inputs)

    def bprop(self, error):
        aa = 1 - self.result
        bb = self.result * aa
        cc = error * bb
        return error *  self.result*(1 - self.result)

class DropoutLayer:
    def __init__(self, p):
        self.p = p
        self.mask = None

    def update(self, learning_rate):
        pass

    def fprop(self, inputs, pass_type='train'):
        if pass_type == 'train':
            count = inputs.shape[0]
            self.mask = np.transpose((np.random.random_sample((count)) > self.p).astype(int))
            res = np.multiply(inputs, self.mask[:, np.newaxis])
            return res
        return inputs * (1 - self.p)

    def bprop(self, error):
        return error * self.mask[:, np.newaxis]

class SoftmaxLayer:
    def fprop(self, inputs, pass_type='train'):
        weights = inputs[0]
        buf = np.exp(weights)
        result = buf / buf.sum(axis=0)
        if len(inputs) > 1:
            labels = inputs[1]
            error = result - labels
            return result, error
        return result

    def update(self, learning_rate):
        pass

    def bprop(self, error=None):
        pass

class NeuralNet:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)


    def fit(self, X_train, y_train, num_epochs, learning_rate,\
            batch_size, X_test=None, y_test=None):
        history = []
        for epoch_index in range(num_epochs):
            indexes = random.sample(range(X_train.shape[0]), X_train.shape[0])

            train_acc = []
            train_loss = []
            progress = 0
            for it in range(0, X_train.shape[0], batch_size):
                newprogress = it*100/X_train.shape[0]
                batch_indexes = sorted(indexes[it:min(it+batch_size,X_train.shape[0])])
                input = np.transpose(X_train[batch_indexes,:])

                labels = np.transpose(y_train[batch_indexes])
                labels_temp = np.argmax(labels, axis=0)

                # forward propagation
                for i in range(len(self.layers) - 1):
                    input = self.layers[i].fprop(input)
                result, error = self.layers[-1].fprop([input, labels])
                predictions = np.argmax(result, axis=0)
                train_acc.append(sum(labels_temp == predictions)*1.0/len(predictions))
                train_loss.append(-np.sum(labels*np.log(result))/batch_size)

                # back propagation
                for i in range(len(self.layers) - 2, -1, -1):
                    error = self.layers[i].bprop(error)

                # update weights
                for i in range(len(self.layers)):
                    self.layers[i].update(learning_rate)

            # # calculate test value
            if not X_test is None:
                labels_test = np.transpose(y_test)
                labels_test_temp = np.argmax(labels_test, axis=0)

                input = np.transpose(X_test)
                for i in range(len(self.layers) - 1):
                    input = self.layers[i].fprop(input, "test")
                result, error = self.layers[-1].fprop([input, labels_test], "test")
                predictions = np.argmax(result, axis=0)
                test_accuracy = sum(labels_test_temp == predictions)*1.0/len(predictions)

                print "epoch: {}, train_acc={}, train_loss={}, test_accuracy={}"\
                    .format(epoch_index, sum(train_acc)/len(train_acc),\
                            sum(train_loss)/len(train_loss), test_accuracy)
                history.append(test_accuracy)
            else:
                print "epoch: {}, train_acc={}, train_loss={}"\
                    .format(epoch_index, sum(train_acc)/len(train_acc),\
                            sum(train_loss)/len(train_loss))
        return history

    def predict(self, X_test):
        input = np.transpose(X_test)
        for i in range(len(self.layers) - 1):
            input = self.layers[i].fprop(input, "test")
        result = self.layers[-1].fprop([input], "test")
        return result