import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def activate(layer, weights):
    next_layer = np.matmul(weights, layer)
    next_layer = sigmoid(next_layer)
    return next_layer


def format_input(image, label_pos):
    input_layer = image.flatten()
    labels = np.zeros(10)
    if label_pos is not None:
        labels[label_pos] = 1
    input_layer = np.append(input_layer, labels)
    input_layer = np.append(input_layer, 1)  # add bias

    return input_layer


class NumberPrediction:
    learnRate = 0.01

    def __init__(self):
        # load mnist data from tensorflow
        mnist = tf.keras.datasets.mnist
        (self.image_train, self.labels_train), (self.image_test, self.labels_test) = mnist.load_data()
        self.image_train, self.image_test = self.image_train / 255.0, self.image_test / 255.0
        # init weights
        weight_size = len(self.image_train[0].flatten()) + 10 + 1
        self.weights = np.random.rand(weight_size, weight_size)

    def learn(self, input_vec, hidden_vec, output_vec):
        learn_rate = 0.1
        self.weights += np.multiply.outer(hidden_vec, (input_vec - output_vec) * learn_rate)

    def training(self, episodes):
        for episode in range(episodes):
            # forward
            input_vec = format_input(self.image_train[episode], self.labels_train[episode])
            hidden_vec = activate(input_vec, self.weights)
            output_vec = activate(hidden_vec, np.matrix.transpose(self.weights))
            # training
            self.learn(input_vec, hidden_vec, output_vec)
            # visualize
            if episode % 1000 == 0:
                print("Learning pattern number: " + str(episode))

    def testing(self, episodes):
        for episode in range(episodes):
            # forward
            input_vec = format_input(self.image_train[episode], self.labels_train[episode])
            hidden_vec = activate(input_vec, self.weights)
            output_vec = activate(hidden_vec, np.matrix.transpose(self.weights))
            # visualize
            print("Pattern " + str(episode) + ":")
            self.plot_result(episode, hidden_vec, output_vec)
            time.sleep(1)

    def plot_result(self, episode, hidden_layer, output_layer):
        bars = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        test_img = self.image_train[episode]
        plt.subplot(2, 3, 1)
        plt.title("Input")
        plt.imshow(test_img, cmap='Greys')

        plt.subplot(2, 3, 2)
        plt.title("Hidden")
        plt.imshow(hidden_layer[0:-11].reshape(28, 28), cmap='Greys')

        plt.subplot(2, 3, 3)
        plt.title("Output")
        plt.imshow(output_layer[0:-11].reshape(28, 28), cmap='Greys')
        print('Predicted label:' + str(np.argmax(output_layer[-11: -1])))
        print('Real label:' + str(self.labels_train[episode]))

        plt.subplot(2, 2, 4)
        plt.title("Reconstructed label")
        plt.ylim(0.0, 1.0)
        y_pos = np.arange(0, 10)
        height = output_layer[-11: -1]
        plt.bar(y_pos, height)
        plt.xticks(y_pos, bars)

        plt.show()


rb = NumberPrediction()
rb.training(10000)
rb.testing(10000)
