import random

import numpy as np
import pygame


class Digits:
    AMOUNT_NEURONS = (28 * 28) + 10

    def __init__(self):
        self.weights = None
        pygame.init()
        self.screen = pygame.display.set_mode((600, 200))
        no_of_different_labels = 10  # i.e. 0, 1, 2, 3, ..., 9
        # self.train_data = np.loadtxt("mnist_train.csv", delimiter=",")
        self.test_data = np.loadtxt("mnist.csv", delimiter=",")
        test_labels = np.asfarray(self.test_data[:, :1])
        lr = np.arange(no_of_different_labels)
        # transform labels into one hot representation
        self.test_labels_one_hot = (lr == test_labels).astype(float)
        # we don't want zeroes and ones in the labels neither:
        self.test_labels_one_hot[self.test_labels_one_hot == 0] = 0.01
        self.test_labels_one_hot[self.test_labels_one_hot == 1] = 0.99
        self.in_vec = np.zeros(28 * 28 + 10)
        self.hid_vec = np.random.rand(28 * 28 + 10)
        self.out_vec = np.random.rand(28 * 28 + 10)

    def run_game(self, training):

        for pattern in range(100):
            print("Pattern: ", pattern)
            if not training:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return False

            # get next number
            for pixel in range(0, 784):
                self.in_vec[pixel] = self.test_data[pattern, pixel] / 255.0

            for pixel in range(0, 10):
                self.in_vec[pixel + 784] = self.test_labels_one_hot[pattern, pixel]

            # --- draw input activation ---
            self.activate(self.in_vec, self.hid_vec)
            self.activate(self.hid_vec, self.out_vec)

            if not training:
                self.draw_layer(1, self.in_vec)
                # --- draw hidden activation ---
                self.draw_layer(200, self.hid_vec)
                # --- draw output activation ---
                self.draw_layer(400, self.out_vec)
            #  insert activation here using in_vec[] as input vector

            self.learn()

            if not training:
                pygame.time.delay(1000)  # may be reduced in training time
                pygame.display.flip()
        return True

    def draw_layer(self, pos, layer_vec):
        jj = 0
        for y in range(0, 28):
            for x in range(0, 28):
                val = layer_vec[jj]
                jj = jj + 1
                pygame.draw.rect(self.screen, (0, 255 * val, 0), pygame.Rect(pos + x * 5, y * 5, 5, 5))
        for x in range(0, 10):
            val = layer_vec[jj]
            jj = jj + 1
            pygame.draw.rect(self.screen, (0, 255 * val, 0), pygame.Rect(pos + x * 5, 28 * 5, 5, 5))

    def activate(self, first, second):
        probabilities = first.dot(self.weights)
        for prob in range(len(probabilities)):
            probabilities[prob] = 1.0 if probabilities[prob] >= random.random() else 0.0
        second = probabilities

    def learn(self):
        n = 0.1
        for row in range(self.AMOUNT_NEURONS):
            for column in range(self.AMOUNT_NEURONS):
                self.weights[row][column] += n * (
                        (self.out_vec[row] * self.in_vec[column]) - (self.hid_vec[row] * self.out_vec[column]))

    def init_weights(self):
        self.weights = np.array([random.uniform(-0.1, 0.1)] * self.AMOUNT_NEURONS * self.AMOUNT_NEURONS).reshape(794,
                                                                                                                 794)

    def train_or_test(self):
        self.init_weights()

        if not self.run_game(True):
            return

        if not self.run_game(False):
            return


Digits().train_or_test()
