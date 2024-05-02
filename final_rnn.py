from abc import ABC

from utils import Model
import random
import pickle
from data_setup import get_data_sets
import numpy as np
import sys


class RNNModel(Model):
    def __init__(self):
        self.data = {}

    def train(self, train_set):
        print(type(train_set))
        return
        chars = list(set(train_set))
        data_size, vocab_size = len(train_set), len(chars)
        # print('data has %d characters, %d unique.' % (data_size, vocab_size))
        char_to_ix = {ch: i for i, ch in enumerate(chars)}
        ix_to_char = {i: ch for i, ch in enumerate(chars)}

        # hyperparameters
        hidden_size = 100  # size of hidden layer of neurons
        seq_length = 25  # number of steps to unroll the RNN for
        learning_rate = 1e-1

        # model parameters
        Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # input to hidden
        Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
        Why = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden to output
        bh = np.zeros((hidden_size, 1))  # hidden bias
        by = np.zeros((vocab_size, 1))  # output bias

    def save_to_pickle(self, filename):
        with open("saved.pickle", 'wb') as f:
            pickle.dump(self.data, f)

    def load_from_pickle(self, filename):
        with open('saved.pickle', 'rb') as f:
            data = pickle.load(filename)
            return data

    def generate_passwords(self, count):
        pass


def main():
    X_train, _, _ = get_data_sets()
    model = RNNModel()
    model.train(X_train)
    model.save_to_pickle('markov.pickle')


if __name__ == '__main__':
    main()
