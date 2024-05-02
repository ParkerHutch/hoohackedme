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
        # train set is a list of each password: length = 10 million
        train_set = train_set[:1000]
        data = self.convert_to_string(train_set)
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)
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

        def lossFun(inputs, targets, hprev):
            """
            inputs,targets are both list of integers.
            hprev is Hx1 array of initial hidden state
            returns the loss, gradients on model parameters, and last hidden state
            """
            xs, hs, ys, ps = {}, {}, {}, {}
            hs[-1] = np.copy(hprev)
            loss = 0
            # forward pass
            for t in range(len(inputs)):
                xs[t] = np.zeros((vocab_size, 1))  # encode in 1-of-k representation
                xs[t][inputs[t]] = 1
                hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh)  # hidden state
                ys[t] = np.dot(Why, hs[t]) + by  # unnormalized log probabilities for next chars
                ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # probabilities for next chars
                loss += -np.log(ps[t][targets[t], 0])  # softmax (cross-entropy loss)
            # backward pass: compute gradients going backwards
            dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
            dbh, dby = np.zeros_like(bh), np.zeros_like(by)
            dhnext = np.zeros_like(hs[0])
            for t in reversed(range(len(inputs))):
                dy = np.copy(ps[t])
                dy[targets[
                    t]] -= 1  # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
                dWhy += np.dot(dy, hs[t].T)
                dby += dy
                dh = np.dot(Why.T, dy) + dhnext  # backprop into h
                dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
                dbh += dhraw
                dWxh += np.dot(dhraw, xs[t].T)
                dWhh += np.dot(dhraw, hs[t - 1].T)
                dhnext = np.dot(Whh.T, dhraw)
            for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
                np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
            return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]

        def sample(h, seed_ix, n):
            """
            sample a sequence of integers from the model
            h is memory state, seed_ix is seed letter for first time step
            """
            x = np.zeros((vocab_size, 1))
            x[seed_ix] = 1
            ixes = []
            for t in range(n):
                h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
                y = np.dot(Why, h) + by
                p = np.exp(y) / np.sum(np.exp(y))
                ix = np.random.choice(range(vocab_size), p=p.ravel())
                x = np.zeros((vocab_size, 1))
                x[ix] = 1
                ixes.append(ix)
            return ixes

        def produce_bunch(hprev, p, data, seq_length):
            inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
            sample_ix = sample(hprev, inputs[0], 200)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print('%s' % (txt,))

        n, p = 0, 0
        mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
        mbh, mby = np.zeros_like(bh), np.zeros_like(by)  # memory variables for Adagrad
        smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss iteration 0
        while True:
            # prepare inputs (we're sweeping from left to right in steps seq_length long)
            if p + seq_length + 1 >= len(data) or n == 0:
                hprev = np.zeros((hidden_size, 1))  # reset RNN memory
                p = 0  # go from start of data
            inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
            targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]

            # sample from the model now and then
            # if n % 100 == 0:
            #     produce_bunch(hprev, p, data, seq_length)

            # forward seq_length characters through the net and fetch gradient
            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
            # print("loss", loss)
            # input()
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            if n % 100 == 0:
                print('iter %d, loss: %f' % (n, smooth_loss))  # print progress

            # the important code
            if smooth_loss < 50.00:
                saved = {"hprev": hprev,
                         "p": p,
                         "Wxh": Wxh,
                         "Whh": Whh,
                         "Why": Why,
                         "vocab_size": vocab_size,
                         "bh": bh,
                         "by": by,
                         'chars': chars,
                         'data': data}
                self.data['data'] = saved
                self.save_to_pickle("saved.pickle")
                exit()

            # perform parameter update with Adagrad
            for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                          [dWxh, dWhh, dWhy, dbh, dby],
                                          [mWxh, mWhh, mWhy, mbh, mby]):
                mem += dparam * dparam
                param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

            p += seq_length  # move data pointer
            n += 1  # iteration counter

    def save_to_pickle(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.data, f)

    def load_from_pickle(self, filename):
        with open("saved.pickle", 'rb') as f:
            self.data = pickle.load(f)
        return self.data

    def generate_password(self, n):
        return self.enerate_passwords(1)

    def generate_passwords(self, count):

        data = self.load_from_pickle("saved.pickle")

        chars = data['data']['chars']
        Wxh = data['data']['Wxh']
        Whh = data['data']['Whh']
        Why = data['data']['Why']
        bh = data['data']['bh']
        by = data['data']['by']
        d = data['data']['data']
        vocab_size = data['data']['vocab_size']
        hprev = data['data']['hprev']
        p = data['data']['p']


        char_to_ix = {ch: i for i, ch in enumerate(chars)}
        ix_to_char = {i: ch for i, ch in enumerate(chars)}


        def sample(h, seed_ix, n):
            """
            sample a sequence of integers from the model
            h is memory state, seed_ix is seed letter for first time step
            """
            x = np.zeros((vocab_size, 1))
            x[seed_ix] = 1
            ixes = []
            for t in range(n):
                h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
                y = np.dot(Why, h) + by
                p = np.exp(y) / np.sum(np.exp(y))
                ix = np.random.choice(range(vocab_size), p=p.ravel())
                x = np.zeros((vocab_size, 1))
                x[ix] = 1
                ixes.append(ix)
            return ixes

        def produce_bunch(hprev, p, data, seq_length):
            inputs = [char_to_ix[ch] for ch in d[p:p + seq_length]]
            sample_ix = sample(hprev, inputs[0], 200)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            return '%s' % (txt,)

        res = []
        passwords_generated = 0
        while passwords_generated < count:
            for password in produce_bunch(hprev, p, d, seq_length=100).split('\n'):
                passwords_generated += 1
                res.append(password)
        # for i in range(count):
        #     produce_bunch(hprev, p, d, seq_length=100)
        return res

    def convert_to_string(self, train_set):
        text = ""
        for line in train_set:
            text = text + line + "\n"
        return text



def main():
    X_train, _, _ = get_data_sets()
    model = RNNModel()
    # model.train(X_train)
    model.load_from_pickle('saved.pickle')
    print(model.generate_passwords(100))


if __name__ == '__main__':
    main()
