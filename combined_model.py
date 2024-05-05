from utils import Model
import random
import pickle
from data_setup import get_data_sets
from final_rnn import RNNModel
from markov import MarkovModel

class CombinedModel(Model):
    def __init__(self, markov_proportion=0.3, markov_model = None, rnn_model = None):
        self.markov_model = markov_model if markov_model else MarkovModel() 
        self.rnn_model = rnn_model if rnn_model else RNNModel()
        self.markov_proportion = markov_proportion # the percentage of the passwords that should come from the markov model

    def train(self, train_set):
        raise NotImplementedError('Use load_from_pickle.')
    
    def save_to_pickle(self, filename):
        raise NotImplementedError('Combined model can\'t be saved yet. ')

    def load_from_pickle(self, markov_pickle_filename, rnn_pickle_filename):
        self.markov_model.load_from_pickle(markov_pickle_filename)
        self.rnn_model.load_from_pickle(rnn_pickle_filename)

    def generate_password(self, n):
        raise NotImplementedError('Not sure how this should be implemented yet.')

    def generate_passwords(self, count):
        markov_passwords_count = int(count * self.markov_proportion)
        for password in self.markov_model.generate_passwords(markov_passwords_count):
            print(password)
        self.rnn_model.generate_passwords(count - markov_passwords_count) # assuming that this will print its passwords as they are generated


def main():
    X_train, _, _ = get_data_sets()
    model = CombinedModel(0.5)
    model.load_from_pickle('markov.pickle', 'new_saved.pickle')
    model.generate_passwords(3)


if __name__ == '__main__':
    main()
