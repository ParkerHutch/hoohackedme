from utils import Model
import random

class MarkovModel(Model):
    def __init__(self):
        self.stats = {}

    def train(self, train_set):
        X_train = train_set[::] # make a copy to modify locally
        self.stats = {}
        max_ngrams = 3
        for idx, line in enumerate(X_train):
            X_train[idx] += '\n' # to make the next part work

        # create a list of ngrams from a single line in
        # the training data
        def get_ngram(line, n):
            output = []
            for i, char in enumerate(line):
                # use backticks as start of line characters
                # e.g. test == "```t... ``te... `tes... test" for 4grams
                if i - n < 0:
                    buff = ''
                    for j in range(abs(i - n)):
                        buff += '`'
                    buff += line[0:i]
                    output.append((buff, char))
                else:
                    output.append((line[i - n:i], char))
            return output

        for line in X_train:
            # add ngrams to the stats dict for all n less than or
            # equal to max_ngrams (unigrams, bigrams, trigrams, etc...)
            # line = line + '\\n'
            for i in range(max_ngrams):
                for gram in get_ngram(line, i + 1):
                    prev = gram[0] # previous characters, ngram
                    nxt = gram[1] # next character
                    # if this ngram hasn't been seen yet
                    # add it to the stats dict
                    if not prev in self.stats:
                        self.stats[prev] = {}
                    # if the next character hasn't been seen to
                    # follow the ngram yet, add it the ngram's 
                    # dict of seen characters
                    if not nxt in self.stats[prev]:
                        self.stats[prev][nxt] = 0
                    # increment the statistic
                    self.stats[prev][nxt] += 1

        # convert frequency counts to probabilities
        for ngram in self.stats:
            
            chars = []
            occur = []
            probs = []

            for key, value in self.stats[ngram].items():
                chars.append(key)
                occur.append(value)

            total = sum(occur)
            probs = [float(x) / float(total) for x in occur]

            for key, value in self.stats[ngram].items():
                self.stats[ngram][key] = float(value) / float(total)
    
    def save_to_pickle(filename):
        pass
    
    def load_from_pickle(filename):
        pass

    def generate_passwords(count):
        pass

    def get_password_confidence(self, password):
        random.seed(0)
        probability = self.stats['`'][password[0]]
        for idx, first_letter in enumerate(password):
            next_letter_idx = idx + 1
            if next_letter_idx < len(password):
                probability *= self.stats[first_letter][password[next_letter_idx]]