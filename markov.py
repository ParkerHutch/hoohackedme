from utils import Model
import random
import pickle
from data_setup import get_data_sets

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
    
    def save_to_pickle(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.stats, file)

    def load_from_pickle(self, filename):
        with open(filename, 'rb') as file:
            self.stats = pickle.load(file)

    def generate_char(self, ngram):
            if ngram in self.stats:
                # sample from the probability distribution
                return random.choices(list(self.stats[ngram].keys()), weights=self.stats[ngram].values(), k=1)[0]
                # return np.random.choice(stats[ngram].keys(), p=stats[ngram].values())
            else:
                # print('{} not in stats dict'.format(ngram))
                return self.generate_char(ngram[0:-1])
    
    def generate_password(self, n):
        output = '`' * n
        for i in range(100):
            output += self.generate_char(output[i:i + n])
            if output[-1] == '\n':
                return output[0:-1].replace('`', '')[0:-1]
    
    def generate_passwords(self, count):
        max_ngrams = 3 # ngram size
        num_generate = count # number of passwords to generate

        # generate a single new password using a stats dict
        # created during the training phase 
        

        # Sample a character if the ngram appears in the stats dict.
        # Otherwise recursively decrement n to try smaller grams in
        # hopes to find a match (e.g. "off" becomes "of").
        # This is a deviation from a vanilla markov text generator
        # which one n-size. This generator uses all values <= n.
        # preferencing higher values of n first. 
        

        # with open('data/{}-gram.pickle'.format(max_ngrams)) as file:
        # 	stats = pickle.load(file)

        # start = time.time()

        res = []
        for i in range(num_generate):
            pw = self.generate_password(max_ngrams)
            if pw is not None:
                res.append(pw)
        return res

    def get_password_confidence(self, password):
        random.seed(0)
        probability = self.stats['`'][password[0]]
        for idx, first_letter in enumerate(password):
            next_letter_idx = idx + 1
            if next_letter_idx < len(password):
                probability *= self.stats[first_letter][password[next_letter_idx]]

def main():
    X_train, _, _ = get_data_sets()
    model = MarkovModel()
    model.train(X_train)
    model.save_to_pickle('markov.pickle')

if __name__ == '__main__':
    main()