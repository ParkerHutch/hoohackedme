from markov import MarkovModel
from sklearn.model_selection import train_test_split
from data_setup import get_data_sets
import argparse
import random

def main():
    
    parser = argparse.ArgumentParser(description='Password Generation Tool for NetSec class.')
    parser.add_argument('--store-markov', help='generate a Markov model and store its pickle in the filename given')
    parser.add_argument('--load-markov', help='load a Markov model from the pickle in the filename given')
    parser.add_argument('--store-rnn', help='generate an RNN model and store its pickle in the filename given')
    parser.add_argument('--load-rnn', help='load an RNN model from the pickle in the filename given')
    parser.add_argument('--guesses_count', type=int, default=0, help='The number of passwords to generate')
    parser.add_argument('-o', '--output', type=str, help='The file to write password guesses to')
    parser.add_argument('-r', '--random_seed', type=int, default=0, help='The random seed')
    args = parser.parse_args()

    random.seed(args.random_seed)

    markov_model = MarkovModel()
    if args.store_markov or args.store_rnn:
        X_train, X_test, X_val = get_data_sets()
        if args.store_markov:
            markov_model.train(X_train)
            markov_model.save_to_pickle(args.store_markov)
    elif args.load_markov:
        markov_model.load_from_pickle(args.load_markov)
    elif args.load_rnn:
        pass
    else:
        # generate both models?
        pass
    

    output_file = open(args.output, 'w') if args.output else None
    try:
        use_markov = args.store_markov or args.load_markov
        if args.guesses_count > 0:
            if use_markov:
                for _ in range(args.guesses_count):
                    output_line = markov_model.generate_password(4)
                    if output_file:
                        output_file.write(output_line + '\n')
                    else:
                        print(output_line)
    finally:
        if output_file:
            output_file.close()
    
if __name__ == '__main__':
    main()