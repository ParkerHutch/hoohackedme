from markov import MarkovModel
from final_rnn import RNNModel
from sklearn.model_selection import train_test_split
from data_setup import get_data_sets
import argparse
import random
import numpy as np

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

    # set random seeds to ensure consistent results
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    markov_model = MarkovModel()
    rnn_model = RNNModel()
    if args.store_markov or args.store_rnn:
        X_train, X_test, X_val = get_data_sets()
        if args.store_markov:
            markov_model.train(X_train)
            markov_model.save_to_pickle(args.store_markov)
        if args.store_rnn:
            rnn_model.train(X_train)
            rnn_model.save_to_pickle(args.store_rnn)

    elif args.load_markov:
        markov_model.load_from_pickle(args.load_markov)
    elif args.load_rnn:
        rnn_model.load_from_pickle(args.load_rnn)
    else:
        # generate both models?
        pass
    

    output_file = open(args.output, 'w') if args.output else None
    try:
        use_markov = args.store_markov or args.load_markov
        use_rnn = args.store_rnn or args.load_rnn
        if args.guesses_count > 0:
            if use_markov:
                for _ in range(args.guesses_count):
                    output_line = markov_model.generate_password(4)
                    if output_file:
                        output_file.write(output_line + '\n')
                    else:
                        print(output_line)
            elif use_rnn:
                rnn_model.generate_passwords(args.guesses_count)
                # for password in rnn_model.generate_passwords(args.guesses_count):
                #     if output_file:
                #         output_file.write(password + '\n')
                #     else:
                #         # print(password)
                #         pass

    finally:
        if output_file:
            output_file.close()
    
if __name__ == '__main__':
    main()