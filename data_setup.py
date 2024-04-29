
from sklearn.model_selection import train_test_split

def get_data_sets():
    lines = []
    # need errors='ignore' to get rid of some utf-8 errors when decoding
    with open('rockyou.txt', 'r', encoding='utf-8', errors='ignore') as data_file:
        for line in data_file:
            lines.append(line.rstrip())

    test_proportion = 0.2
    validation_proportion = 0.1
    test_set_count = (int) (test_proportion * len(lines))
    validation_set_count = (int) (validation_proportion * len(lines))


    X_train, X_test = train_test_split(lines, test_size=test_set_count, random_state=1)
    X_train, X_val = train_test_split(X_train, test_size=validation_set_count, random_state=1)
    return X_train, X_test, X_val