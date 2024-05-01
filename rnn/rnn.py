import sklearn.model_selection
import tensorflow as tf
import numpy as np
import random
import string


def get_chars(text: str):
    return sorted(list(set(text)))

def preprocess(text: str):
    # using a smaller text length
    # text = text[:1000000]
    text = text.replace("\n", "")
    text = text[:5000]

    chars = sorted(list(set(text)))
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}

    # Set the maximum sequence length (max_len) to be the length of the longest sequence
    max_len = max([len(s) for s in text])

    # Create training examples and labels
    X = []
    y = []

    for i in range(0, len(text) - max_len, 1):
        X.append([char_to_int[ch] for ch in text[i:i + max_len]])
        y.append(char_to_int[text[i + max_len]])

    # Set the maximum sequence length (max_len) to be the length of the longest sequence
    max_len = max([len(s) for s in text])

    # Create training examples and labels
    X = []
    y = []

    for i in range(0, len(text) - max_len, 1):
        X.append([char_to_int[ch] for ch in text[i:i + max_len]])
        y.append(char_to_int[text[i + max_len]])

    return X, y


def create_model(X_train, X_test, y_train, y_test, chars):
    # Define the model architecture
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=len(chars), output_dim=8))
    model.add(tf.keras.layers.GRU(64, return_sequences=True))
    model.add(tf.keras.layers.SimpleRNN(128))
    model.add(tf.keras.layers.Dense(units=len(chars), activation='softmax'))
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="nadam", metrics=["accuracy"])

    # Train the model
    validation_data = (X_test, y_test)
    model.fit(X_train, y_train, epochs=20, batch_size=64)
    model.summary()
    return model


# def generate_text(seed, num_chars):
#     # Initialize the generated text
#     generated_text = seed
#
#     # Encode the seed as integers
#     encoded_seed = [char_to_int[ch] for ch in seed]
#
#     # Pad the seed
#     padded_seed = tf.keras.preprocessing.sequence.pad_sequences([encoded_seed], maxlen=max_len, padding='post')
#
#     # Generate characters
#     for i in range(num_chars):
#         # Get the next character probabilities
#         probs = model.predict(padded_seed)[0]
#
#         # Get the index of the character with the highest probability
#         index = np.argmax(probs)
#
#         # Add the character to the generated text
#         generated_text += int_to_char[index]
#
#         # Update the padded seed with the latest character
#         padded_seed = np.append(padded_seed[0][1:], index)
#         padded_seed = tf.keras.preprocessing.sequence.pad_sequences([padded_seed], maxlen=max_len, padding='post')
#
#     return generated_text


def next_char(t, chars, temperature=1, ):
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}
    encoded_seed = [char_to_int[ch] for ch in t]
    padded_seed = tf.keras.preprocessing.sequence.pad_sequences([encoded_seed], maxlen=max_len, padding='post')
    y_proba = model.predict(padded_seed)[0, -1:]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1)[0, 0]
    return chars[char_id]


def extend_text(text, chars, n_chars=10, temperature=1, ):
    for _ in range(n_chars):
        text += next_char(text, temperature, chars)
    return text


# Generate text

if __name__ == "__main__":
    with open('rnn/temppasswords.txt', 'r') as f:
        text = f.read()
        f.close()
    # step 1: preprocess the text
    X, y = preprocess(text)
    chars = get_chars(text)
    print("char length:", len(chars))
    max_len = max([len(s) for s in text])
    # step 2: get X_train, X_valid, y_train, y_valid
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.20, random_state=1000)

    # Pad the examples
    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len, padding='post')
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_len, padding='post')
    # Convert labels to categorical format

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    # step 3: create and fit the model
    model = create_model(X_train, X_test, y_train, y_test, chars)

    # step 4: generate passwords
    passwords_generated = 0
    passwords = []
    while passwords_generated < 10:
        starting_character = ''.join(
            random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(1))
        if starting_character in chars:
            passwords.append(extend_text(starting_character, chars=chars))
            passwords_generated += 1

    for p in passwords:
        print(p)
