# Let's import TensorFlow library:
import tensorflow as tf

# Let's open and read the downloaded text file:
with open("rnn/temppasswords.txt") as f:
    text = f.read()

print(text[:100])

text = text[:1_000_000]
print("Length of text:", len(text))

print("".join(sorted(set(text.lower()))))

print("length of alphabet:", len("".join(sorted(set(text.lower())))))
#input()

text_vec_layer = tf.keras.layers.TextVectorization(split="character", standardize="lower")

text_vec_layer.adapt([text])
print("Type of shape of text vec layer:", type(text_vec_layer([text]).shape))
print("Shape of text vec layer:", text_vec_layer([text]).shape)
#input()

encoded = text_vec_layer([text])[0]
print("Encoded:", encoded)
#input()
encoded -= 2
n_tokens = text_vec_layer.vocabulary_size()-2
print("n tokens:", n_tokens)
#input()

dataset_size = len(encoded)
print("dataset size", dataset_size)
#input()


def to_dataset(sequence, length, shuffle=False, seed=None, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift=1,drop_remainder=True)
    ds = ds.flat_map(lambda window_ds: window_ds.batch(length + 1))
    if shuffle:
        ds = ds.shuffle(100_000, seed=seed)
    ds = ds.batch(batch_size)
    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)

print(list(to_dataset(text_vec_layer(["I like"])[0], length=5)))

length = 10
tf.random.set_seed(42)
# The training dataset:
# print("type of encoded train", type(encoded[:750_000]))
# print("encoded up to 1000000:", encoded[:750_000])
# print("type of encoded valid:", type(encoded[750_000:900_000]))
# print("encoded past 1000000", encoded[750_000:900_000])
# print(750000 == 750_000)

train_length = int(0.75 * len(text))
valid_length = int(0.15 * len(text))
test_length = int(0.10 * len(text))

train_set = to_dataset(encoded[:train_length], length=length, shuffle=True,seed=42)
# The validation dataset:
valid_set = to_dataset(encoded[train_length:train_length + valid_length], length=length)
# Test dataset:
test_set = to_dataset(encoded[train_length + valid_length:], length=length)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=n_tokens, output_dim=8),
    tf.keras.layers.GRU(64,return_sequences=True),
    tf.keras.layers.Dense(n_tokens,activation="softmax")
])
# Let's compile the model:
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="nadam", metrics=["accuracy"])

#  Let's train the model and save the best checkpoints:
model_ckpt = tf.keras.callbacks.ModelCheckpoint("passwords.keras", monitor="val_accuracy", save_best_only=True)



# Let's train the model:
history = model.fit( train_set, validation_data=valid_set, epochs=1,callbacks=[model_ckpt], steps_per_epoch=160)


password_model = tf.keras.Sequential([
    text_vec_layer,
    tf.keras.layers.Lambda(lambda X: X - 2),
    model
])

log_probas = tf.math.log([[0.6, 0.3, 0.1]])
tf.random.set_seed(42)
tf.random.categorical(log_probas, num_samples=10)

def next_char(text, temperature=1):
    y_proba = password_model.predict([text])[0, -1:]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1)[0, 0]
    return text_vec_layer.get_vocabulary()[char_id + 2]

def extend_text(text, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text

print(extend_text("i", temperature=0.01))

print(extend_text("i", temperature=1))

