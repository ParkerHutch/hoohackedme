import tensorflow as tf
import numpy as np

new_model = tf.keras.models.load_model('passwords.keras')
new_model.summary()
print(new_model.predict())


