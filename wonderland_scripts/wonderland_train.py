import tensorflow as tf
import numpy as np
import os
import pickle
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from string import punctuation

sequence_length = 100
BATCH_SIZE = 256
EPOCHS = 5
FILE_PATH = 'data/wonderland.txt'
BASENAME = os.path.basename(FILE_PATH)

# content = requests.get('http://www.gutenberg.org/cache/epub/11/pg11.txt').text
# open('data/wonderland.txt', 'w', encoding='utf-8').write(content)

# read the data
text = open(FILE_PATH, encoding='utf-8').read()
text = text.lower()
text = text.translate(str.maketrans('', '', punctuation))
n_chars = len(text)
vocab = ''.join(sorted(set(text)))
n_unique_chars = len(vocab)
print('unique_chars:', vocab)
print('Number of characters:', n_chars)
print('Number of unique characters:', n_unique_chars)

char2int = {c: i for i, c in enumerate(vocab)}
int2char = {i: c for i, c in enumerate(vocab)}

pickle.dump(char2int, open(f'data/{BASENAME}-char2int.pickle', 'wb'))
pickle.dump(int2char, open(f'data/{BASENAME}-int2char.pickle', 'wb'))

encoded_text = np.array([char2int[c] for c in text])

char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)

sequences = char_dataset.batch(2*sequence_length + 1, drop_remainder=True)

def split_sample(sample):
    ds = tf.data.Dataset.from_tensors((sample[:sequence_length], sample[sequence_length]))
    for i in range(1, (sample.shape[0]-1) // 2):
        input_ = sample[i: i+sequence_length]
        target = sample[i+sequence_length]
        other_ds = tf.data.Dataset.from_tensors((input_, target))
        ds = ds.concatenate(other_ds)
    return ds

dataset = sequences.flat_map(split_sample)

def one_hot_samples(input_, target):
    return tf.one_hot(input_, n_unique_chars), tf.one_hot(target, n_unique_chars)

dataset = dataset.map(one_hot_samples)

ds = dataset.repeat().shuffle(1024).batch(BATCH_SIZE, drop_remainder=True)

model = Sequential([
    LSTM(256, input_shape=(sequence_length, n_unique_chars), return_sequences=True),
    Dropout(0.3),
    LSTM(256),
    Dense(n_unique_chars, activation='softmax'),
])

model.load_weights(f'results/{BASENAME}-{sequence_length}.h5')

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

if not os.path.isdir('results'):
    os.mkdir('results')

model.fit(ds, steps_per_epoch=(encoded_text.shape[0] - sequence_length) // BATCH_SIZE, epochs=EPOCHS)

model.save(f'results/{BASENAME}-{sequence_length}.h5')
