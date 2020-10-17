import numpy as np
import pickle
import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation
import os

FILE_PATH = 'data/wonderland.txt'
BASENAME = os.path.join(FILE_PATH)

char2int = pickle.load(open(f'{BASENAME}-char2int.pickle', 'rb'))
int2char = pickle.load(open(f'{BASENAME}-int2char.pickle', 'rb'))

FILE_PATH = 'wonderland.txt'
BASENAME = os.path.join(FILE_PATH)

sequence_length = 100
vocab_size = len(char2int)

# building the model
model = Sequential([
    LSTM(256, input_shape=(sequence_length, vocab_size), return_sequences=True),
    Dropout(0.3),
    LSTM(256),
    Dense(vocab_size, activation='softmax'),
])

model.load_weights(f'results/{BASENAME}-{sequence_length}.h5')
seed = 'chapter xiii:'
s = seed
n_chars = 400
generated = ''
for i in tqdm.tqdm(range(n_chars), 'Generating text'):
    X = np.zeros((1, sequence_length, vocab_size))
    for t, char in enumerate(seed):
        X[0, (sequence_length - len(seed)) + t, char2int[char]] = 1
    predicted = model.predict(X, verbose=0)[0]
    next_index = np.argmax(predicted)
    next_char = int2char[next_index]
    generated += next_char
    seed = seed[1:] + next_char

print(seed, generated)
open('results/results2.txt', 'w', encoding='utf-8').write(generated)
