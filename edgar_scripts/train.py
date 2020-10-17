import tensorflow as tf
import numpy as np
import os
import pickle
import requests
import random
import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from string import punctuation

sequence_length = 100
BATCH_SIZE = 128
EPOCHS = 3
FILE_PATH = 'train_data/new.txt'
BASENAME = os.path.basename(FILE_PATH)

# read the train_data
text = open(FILE_PATH, encoding='utf-8').read()
text = text.split('\n')
text.pop()

n_words = len(text)
vocab = ''.join(sorted(set(text)))
unique_words = len(set(text))

#tokenize stuff
tokenizer = Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(text)
vocab_size = len(tokenizer.word_index) + 1
word_2_index = tokenizer.word_index
index_2_word = dict(map(reversed, word_2_index.items()))

pickle.dump(word_2_index, open(f'train_data/{BASENAME}-word2index.pickle', 'wb'))
pickle.dump(index_2_word, open(f'train_data/{BASENAME}-index2word.pickle', 'wb'))

input_sequence = []
output_words = []
input_seq_length = 100

for i in range(0, n_words - input_seq_length , 1):
    in_seq = text[i:i + input_seq_length]
    out_seq = text[i + input_seq_length]
    input_sequence.append([word_2_index[word] for word in in_seq])
    output_words.append(word_2_index[out_seq])

X = np.reshape(input_sequence, (len(input_sequence), input_seq_length, 1))
X = X / float(vocab_size)

y = to_categorical(output_words)

model = Sequential([
    LSTM(800, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
    Dropout(0.3),
    LSTM(800),
    Dense(y.shape[1], activation='softmax'),
])

model.load_weights(f'results/{BASENAME}-{sequence_length}.h5')

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

if not os.path.isdir('results'):
    os.mkdir('results')

model.fit(X, y, batch_size=256, epochs=EPOCHS, verbose=1)

model.save(f'results/{BASENAME}-{sequence_length}.h5')


word_2_index = pickle.load(open(f'train_data/{BASENAME}-word2index.pickle', 'rb'))
index_2_word = pickle.load(open(f'train_data/{BASENAME}-index2word.pickle', 'rb'))

model.load_weights(f'results/{BASENAME}-{sequence_length}.h5')

random_seq_index = np.random.randint(0, len(input_sequence)-1)
random_seq = input_sequence[random_seq_index]

word_sequence = [index_2_word[value] for value in random_seq]

for i in tqdm.tqdm(range(100), 'Generating text'):
    int_sample = np.reshape(random_seq, (1, len(random_seq), 1))
    int_sample = int_sample / float(vocab_size)

    predicted_word_index = model.predict(int_sample, verbose=0)

    predicted_word_id = np.argmax(predicted_word_index)
    seq_in = [index_2_word[index] for index in random_seq]

    word_sequence.append(index_2_word[ predicted_word_id])

    random_seq.append(predicted_word_id)
    random_seq = random_seq[1:len(random_seq)]

final_output = ''
for word in word_sequence:
    final_output = final_output + ' ' + word

print(final_output)
open('results/results.txt', 'w', encoding='utf-8').write(final_output)
