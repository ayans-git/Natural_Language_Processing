import tensorflow as tf
import numpy as np
import nltk
from nltk.corpus import gutenberg
import random
import sys

#nltk.download('gutenberg') #if not downloaded already
#print(gutenberg.fileids()) #verify


lstm_neurons = 128
maxlen = 40
step = 3
sentences = []
next_chars = []
batch_size = 128
epochs = 1


def get_text_chars(file):
    _text = ''
    for txt in file:
        if 'shakespeare' in txt:
            _text += gutenberg.raw(txt).lower()
    _chars = sorted(list(set(_text)))
    return _chars, _text


chars, text = get_text_chars(gutenberg.fileids())

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])
char_indices = dict((char, chars.index(char)) for char in chars)

# One-hot encoding of training examples
# Encodes the characters into binary arrays
X = np.zeros((len(sentences), maxlen, len(chars)), dtype = np.bool)
y = np.zeros((len(sentences), len(chars)), dtype = np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
    

def sample(preds, temperature = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def make_char_lstm_model():
    inp = tf.keras.layers.Input(shape = (maxlen, len(chars)))
    # Use GRU to have less number of trainable parameters
    x = tf.keras.layers.LSTM(lstm_neurons)(inp)
    output = tf.keras.layers.Dense(len(chars), activation = 'softmax')(x)
    model = tf.keras.models.Model(inp, output, name = 'textgen_char_lstm_model')
    model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = 0.01),
                  loss = tf.keras.losses.categorical_crossentropy)
    return model
    
    
model = make_char_lstm_model()
for epoch in range(1, 60):
    model.fit(X, y, batch_size = batch_size, epochs = epochs)
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_sentence = text[start_index : start_index + maxlen]
    print('\n\n** Generating with seed: " ' + generated_sentence + '"')
    for diversity in [0.2, 0.5, 0.8, 1.0, 1.2]:
        print('\n****** diversity: ', diversity)
        sys.stdout.write(generated_sentence)
        for i in range(400):
            encoded_sample = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_sentence):
                encoded_sample[0, t, char_indices[char]] = 1.
            preds = model.predict(encoded_sample, verbose = 0)[0]
            next_index = sample(preds, diversity)
            next_char = chars[next_index]
            generated_sentence += next_char
            generated_sentence = generated_sentence[1:]
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
