# Initializations
import os
from random import shuffle
import tensorflow as tf
import numpy as np

train_review_dir = './Datasets/aclImdb/train'
embedding_dims = 200
maxwords = 15000
glove_file_loc = './Datasets'
maxlen = 200
lstm_neurons = 50
epochs = 2
batch_size = 32

# Process Reviews and Labels
def preprocess_reviews(review_dir):
    reviews_collection = []
    for review_type in ['neg', 'pos']:
        review_folder = os.path.join(review_dir, review_type)
        for review_file in os.listdir(review_folder):
            if review_file[-4 :] == '.txt':
                _file = open(os.path.join(review_folder, review_file), encoding = "utf8")
            if review_type == 'neg':
                reviews_collection.append((0, _file.read())) 
                _file.close()
            else:
                reviews_collection.append((1, _file.read())) 
                _file.close()
    shuffle(reviews_collection)
    return reviews_collection

# Load Reviews and Labels
def get_reviews_and_labels(reviews_and_labels):
    reviews_text = []
    reviews_labels = []
    for review in reviews_and_labels:
        reviews_text.append(review[1])
        reviews_labels.append(review[0])
    labels_as_array = np.asarray(reviews_labels)
    return reviews_text, labels_as_array

# Get Padded Sequences and Tokenizer
def get_sequences_and_tokenizer(reviews_text):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = maxwords)
    tokenizer.fit_on_texts(reviews_text)
    sequences = tokenizer.texts_to_sequences(reviews_text)
    padded_review_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen = maxlen)
    return padded_review_sequences, tokenizer

# Get gloVe Embedding Matrix
def get_glove_embedding_matrix(indexes_of_words, maxwords, embedding_dims):
    embeddings_index = {}
    embedding_matrix = np.zeros((maxwords, embedding_dims))
    
    glove_file = open(os.path.join(glove_file_loc, 'glove.6B.200d.txt'), encoding = "utf8")
    for line in glove_file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype = 'float32')
        embeddings_index[word] = coefs
    glove_file.close()
    
    for word, idx in indexes_of_words.items():
        if idx < maxwords:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[idx] = embedding_vector
    return embedding_matrix
 
 # Build LSTM Model with Pretrained GloVe Embedding
 def make_glove_model(embedding_matrix):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim = maxwords, output_dim = embedding_dims, input_length = maxlen))
    model.add(tf.keras.layers.LSTM(lstm_neurons, 
                               activation = 'tanh',
                               recurrent_activation = 'sigmoid',
                               use_bias = True,
                               dropout = 0.2,
                               recurrent_dropout = 0.0,
                               return_sequences = True,
                               unroll = False,
                               input_shape = (maxlen, embedding_dims)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation = 'relu'))
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
    model.save('lstm_sentiment_analysis.h5')
    model.layers[0].set_weights([embedding_matrix])  
    model.layers[0].trainable = False
    model.compile(optimizer = tf.keras.optimizers.RMSprop(),
             loss = tf.keras.losses.binary_crossentropy,
             metrics = [tf.keras.metrics.binary_accuracy])
    return model
 
 # Load and process data
 reviews_and_labels = preprocess_reviews(train_review_dir)
reviews_text, reviews_labels = get_reviews_and_labels(reviews_and_labels)
padded_review_sequences, tokenizer = get_sequences_and_tokenizer(reviews_text)
data_split_point = int(len(padded_review_sequences) * 0.8) 
x_train = padded_review_sequences[:data_split_point] 
y_train = reviews_labels[:data_split_point] 
x_test = padded_review_sequences[data_split_point:] 
y_test = reviews_labels[data_split_point:] 
indexes_of_words = tokenizer.word_index
glove_embedding_matrix = get_glove_embedding_matrix(indexes_of_words, maxwords, embedding_dims)

# Train Model
model = make_glove_model(glove_embedding_matrix)
history = model.fit(x_train,
                    y_train,
                    batch_size = batch_size,
                    epochs = epochs,
                    validation_data = (x_test, y_test))

model_json = model.to_json()
with open("lstm_sentiment_analysis.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("lstm_sentiment_analysis.h5")
