from __future__ import print_function
import ijson
import json
import numpy as np

from WordsToNumbers import *

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D


np.random.seed(1337)  # for reproducibility

interestResolver = dict()
interestResolver['medium'] = 1
interestResolver['low'] = 2
interestResolver['high'] = 0

f = open('../data/train.json', 'r')
x = json.load(f)

x_train = [sentence_to_word_array(v) for v in x['description'].values()]
x_train, words_dictionary = words_to_numbers(x_train, "words_dictionary.pickle")

y_train = pickle.load(open('../data/simple_train_labels.pickle', 'rb'))

max_features = 40000
maxlen = 1000
batch_size = 32
embedding_dims = 100
nb_filter = 250
filter_length = 3
hidden_dims = 500
nb_epoch = 10

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
print('x_train shape:', x_train.shape)

model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features, embedding_dims, input_length=maxlen, dropout=0.2))

# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='valid', activation='relu', subsample_length=1))

# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(3))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_accuracy', 'fbeta_score'])

model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.2, verbose=1)

classes = model.predict_classes(x_train)
h = np.histogram(classes)
print(h)

model.save("description_model.km")
