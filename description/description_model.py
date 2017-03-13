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
should_train = False  # load existing model if false

interestResolver = dict()
interestResolver['medium'] = 1
interestResolver['low'] = 2
interestResolver['high'] = 0

if not should_train:
    print("evaluation mode")

print("reading training data")

f_train = open('../data/train.json', 'r')
x_train = json.load(f_train)
x_train = [sentence_to_word_array(v) for v in x_train['description'].values()]
x_train, words_dictionary = words_to_numbers(x_train, "words_dictionary.pickle")
y_train = pickle.load(open('../data/simple_train_labels_with_outliers.pickle', 'rb'))


print("reading test data")
f_test = open('../data/test.json', 'r')
x_test = json.load(f_test)
x_test = [sentence_to_word_array(v) for v in x_test['description'].values()]

if should_train:
    x_test = words_to_numbers_from_old_words_dict(x_test, words_dictionary)
else:
    x_test = words_to_numbers_from_old_words_dict(x_test, [], unk_integer=39000, file_to_read="words_dictionary.pickle")

max_features = 40000
maxlen = 1500
batch_size = 32
embedding_dims = 120
nb_filter = 128
filter_length = 6
hidden_dims = 1024
nb_epoch = 40

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
print('x_train shape:', x_train.shape)

x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_test shape:', x_test.shape)

model = Sequential()

model.add(Embedding(max_features, embedding_dims, input_length=maxlen, dropout=0.2))

model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='valid', activation='relu',
                        subsample_length=1))
model.add(GlobalMaxPooling1D())

model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

model.add(Dense(3))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_accuracy', 'fbeta_score'])

if should_train:
    for i in range(nb_epoch/2):
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=2, validation_split=0.1, verbose=1)
        classes = model.predict_classes(x_train)
        h = np.histogram(classes)
        print(h)

    model.save("description_model.km")
else:
    model.load_weights("description_model.km")

print("predicting test results")
results = model.predict(x_test)
predicted_classes = model.predict_classes(x_test)
print(np.histogram(predicted_classes))
pickle.dump(results, open("description_test_results.pickle", 'wb'))

print("predicting train results")
results = model.predict(x_train)
predicted_classes = model.predict_classes(x_train)
print(np.histogram(predicted_classes))
pickle.dump(results, open("description_train_results.pickle", 'wb'))
