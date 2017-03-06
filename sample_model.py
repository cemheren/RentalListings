import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import pickle

x1 = pickle.load(open('data/simple_train_inputs.pickle', 'rb'))
y1 = pickle.load(open('data/simple_train_labels.pickle', 'rb'))

model = Sequential()
model.add(Dense(output_dim=10, input_dim=3, activation='sigmoid'))
model.add(Dense(output_dim=3, input_dim=10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# generate dummy data

# train the model, iterating on the data in batches
# of 32 samples
model.fit(x1, y1, nb_epoch=10, batch_size=32)



