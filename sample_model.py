import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import numpy as np
import pickle

x1 = pickle.load(open('data/simple_train_inputs.pickle', 'rb'))
y1 = pickle.load(open('data/simple_train_labels.pickle', 'rb'))

model = Sequential()
model.add(Dense(output_dim=10, input_dim=3, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(output_dim=3, input_dim=10, activation='softmax'))

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# train the model, iterating on the data in batches
# of 32 samples
model.fit(x1, y1, validation_split=0.2, nb_epoch=1, batch_size=64)
