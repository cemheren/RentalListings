import numpy as np
import pickle

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout


x1 = pickle.load(open('data/simple_train_inputs.pickle', 'rb'))
y1 = pickle.load(open('data/simple_train_labels.pickle', 'rb'))

# Use New Handler to Save Data that has 28 fields not 4
input_size = 79
hidden_size = 1024

model = Sequential()
model.add(Dense(output_dim=hidden_size, input_dim=input_size, init='glorot_normal', activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(output_dim=hidden_size, input_dim=hidden_size, init='glorot_normal', activation='sigmoid'))
model.add(Dropout(0.3))

model.add(Dense(output_dim=3, input_dim=hidden_size, init='glorot_normal', W_regularizer='l1l2', activation='softmax'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', 'categorical_accuracy', 'fbeta_score'])

# train the model, iterating on the data in batches // VERBOSE=2 for printing metrics
model.fit(x1, y1, validation_split=0.01, nb_epoch=256, batch_size=512, verbose=2)

model.save('sample_model_79.km')

print("-- TESTING...")
for i in range(30):
    print("-")
    q = model.predict(np.reshape(x1[i], (1, input_size)))[0]
    q = np.argmax(q, axis=0)
    print("prediction = ", q)
    print("y = ", np.argmax(y1[i], axis=0))
    print("x = ", x1[i])
    print("-")
