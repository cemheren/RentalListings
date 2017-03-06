import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import numpy as np
import pickle

x1 = pickle.load(open('data/simple_train_inputs.pickle', 'rb'))
y1 = pickle.load(open('data/simple_train_labels.pickle', 'rb'))

input_size = 4
hidden_size = 128

model = Sequential()
model.add(Dense(output_dim=hidden_size, input_dim=input_size, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(output_dim=hidden_size, input_dim=hidden_size, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(output_dim=3, input_dim=hidden_size, activation='softmax'))

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# train the model, iterating on the data in batches
# of 32 samples
model.fit(x1, y1, validation_split=0.2, nb_epoch=100, batch_size=64, class_weight={0: 1.0, 1: 3.0, 2: 9.0})

print("-- TESTING...")
for i in range(30):
    print("-")
    q = model.predict(np.reshape(x1[i], (1, input_size)))[0]
    q = np.argmax(q, axis=0)
    print("prediction = ", q)
    print("y = ", np.argmax(y1[i], axis=0))
    print("x = ", x1[i])
    print("-")
