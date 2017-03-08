import numpy as np
import pickle
import csv
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout


x1 = pickle.load(open('data/simple_test_inputs.pickle', 'rb'))
ids = pickle.load(open('data/listing_ids.pickle', 'rb'))
# y1 = pickle.load(open('data/simple_train_labels.pickle', 'rb'))

# Use New Handler to Save Data that has 28 fields not 4
input_size = 49
hidden_size = 128

model = Sequential()
model.add(Dense(output_dim=2 * hidden_size, input_dim=input_size, init='glorot_normal', activation='tanh'))
model.add(Dense(output_dim=hidden_size, input_dim=2 * hidden_size, init='glorot_normal', activation='sigmoid'))
# model.add(Dropout(0.5))
model.add(Dense(output_dim=hidden_size, input_dim=hidden_size, init='glorot_normal', activation='sigmoid'))
# model.add(Dropout(0.5))
model.add(Dense(output_dim=hidden_size, input_dim=hidden_size, init='glorot_normal', activation='sigmoid'))
# model.add(Dropout(0.5))
model.add(Dense(output_dim=hidden_size, input_dim=hidden_size, init='glorot_normal', activation='sigmoid'))
# model.add(Dropout(0.5))
model.add(Dense(output_dim=hidden_size, input_dim=hidden_size, init='glorot_normal', activation='sigmoid'))
# model.add(Dropout(0.5))
model.add(Dense(output_dim=3, input_dim=hidden_size, init='glorot_normal', W_regularizer='l1l2', activation='softmax'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', 'categorical_accuracy', 'fbeta_score'])

# train the model, iterating on the data in batches // VERBOSE=2 for printing metrics
# model.fit(x1, y1, validation_split=0.2, nb_epoch=1000, batch_size=512, verbose=2)

model.load_weights('sample_model.km')

result = model.predict(x1, verbose=2)

submission_file = []
submission_file.append("listing_id,high,medium,low")

for i in range(len(result)):
    current_id = ids[i]
    r = result[i]
    line = str(current_id) + ',' + ",".join(map(str, r))
    submission_file.append(line)

sub_file = open('submission.txt', 'w')
for item in submission_file:
    sub_file.write("%s\n" % item)

