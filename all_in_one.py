import string
import random
import datetime

import numpy as np
import pickle
import csv
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

import sys
sys.path.insert(0, './data')
import jsonDataHandler as jDH
import RL_preprocessor as RL_prep

# Load Training & Test Data
x1_train = pickle.load(open('data/simple_train_inputs.pickle', 'rb'))
y1_train = pickle.load(open('data/simple_train_labels.pickle', 'rb'))

x1_test = pickle.load(open('data/simple_test_inputs.pickle', 'rb'))
ids_test = pickle.load(open('data/listing_ids.pickle', 'rb'))


# Prepare and Process Data
num_features_to_extract = 100
# If you want you may run data handling by de-commenting the next line
jDH.handle_data_and_picle_it(num_features_to_extract)
normalizableColumnResolver = jDH.get_normalizable_column_resolver()

##################################################
# Normalize Train and Test Together
##################################################
# Get Normalizing Parameters from Training Data
(x1_train, x1_test) = RL_prep.normalize_cols(x1_train, x1_test, normalizableColumnResolver, ['price', 'num_images', 'num_desc_words'])


random_file_postfix = ''.join(random.choice(string.lowercase) for x in range(10))
model_name = 'mi0129_' + datetime.datetime.now().strftime("t%H_%M_")
##################################################
# Train Model on Train Data
##################################################
input_size = 29 + num_features_to_extract
hidden_size = 1024

model = Sequential()
model.add(Dense(output_dim=hidden_size, input_dim=input_size, init='glorot_normal', activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(output_dim=hidden_size, input_dim=hidden_size, init='glorot_normal', activation='tanh'))
model.add(Dropout(0.3))

model.add(Dense(output_dim=3, input_dim=hidden_size, init='glorot_normal', W_regularizer='l1l2', activation='softmax'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', 'categorical_accuracy', 'fbeta_score'])

# train the model, iterating on the data in batches // VERBOSE=2 for printing metrics
model.fit(x1_train, y1_train, validation_split=0.05, nb_epoch=256, batch_size=512, verbose=2)

# Save This Model
model.save('sample_model_' + model_name + random_file_postfix + '.km')


predicted_classes_on_training_data = model.predict_classes(x1_train)
print 'Trained Network Prediction Histogram on Training Data:', np.histogram(predicted_classes_on_training_data)


##################################################
# Run This Model on Test Data
##################################################
result = model.predict(x1_test, verbose=2)


submission_file = []
submission_file.append("listing_id,high,medium,low")

for i in range(len(result)):
    current_id = ids_test[i]
    r = result[i]
    line = str(current_id) + ',' + ",".join(map(str, r))
    submission_file.append(line)

# Save Output of Testing
sub_file = open('submission_' + model_name + random_file_postfix + '.txt', 'w')
for item in submission_file:
    sub_file.write("%s\n" % item)

print '\nDone'
