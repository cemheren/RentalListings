import numpy as np
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

import re

import time
import sys
sys.path.insert(0, './data')
import jsonDataHandler as jDH
import RL_preprocessor as RL_prep
import RL_utils as RL_utils

# Import smtplib for the actual sending function
import smtplib

# Import the email modules we'll need
from email.mime.text import MIMEText


print '\n==> If There is no Change in Data Handling, You May Deactivate Data Handling or Feature Handling by Setting Booleans'
#########################################################
# Set BOOLEANS for DEACTIVATING !!!! @@@@@ !!!!!!
#########################################################
HANDLE_FEATURES = False
HANDLE_REMAINING_DATA = False
num_features_to_extract = 200

if HANDLE_FEATURES:
    print 'HANDLE_FEATURES status: ON'
    start = time.time()
    jDH.handle_features_only_and_picle_it(num_features_to_extract)
    end = time.time()
    print 'Feature Handling Time:', (end - start)
else:
    print 'HANDLE_FEATURES status: OFF'

if HANDLE_REMAINING_DATA:
    print 'HANDLE_REMAINING_DATA status: ON'
    start = time.time()
    jDH.handle_data_and_picle_it()
    end = time.time()
    print 'Data Handling Time:', (end - start)
else:
    print 'HANDLE_REMAINING_DATA status: OFF'


normalizableColumnResolver = jDH.get_normalizable_column_resolver()

print '\n==> Reading Pickle Files'
# Load Training & Test Data
x1_train = pickle.load(open('data/simple_train_inputs.pickle', 'rb'))
y1_train = pickle.load(open('data/simple_train_labels.pickle', 'rb'))

x1_test = pickle.load(open('data/simple_test_inputs.pickle', 'rb'))
ids_test = pickle.load(open('data/listing_ids.pickle', 'rb'))

x1_train_features = pickle.load(open('data/simple_train_inputs_features.pickle', 'rb'))
x1_test_features = pickle.load(open('data/simple_test_inputs_features.pickle', 'rb'))


print '\n==> Merging Features to Rest of the columns'
x1_train = np.append(x1_train, x1_train_features, axis=1)
x1_test = np.append(x1_test, x1_test_features, axis=1)


print '\n==> Removing Outliers From Training_Data'
##################################################
# Find Keep Mask and Apply IT to Training_Data and Training_Labels
##################################################
price_keep_mask = RL_prep.find_keep_mask_for_price( x1_train[:, normalizableColumnResolver['price']] )

x1_train = x1_train[price_keep_mask, :]
y1_train = y1_train[price_keep_mask, :]


print '\n==> Normalizing Given Columns'
##################################################
# Normalize Train and Test Together
##################################################
normalize_names = ['price', 'num_images', 'num_desc_words', 'days_passed', 'dist_1', 'dist_2', 'dist_3', 'dist_4', 'dist_5', 'price_per_bedroom']
(x1_train, x1_test) = RL_prep.normalize_cols(x1_train, x1_test, normalizableColumnResolver, normalize_names)


print '\n==> Starting to Train Model\n'
##################################################
# Train Model on Training_Data
##################################################
input_size = 36 + num_features_to_extract
hidden_size = 1024

model = Sequential()
#model.add(Dense(output_dim=hidden_size, input_dim=input_size, init='glorot_normal', activation='tanh'))
model.add(Dense(output_dim=hidden_size, input_dim=input_size, init='glorot_normal'))
model.add(keras.layers.advanced_activations.ELU(alpha=1.0))
model.add(Dropout(0.3))
model.add(Dense(output_dim=hidden_size, input_dim=hidden_size, init='glorot_normal'))
model.add(keras.layers.advanced_activations.ELU(alpha=1.0))
model.add(Dropout(0.3))

model.add(Dense(output_dim=3, input_dim=hidden_size, init='glorot_normal', W_regularizer='l1l2', activation='softmax'))

model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy', 'categorical_accuracy'])

hist = model.fit(x1_train, y1_train, validation_split=0.05, nb_epoch=10, batch_size=512, verbose=2)
# model.fit(x1_train, y1_train, validation_split=0.04, nb_epoch=500, batch_size=512, class_weight={0: 1.25, 1: 1.1, 2: 1.0}, verbose=2)


# Generate Trailer Metric String
metric_str = re.sub(r'[.]', '_', '%.3f' % (hist.history['val_loss'][-1]))

# Get File Names (model name is formed by number of hidden and input layer nodes)
fname_dictionary = jDH.get_model_and_submission_file_name_dictionary(input_size, hidden_size, metric_str)

# Save This Model
model.save(fname_dictionary['model_fname'])


print '\n==> Running Trained Model on Training Data'
predicted_classes_on_training_data = model.predict_classes(x1_train, verbose=2)
print '\nOriginal TD ClassHistogram:', np.histogram(jDH.convert_categorical_to_class3(y1_train), bins=[0, 1, 2, 3], density=True)
print 'Predicted TD ClassHistogram:', np.histogram(predicted_classes_on_training_data, bins=[0, 1, 2, 3], density=True)


print '\n==> Running Trained Model on Test Data'
##################################################
# Run This Model on Test Data
##################################################
results_test = model.predict(x1_test, verbose=2)


# Prepare Submission File
RL_utils.prepare_submission_file(fname_dictionary['submission_fname'], results_test, ids_test)




print '\n==> All Done'
