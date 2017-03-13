import numpy as np
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

import sys
sys.path.insert(0, './data')
import jsonDataHandler as jDH
import RL_preprocessor as RL_prep


print '\n==> If There is no Change in Data Handling, You May Comment-out Data Handling'
num_features_to_extract = 100
# For data handling --> next line
jDH.handle_data_and_picle_it(num_features_to_extract)
normalizableColumnResolver = jDH.get_normalizable_column_resolver()

print '\n==> Reading Pickle Files'
# Load Training & Test Data
x1_train = pickle.load(open('data/simple_train_inputs.pickle', 'rb'))
y1_train = pickle.load(open('data/simple_train_labels.pickle', 'rb'))

x1_test = pickle.load(open('data/simple_test_inputs.pickle', 'rb'))
ids_test = pickle.load(open('data/listing_ids.pickle', 'rb'))


print '\n==> Normalizing Given Columns'
##################################################
# Normalize Train and Test Together
##################################################
(x1_train, x1_test) = RL_prep.normalize_cols(x1_train, x1_test, normalizableColumnResolver, ['price', 'num_images', 'num_desc_words', 'days_passed'])


print '\n==> Starting to Train Model\n'
##################################################
# Train Model on Train Data
##################################################
input_size = 30 + num_features_to_extract
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

# Get File Names (model name is formed by number of hidden and input layer nodes)
fname_dictionary = jDH.get_model_and_submission_file_name_dictionary(input_size, hidden_size)
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
result = model.predict(x1_test, verbose=2)

print '\n==> Preparing Submission File'
submission_file = []
submission_file.append("listing_id,high,medium,low")

for i in range(len(result)):
    current_id = ids_test[i]
    r = result[i]
    line = str(current_id) + ',' + ",".join(map(str, r))
    submission_file.append(line)

# Save Output of Testing
sub_file = open(fname_dictionary['submission_fname'], 'w')
for item in submission_file:
    sub_file.write("%s\n" % item)

print '\n==> All Done'
