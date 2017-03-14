import ijson
import json
import keras.utils.np_utils
import pickle
import numpy as np
import RL_encoders as RL_enc
import RL_preprocessor as RL_prep
import os
import string
import re
import random
from datetime import datetime as DT


def handle_features_only_and_picle_it(num_features_to_extract):
    if os.path.exists('train.json'):
        path_prefix = ''
    elif os.path.exists('./data/train.json'):
        path_prefix = './data/'
    else:
        print('\nError in Path')

    f_train = open(path_prefix + 'train.json', 'r')
    x = json.load(f_train)

    f_test = open(path_prefix + 'test.json', 'r')
    test_y = json.load(f_test)

    # Get Common Features from TRAIN DATA
    features_array = RL_enc.get_top_n_features(x, num_features_to_extract)

    x_features = x['features']

    # Training Data PART STARTS!
    feature_matrix = np.zeros(shape=(x_features.__len__(), num_features_to_extract))
    ii = 0
    # Add Features as Categorical
    for feature in features_array:
        f_input = [1 if feature[0] in map(lambda k: re.sub(r'[^\w]', '', k.translate(string.punctuation).lower()), v) else 0 for v in x_features.values()]
        feature_matrix[:,ii] = f_input
        # Bu da kolona liste vererek yapilabilen bir atamaymis
        ii += 1

    print 'Pickling Training_Data (Encoded) Features \t[Started]'
    ################################################
    # SAVE TRAIN DATA (INPUTS, LABELS)
    ################################################
    pickle.dump(feature_matrix, open(path_prefix + 'simple_train_inputs_features.pickle', 'wb'))
    print 'Pickling Training_Data (Encoded) Features \t[Finished]'

    # Test Data Part STARTS!
    test_y_features = test_y['features']

    test_feature_matrix = np.zeros(shape=(test_y_features.__len__(), num_features_to_extract))
    ii = 0
    # Add Features as Categorical
    for feature in features_array:
        f_input = [1 if feature[0] in map(lambda k: re.sub(r'[^\w]', '', k.translate(string.punctuation).lower()), v) else 0 for v in test_y_features.values()]
        test_feature_matrix[:, ii] = f_input
        # Bu da kolona liste vererek yapilabilen bir atamaymis
        ii += 1

    print 'Pickling Test_Data (Encoded) Features \t[Started]'
    ################################################
    # SAVE TRAIN DATA (INPUTS, LABELS)
    ################################################
    pickle.dump(test_feature_matrix, open(path_prefix + 'simple_test_inputs_features.pickle', 'wb'))
    print 'Pickling Test_Data (Encoded) Features \t[Finished]'




def handle_data_and_picle_it():
    interestResolver = dict()
    interestResolver['high'] = 0
    interestResolver['medium'] = 1
    interestResolver['low'] = 2

    if os.path.exists('train.json'):
        path_prefix = ''
    elif os.path.exists('./data/train.json'):
        path_prefix = './data/'
    else:
        print('\nError in Path')

    f_train = open(path_prefix + 'train.json', 'r')
    x = json.load(f_train)

    f_test = open(path_prefix + 'test.json', 'r')
    test_y = json.load(f_test)


    print '\nStarting to prepare Training_Data'
    ################################################
    # Train DATA Handling Starts
    ################################################
    ################################################
    # Input Variables
    ################################################
    # Numerical Columns
    bath = [v for v in x['bathrooms'].values()]
    bed = [v for v in x['bedrooms'].values()]
    price = [float(v) for v in x['price'].values()]
    number_of_images = [float(v.__len__()) for v in x['photos'].values()]
    number_of_description_words = [float(v.__len__()) for v in x['description'].values()]

    date_format = "%Y-%m-%d %H:%M:%S"
    fixed_date = DT.strptime('2017-03-01 15:30:30', date_format)
    days_passed = [(fixed_date - DT.strptime(v, date_format)).days for v in x['created'].values()]

    # Create INPUT Matrix
    input_matrix = np.column_stack( (bath, bed, price, number_of_images, number_of_description_words, days_passed) )

    ################################################
    # Get Distances to 5 fixed points
    ################################################
    x_distances = RL_enc.get_distances(x)
    for ii in range(0, x_distances.shape[1]):
        input_matrix = np.c_[input_matrix, x_distances[:, ii].tolist()]

    ################################################
    # Get Region Id Values
    ################################################
    # LatLong ==> Region Id ==> Categorical Encoding  (extra 24 inputs)
    # ZERO means outside
    region_xids = RL_enc.to_region_idx_encoding(x)
    for ii in range(0, region_xids.shape[1]):
        input_matrix = np.c_[input_matrix, region_xids[:, ii].tolist()]

    ################################################
    # Get LABEL Values
    ################################################
    interest = [interestResolver[v] for v in x['interest_level'].values()]
    trainlabels = keras.utils.np_utils.to_categorical(interest)

    # Buna Gerek Yok Sanki ARTIK cunku Outlier ayiklamayi 1 ust seviyeye tasidim !!!!
    # pickle.dump(trainlabels, open(path_prefix + 'simple_train_labels_with_outliers.pickle', 'wb'))

    ################################################
    # Get Description based predictions of other model
    ################################################
    # description_train_array = pickle.load(open("./description/description_train_results.pickle", 'rb'))
    # input_matrix = np.c_[input_matrix, description_train_array]


    print 'Pickling Training_Data \t[Started]'
    ################################################
    # SAVE TRAIN DATA (INPUTS, LABELS)
    ################################################
    pickle.dump(input_matrix, open(path_prefix + 'simple_train_inputs.pickle', 'wb'))
    pickle.dump(trainlabels, open(path_prefix + 'simple_train_labels.pickle', 'wb'))
    print 'Pickling Training_Data \t[Finished]'


    ################################################
    # Continue to Test Data Part
    ################################################


    print '\nStarting to prepare Test_Data'
    ################################################
    # Test DATA Handling Starts
    ################################################
    listing_ids = test_y['listing_id'].values()
    pickle.dump(listing_ids, open(path_prefix + 'listing_ids.pickle', 'wb'))

    test_bath = [v for v in test_y['bathrooms'].values()]
    test_bed = [v for v in test_y['bedrooms'].values()]
    test_price = [float(v) for v in test_y['price'].values()]
    test_number_of_images = [v.__len__() for v in test_y['photos'].values()]
    test_number_of_description_words = [v.__len__() for v in test_y['description'].values()]

    test_days_passed = [(fixed_date - DT.strptime(v, date_format)).days for v in test_y['created'].values()]

    # Create TEST INPUT Matrix (Same Order with TRAINING DATA)
    test_input_matrix = np.column_stack( (test_bath, test_bed, test_price, test_number_of_images, test_number_of_description_words, test_days_passed) )

    ################################################
    # Get Distances to 5 fixed points
    ################################################
    test_y_distances = RL_enc.get_distances(test_y)
    for ii in range(0, test_y_distances.shape[1]):
        test_input_matrix = np.c_[test_input_matrix, test_y_distances[:, ii].tolist()]


    ################################################
    # Get Region Id Values
    ################################################
    # LatLong ==> Region Id ==> Categorical Encoding  (extra 24 inputs)
    # ZERO means outside
    test_region_xids = RL_enc.to_region_idx_encoding(test_y)
    for ii in range(0, test_region_xids.shape[1]):
        test_input_matrix = np.c_[test_input_matrix, test_region_xids[:, ii].tolist()]

    ################################################
    # Get Description based predictions of other model
    ################################################
    # description_test_array = pickle.load(open("./description/description_test_results.pickle", 'rb'))
    # test_input_matrix = np.c_[test_input_matrix, description_test_array]


    print 'Pickling Test_Data \t[Started]'
    ################################################
    # SAVE TEST DATA (INPUTS)
    ################################################
    pickle.dump(test_input_matrix, open(path_prefix + 'simple_test_inputs.pickle', 'wb'))
    print 'Pickling Test_Data \t[Finished]'

    return 0



def convert_categorical_to_class3(cat_vector):
    num_entries = cat_vector.shape[0]
    cat_converted = [0] * num_entries
    for ii in range(0, num_entries):
        if cat_vector[ii][1] == 1:
            cat_converted[ii] = 1
        elif cat_vector[ii][2] == 1:
            cat_converted[ii] = 2
    return cat_converted




def get_normalizable_column_resolver():
    # If you add numerical columns, give a name and put the colm_id
    normalizableColumnResolver = dict()
    normalizableColumnResolver['bathrooms'] = 0
    normalizableColumnResolver['bedrooms'] = 1
    normalizableColumnResolver['price'] = 2
    normalizableColumnResolver['num_images'] = 3
    normalizableColumnResolver['num_desc_words'] = 4
    normalizableColumnResolver['days_passed'] = 5
    normalizableColumnResolver['dist_1'] = 6
    normalizableColumnResolver['dist_2'] = 7
    normalizableColumnResolver['dist_3'] = 8
    normalizableColumnResolver['dist_4'] = 9
    normalizableColumnResolver['dist_5'] = 10
    return normalizableColumnResolver


def get_model_and_submission_file_name_dictionary(input_size, hidden_size):
    fname_dictionary = dict()
    random_file_postfix = ''.join(random.choice(string.lowercase) for x in range(5))
    model_name = 'mi0' + str(input_size) + '_mh0' + str(hidden_size) + '_' + DT.now().strftime("d%d_t%H_%M_")
    fname_dictionary['model_fname'] = 'samp_model_' + model_name + random_file_postfix + '.km'
    fname_dictionary['submission_fname'] = 'submission_' + model_name + random_file_postfix + '.txt'
    return fname_dictionary
