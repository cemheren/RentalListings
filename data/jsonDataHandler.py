import ijson
import json
import keras.utils.np_utils
import pickle
import numpy as np
import RL_encoders as RL_enc
import RL_preprocessor as RL_prep

import os



def handle_data_and_picle_it(num_features_to_extract):
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

    # Get Common Features from TRAIN DATA
    features_array = RL_enc.get_top_n_features(x, num_features_to_extract)


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

    # Create INPUT Matrix
    input_matrix = np.column_stack( (bath, bed, price, number_of_images, number_of_description_words) )

    # Add Features as Categorical
    for feature in features_array:
        f_input = [1 if feature[0] in v else 0 for v in x['features'].values()]
        input_matrix = np.c_[input_matrix, f_input]

    ################################################
    # Get Region Id Values
    ################################################
    # LatLong ==> Region Id ==> Categorical Encoding  (extra 24 inputs)
    # ZERO means outside
    region_xids = RL_enc.to_region_idx_encoding(x)
    # I need to convert it to categorical encoding
    region_xids = keras.utils.np_utils.to_categorical(region_xids)
    col_num = region_xids.shape[1]
    for ii in range(0, col_num):
        input_matrix = np.c_[input_matrix, region_xids[:, ii].tolist()]

    ################################################
    # Get LABEL Values
    ################################################
    interest = [interestResolver[v] for v in x['interest_level'].values()]
    print 'ClassHistogram:', np.histogram(interest)
    trainlabels = keras.utils.np_utils.to_categorical(interest)



    print 'Removing outliers from Training_Data'
    ################################################
    # Actual CLEANING Part Starts for Training Data
    ################################################
    # OUTLIER DELETION on <input_matrix> and <trainlabels>
    price_keep_mask = RL_prep.find_keep_mask_for_price(price)


    input_matrix = input_matrix[price_keep_mask, :]
    trainlabels = trainlabels[price_keep_mask, :]


    print 'Pickling Training_Data \t[Started]'
    ################################################
    # SAVE TRAIN DATA (INPUTS, LABELS)
    ################################################
    pickle.dump(input_matrix, open(path_prefix + 'simple_train_inputs.pickle', 'wb'))
    pickle.dump(trainlabels, open(path_prefix + 'simple_train_labels.pickle', 'wb'))
    print 'Pickling Training_Data \t[Finished]'




    ################################################
    ################################################


    print '\n\nStarting to prepare Test_Data'
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

    # Create TEST INPUT Matrix (Same Order with TRAINING DATA)
    test_input_matrix = np.column_stack( (test_bath, test_bed, test_price, test_number_of_images, test_number_of_description_words) )

    for feature in features_array:
        f_input = [1 if feature[0] in v else 0 for v in test_y['features'].values()]
        test_input_matrix = np.c_[test_input_matrix, f_input]

    # LatLong ==> Region Id ==> Categorical Encoding  (extra 24 inputs)
    # ZERO means outside
    test_region_xids = RL_enc.to_region_idx_encoding(test_y)
    # I need to convert it to categorical encoding
    test_region_xids = keras.utils.np_utils.to_categorical(test_region_xids, nb_classes=24)
    col_num = test_region_xids.shape[1]
    for ii in range(0, col_num):
        test_input_matrix = np.c_[test_input_matrix, test_region_xids[:, ii].tolist()]


    print 'Pickling Test_Data \t[Started]'
    ################################################
    # SAVE TEST DATA (INPUTS)
    ################################################
    pickle.dump(test_input_matrix, open(path_prefix + 'simple_test_inputs.pickle', 'wb'))
    print 'Pickling Test_Data \t[Finished]'

    # ToDo: One option can be just returning the variables (not saving). Time Cost seems not so big compared to training
    return 0




def get_normalizable_column_resolver():
    # If you add numerical columns, give a name and put the colm_id
    normalizableColumnResolver = dict()
    normalizableColumnResolver['bathrooms'] = 0
    normalizableColumnResolver['bedrooms'] = 1
    normalizableColumnResolver['price'] = 2
    normalizableColumnResolver['num_images'] = 3
    normalizableColumnResolver['num_desc_words'] = 4
    return normalizableColumnResolver


