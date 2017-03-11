import ijson
import json
import keras.utils.np_utils
import pickle
import numpy as np
from RL_encoders import to_region_idx_encoding, get_top_n_features
from RL_preprocessor import find_keep_mask_for_price, normalize_col

interestResolver = dict()
interestResolver['high'] = 0
interestResolver['medium'] = 1
interestResolver['low'] = 2



f = open('train.json', 'r')
x = json.load(f)

###########################
# Output CLASS Values
###########################
interest = [interestResolver[v] for v in x['interest_level'].values()]
print(np.histogram(interest, bins=[0, 1, 2, 3], density=True))
trainlabels = keras.utils.np_utils.to_categorical(interest)


###########################
# Input Variables
###########################
# Numerical Columns
bath = [v for v in x['bathrooms'].values()]
bed = [v for v in x['bedrooms'].values()]
price = [v for v in x['price'].values()]
number_of_images = [v.__len__()/10.0 for v in x['photos'].values()]
number_of_description_words = [v.__len__()/1000.0 for v in x['description'].values()]


# INPUT Matrix  (EASY to manipulate as matrix (NDArray) for trimming and normalizing)
input_matrix = np.column_stack( (bath, bed, price, number_of_images, number_of_description_words) )


# Features
features_array = get_top_n_features(x, 50)
pickle.dump(features_array, open('features_array.pickle', 'wb'))

for feature in features_array:
    f_input = [1 if feature[0] in v else 0 for v in x['features'].values()]
    input_matrix = np.c_[input_matrix, f_input]

# LatLong ==> Region Id ==> Categorical Encoding  (extra 24 inputs)
# ZERO means outside
region_xids = to_region_idx_encoding(x)
pickle.dump(region_xids, open('region_xids.pickle', 'wb'))

# I guess I need to convert it to categorical encoding
region_xids = keras.utils.np_utils.to_categorical(region_xids)
col_num = region_xids.shape[1]
for ii in range(0, col_num):
    input_matrix = np.c_[input_matrix, region_xids[:, ii].tolist()]



###########################
# Actual Cleaning Part Starts
###########################

# Outlier DELETION on <input_matrix> and <trainlabels>
price_keep_mask = find_keep_mask_for_price(price)


input_matrix = input_matrix[price_keep_mask, :]
trainlabels = trainlabels[price_keep_mask, :]

# Normalize Column(s)
_PRICE_COL_ = 2
input_matrix_norm = normalize_col(input_matrix, _PRICE_COL_)



pickle.dump(trainlabels, open('simple_train_labels.pickle', 'wb'))
pickle.dump(input_matrix, open('simple_train_inputs.pickle', 'wb'))

# todo: pre process the data here. e.g. normalize fields, outlier detection, one hot encoding.
# todo: some fields can be simplified, like images -> just take the number of images to begin with
# todo: the data contains tags like 'pet friendly' these should effect the desirability of the apartment. find a way to encode them.
# todo: display_address field should also be included. similar to prev todo. find a way to encode them.

# for prefix, the_type, value in ijson.parse(open('test.json')):
#     print prefix, the_type, value
