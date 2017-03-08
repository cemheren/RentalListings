import ijson
import json
import keras.utils.np_utils
import pickle
import numpy as np
from RL_encoders import to_region_idx_encoding, get_top_n_features

f = open('test.json', 'r')
x = json.load(f)

testinputs = []

listing_ids = x['listing_id'].values()
pickle.dump(listing_ids, open('listing_ids.pickle', 'wb'))

bath = [v for v in x['bathrooms'].values()]
testinputs.append(bath)

bed = [v for v in x['bedrooms'].values()]
testinputs.append(bed)

price = [v/1000.0 for v in x['price'].values()]
testinputs.append(price)

number_of_images = [v.__len__()/10.0 for v in x['photos'].values()]
testinputs.append(number_of_images)

number_of_description_words = [v.__len__()/1000.0 for v in x['description'].values()]
testinputs.append(number_of_description_words)

features_array = pickle.load(open('features_array.pickle', 'r'))
for feature in features_array:
    f_input = [1 if feature[0] in v else 0 for v in x['features'].values()]
    testinputs.append(f_input)

# LatLong ==> Region Id ==> Categorical Encoding  (extra 24 inputs)
# ZERO means outside
region_xids = to_region_idx_encoding(x)
# I guess I need to convert it to categorical encoding
region_xids = keras.utils.np_utils.to_categorical(region_xids, nb_classes=24)
col_num = region_xids.shape[1]
for ii in range(0, col_num):
    testinputs.append(region_xids[:, ii].tolist())


# using only   bath,  bed,  interest,  price,  number_of_images,  region(lat,long)   so far.

pickle.dump(np.array(testinputs).T, open('simple_test_inputs.pickle', 'wb'))

# todo: pre process the data here. e.g. normalize fields, outlier detection, one hot encoding.
# todo: some fields can be simplified, like images -> just take the number of images to begin with
# todo: the data contains tags like 'pet friendly' these should effect the desirability of the apartment. find a way to encode them.
# todo: display_address field should also be included. similar to prev todo. find a way to encode them.

# for prefix, the_type, value in ijson.parse(open('test.json')):
#     print prefix, the_type, value
