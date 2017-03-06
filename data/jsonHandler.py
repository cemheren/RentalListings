import ijson;
import json;
import keras.utils.np_utils
import pickle
import numpy as np

interestResolver = dict()
interestResolver['medium'] = 1
interestResolver['low'] = 0
interestResolver['high'] = 2

f = open('train.json', 'r')
x = json.load(f)

traininputs = []
trainlabels = []

bath = [v for v in x['bathrooms'].values()]
traininputs.append(bath)

bed = [v for v in x['bedrooms'].values()]
traininputs.append(bed)

# created = x['created']

interest = [interestResolver[v] for v in x['interest_level'].values()]
h = np.histogram(interest, bins=[0, 1, 2, 3], density=True)
print(h)

interest = keras.utils.np_utils.to_categorical(interest)
trainlabels = interest

price = [v for v in x['price'].values()]
traininputs.append(price)

latitude = x['latitude']
longitude = x['longitude']

# using only bath, bed, interest, price so far.

pickle.dump(trainlabels, open('simple_train_labels.pickle', 'wb'))
pickle.dump(np.array(traininputs).T, open('simple_train_inputs.pickle', 'wb'))

# todo: pre process the data here. e.g. normalize fields, outlier detection, one hot encoding.
# todo: some fields can be simplified, like images -> just take the number of images to begin with
# todo: the data contains tags like 'pet friendly' these should effect the desirability of the apartment. find a way to encode them.
# todo: display_address field should also be included. similar to prev todo. find a way to encode them.

# for prefix, the_type, value in ijson.parse(open('test.json')):
#     print prefix, the_type, value
