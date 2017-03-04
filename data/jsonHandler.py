import ijson;
import json;

f = open('train.json', 'r')
x = json.load(f)

y = x

bath = x['bathrooms']
bed = x['bedrooms']
created = x['created']
interest = x['interest_level']
price = x['price']

# todo: pre process the data here. e.g. normalize fields, outlier detection, one hot encoding.
# todo: some fields can be simplified, like images -> just take the number of images to begin with
# todo: the data contains tags like 'pet friendly' these should effect the desirability of the apartment. find a way to encode them.

# for prefix, the_type, value in ijson.parse(open('test.json')):
#     print prefix, the_type, value
