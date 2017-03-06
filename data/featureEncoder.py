import ijson
import json
from RL_encoders import to_one_cold_feature_encoding

f_train = open('train.json', 'r')
# IMPORTANT: dumps should be created (you may download it directly or create at your local by running dumper)
f_uniqfeature_dump = open('ufeature_to_index_dictionary.json', 'r')

json_decode_train = json.load(f_train)
features = json_decode_train['features']

# This json file contains key:uniquefeatures  value:fixedID
json_decode_uniqfeature_dump = json.load(f_uniqfeature_dump)

# This dictionary contains key:listing_id (I guess)  value:feature_indicator_vector
encoded_features_dict = to_one_cold_feature_encoding(features, json_decode_uniqfeature_dump)

# encoded_features is ready to write
kkey = '4'
print encoded_features_dict[kkey][160:190]
