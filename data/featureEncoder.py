import operator
import ijson
import json

f = open('train.json', 'r')
json_decode = json.load(f)

features = json_decode['features']


feature_list = []
for item in features:
    this_items_features = features[item]
    for single_feature in this_items_features:
        feature_list.append(single_feature)

set_of_features = set(feature_list)
unique_feature_list = list(set_of_features)
unique_feature_idx_dict = {unique_feature_list[ii]: ii for ii in range(0, len(unique_feature_list))}

num_unique_features = unique_feature_list.__len__()
print num_unique_features

# Now create new dictionary with encoding
encoded_features_dict = dict()
for item in features:
    this_items_features = features[item]
    indices_to_set_1_for_this_item = []
    for single_feature in this_items_features:
        indices_to_set_1_for_this_item.append(unique_feature_idx_dict[single_feature])
    encoded_features = [0] * num_unique_features
    for idx_to_set_1 in indices_to_set_1_for_this_item:
        encoded_features[idx_to_set_1] = 1
        encoded_features_dict[item] = encoded_features

# encoded_features is ready to write
kkey = '4'
print encoded_features_dict[kkey][160:190]
